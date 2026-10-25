"""Compare YOLO11n/s inference resolution and confidence thresholds.

Uses saved models (no training) and the chosen dataset's VALIDATION split.
Runs prediction once per (model, image size), then filters the same
post-NMS detections at several confidence levels. Counts true/false
positives and false negatives by greedy, class-aware, one-to-one
matching at a configurable ground-truth IoU threshold.

Run from the repository root:
    python src/optimise_inference.py --dataset 4
    python src/optimise_inference.py --dataset 4 --models n s --batch 2
    python src/optimise_inference.py --dataset 4 --compute-map

Results (tracked by Git):
    results/inference_optimisation/<dataset_slug>/optimisation_results.csv
    results/inference_optimisation/<dataset_slug>/per_class_results.csv
    results/inference_optimisation/<dataset_slug>/confidence_comparison.png
    results/inference_optimisation/<dataset_slug>/resolution_comparison.png
    results/inference_optimisation/<dataset_slug>/experiment_metadata.json

Note: threshold precision and recall here are *micro-averaged across boxes*,
not Ultralytics' mean per-class, max-F1 precision/recall from model.val().
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Save plots without opening windows on Windows/CI.
import matplotlib.pyplot as plt
import yaml
from ultralytics import YOLO

from experiment_config import (
    PROJECT_ROOT,
    RUNS_DIR,
    find_training_checkpoint,
    select_dataset,
    training_run_base,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
DEFAULT_SIZES = (320, 480, 640)
DEFAULT_THRESHOLDS = (0.10, 0.25, 0.40, 0.60)


def parse_args() -> argparse.Namespace:
    """Define repeatable command-line experiment settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-name", type=str, default=None, help="Custom folder name for optimisation results.")
    parser.add_argument("--dataset", type=int, choices=(1, 2, 3, 4), default=4)
    parser.add_argument("--models", nargs="+", choices=("n", "s"), default=["n", "s"])
    parser.add_argument("--sizes", nargs="+", type=int, default=list(DEFAULT_SIZES))
    parser.add_argument("--conf", nargs="+", type=float, default=list(DEFAULT_THRESHOLDS))
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0", help="GPU '0' or 'cpu'.")
    parser.add_argument("--match-iou", type=float, default=0.50,
                        help="Minimum IoU for a correct predicted box (default 0.50).")
    parser.add_argument("--nms-iou", type=float, default=0.70,
                        help="Ultralytics non-maximum suppression IoU (default 0.70).")
    parser.add_argument("--run-n", help="Exact YOLO11n training run folder (optional).")
    parser.add_argument("--run-s", help="Exact YOLO11s training run folder (optional).")
    parser.add_argument("--compute-map", action="store_true",
                        help="Also run model.val at each resolution for mAP50/mAP50-95. Slower.")
    args = parser.parse_args()
    args.models = list(dict.fromkeys(args.models))
    args.sizes = sorted(set(args.sizes))
    args.conf = sorted(set(args.conf))
    if not args.sizes or any(s <= 0 or s % 32 != 0 for s in args.sizes):
        parser.error("--sizes must contain positive multiples of 32 (e.g. 320 480 640).")
    if not args.conf or any(not math.isfinite(c) or not (0 < c < 1) for c in args.conf):
        parser.error("--conf values must be between 0 and 1 (exclusive).")
    if args.batch <= 0:
        parser.error("--batch must be positive.")
    for key in ("match_iou", "nms_iou"):
        if not math.isfinite(getattr(args, key)) or not (0 < getattr(args, key) <= 1):
            parser.error(f"--{key.replace('_', '-')} must be in (0, 1].")
    return args


def read_dataset_info(dataset_yaml: Path) -> tuple[Path, list[str]]:
    """Resolve the validation image directory and ordered class names."""
    meta = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    names = meta.get("names")
    if isinstance(names, list):
        class_names = [str(name) for name in names]
    elif isinstance(names, dict):
        indexed_names = {int(k): str(v) for k, v in names.items()}
        if set(indexed_names) != set(range(len(indexed_names))):
            raise ValueError(f"Non-consecutive class IDs in {dataset_yaml}")
        class_names = [indexed_names[i] for i in range(len(indexed_names))]
    else:
        raise ValueError(f"Dataset YAML has no valid 'names': {dataset_yaml}")

    yaml_parent = dataset_yaml.resolve().parent
    path_entry = meta.get("path")
    if path_entry:
        candidate = Path(path_entry)
        if not candidate.is_absolute():
            # A path in a YOLO YAML may be rooted at the dataset or project.
            local = yaml_parent / candidate
            project = PROJECT_ROOT / candidate
            candidate = local if local.exists() else project
        base = candidate.resolve()
    else:
        base = yaml_parent
    val_entry = meta.get("val")
    if not isinstance(val_entry, str):
        raise ValueError("This script expects one validation image folder in YAML 'val'.")
    val_dir = Path(val_entry)
    if not val_dir.is_absolute():
        val_dir = base / val_dir
    val_dir = val_dir.resolve()
    if not val_dir.is_dir():
        raise FileNotFoundError(f"Validation image directory not found: {val_dir}")
    return val_dir, class_names


def read_ground_truth(label_path: Path, width: int, height: int, n_classes: int):
    """Return [(class_id, pixel_xyxy), ...] from YOLO-normalised labels."""
    if not label_path.is_file():
        raise FileNotFoundError(f"Missing label: {label_path}. Run data_check.py first.")
    labels = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue  # Valid, intentionally empty labels are background images.
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Invalid YOLO label at {label_path}:{line_number}: {line!r}")
        cls_float, x, y, w, h = map(float, parts)
        if not all(math.isfinite(v) for v in (cls_float, x, y, w, h)):
            raise ValueError(f"Non-finite label at {label_path}:{line_number}")
        if not cls_float.is_integer() or not (0 <= int(cls_float) < n_classes):
            raise ValueError(f"Invalid class ID at {label_path}:{line_number}")
        if w <= 0 or h <= 0 or min(x-w/2, y-h/2) < -1e-5 or max(x+w/2, y+h/2) > 1+1e-5:
            raise ValueError(f"Invalid bounding box at {label_path}:{line_number}")
        labels.append((int(cls_float), (
            (x - w / 2) * width, (y - h / 2) * height,
            (x + w / 2) * width, (y + h / 2) * height,
        )))
    return labels


def box_iou(a, b) -> float:
    """Intersection over union of two pixel-coordinate xyxy boxes."""
    overlap_w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    overlap_h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = overlap_w * overlap_h
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def match_detections(truth, detections, conf_threshold: float, iou_threshold: float, n_classes: int):
    """Count class-aware TP/FP/FN, matching high-confidence predictions first.

    Each ground-truth box can be matched at most once. Wrong-class detections
    count as an FP for the predicted class and an FN for the actual class.
    """
    gt_by_class = {cls: [box for c, box in truth if c == cls] for cls in range(n_classes)}
    pred_by_class = {cls: [] for cls in range(n_classes)}
    for cls, score, box in detections:
        if score >= conf_threshold:
            pred_by_class[cls].append((score, box))
    counts = {}
    for cls in range(n_classes):
        gt = gt_by_class[cls]
        used = set()
        tp = fp = 0
        for _, pred in sorted(pred_by_class[cls], key=lambda item: item[0], reverse=True):
            best_index = -1
            best_iou = iou_threshold
            for idx, actual in enumerate(gt):
                if idx in used:
                    continue
                overlap = box_iou(pred, actual)
                if overlap >= best_iou:
                    best_iou = overlap
                    best_index = idx
            if best_index >= 0:
                used.add(best_index)
                tp += 1
            else:
                fp += 1
        counts[cls] = Counter(tp=tp, fp=fp, fn=len(gt)-tp)
    return counts


def measures(counts: Counter) -> dict[str, float | int]:
    """Compute fixed-threshold micro detection precision, recall and F1."""
    tp, fp, fn = (int(counts[k]) for k in ("tp", "fp", "fn"))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def save_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    """Write a repeatable CSV with fixed column ordering."""
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def plot_confidence(rows: list[dict], path: Path) -> None:
    """Plot fixed-threshold detection precision and recall curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    pairs = sorted({(row["model"], row["imgsz"]) for row in rows})
    for model, size in pairs:
        sample = sorted(
            (row for row in rows if row["model"] == model and row["imgsz"] == size),
            key=lambda row: row["confidence"],
        )
        for axis, metric in zip(axes, ("precision", "recall")):
            axis.plot([r["confidence"] for r in sample], [r[metric] for r in sample],
                      marker="o", label=f"{model} / {size}px")
    for axis, title in zip(axes, ("Precision", "Recall")):
        axis.set(title=title + " vs confidence threshold", xlabel="Confidence threshold",
                 ylabel=title + " (fixed IoU matching)", ylim=(-0.02, 1.02))
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_resolution(rows: list[dict], path: Path, reference_conf: float) -> None:
    """Plot detection quality and inference time at a single threshold."""
    chosen = [row for row in rows if row["confidence"] == reference_conf]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for model in sorted({row["model"] for row in chosen}):
        sample = sorted((row for row in chosen if row["model"] == model), key=lambda r: r["imgsz"])
        for axis, key in zip(axes, ("precision", "recall", "inference_ms_per_image")):
            axis.plot([r["imgsz"] for r in sample], [r[key] for r in sample], marker="o", label=model)
    for axis, title in zip(axes, ("Precision", "Recall", "Inference time")):
        axis.set(xlabel="Input size (pixels)", title=title + f" at conf={reference_conf:g}")
        axis.grid(True, alpha=0.3)
        axis.legend()
    axes[0].set(ylabel="Precision", ylim=(-0.02, 1.02))
    axes[1].set(ylabel="Recall", ylim=(-0.02, 1.02))
    axes[2].set_ylabel("Model inference (ms/image)")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_experiment(args: argparse.Namespace) -> Path:
    """Perform the selected validation grid without changing saved weights."""
    number, yaml_path, slug, description = select_dataset(args.dataset)
    val_dir, class_names = read_dataset_info(yaml_path)
    images = sorted(p for p in val_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise ValueError(f"No images in {val_dir}")
    stems = [p.stem for p in images]
    if len(set(stems)) != len(stems):
        raise ValueError("Images share a filename stem, so labels cannot be matched unambiguously.")
    # Standard YOLO directory convention: valid/images -> valid/labels.
    if val_dir.name != "images":
        raise ValueError("Expected the validation image directory to be named 'images'.")
    label_dir = val_dir.parent / "labels"
    if not label_dir.is_dir():
        raise FileNotFoundError(f"Validation label directory not found: {label_dir}")

    folder_name = args.output_name or slug
    out_dir = PROJECT_ROOT / "results" / "inference_optimisation" / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    class_rows = []
    used_checkpoints = {}
    print(f"Dataset {number}: {description}")
    print(f"Validation set: {len(images)} images from {val_dir}")
    print(f"Testing {len(args.models)} models x {len(args.sizes)} sizes x "
          f"{len(args.conf)} confidence thresholds = "
          f"{len(args.models)*len(args.sizes)*len(args.conf)} combinations.")

    for size_id in args.models:
        base = training_run_base(size_id, slug)
        requested = args.run_n if size_id == "n" else args.run_s
        run_dir, checkpoint = find_training_checkpoint(base, requested)
        model = YOLO(str(checkpoint))
        actual_names = [str(model.names[i]) for i in range(len(model.names))]
        if actual_names != class_names:
            raise ValueError(f"Class mismatch in {checkpoint}: {actual_names} != {class_names}")
        name = f"yolo11{size_id}"
        used_checkpoints[name] = {"run": run_dir.name, "weights": str(checkpoint)}
        print(f"\n{name}: {checkpoint}")

        for size in args.sizes:
            # Predict only once per size. Threshold filtering below is inexpensive;
            # object detection was performed using the lowest chosen confidence.
            metrics_by_conf = {c: {i: Counter() for i in range(len(class_names))} for c in args.conf}
            speed_values = []
            seen = set()
            start = time.perf_counter()
            stream = model.predict(
                source=str(val_dir), imgsz=size, conf=min(args.conf),
                iou=args.nms_iou, max_det=300, batch=args.batch,
                device=args.device, rect=False, stream=True,
                save=False, verbose=False,
            )
            for result in stream:
                source = Path(result.path)
                if source.stem not in stems:
                    raise ValueError(f"Unexpected predicted image: {source}")
                if source.stem in seen:
                    raise ValueError(f"Duplicate predicted image: {source}")
                seen.add(source.stem)
                height, width = result.orig_shape
                truth = read_ground_truth(label_dir / (source.stem + ".txt"),
                                          width, height, len(class_names))
                boxes = result.boxes
                detections = []
                if boxes is not None and len(boxes):
                    xyxy = boxes.xyxy.cpu().numpy()
                    classes = boxes.cls.cpu().numpy().astype(int)
                    scores = boxes.conf.cpu().numpy()
                    for cls, conf, box in zip(classes, scores, xyxy):
                        if not (0 <= int(cls) < len(class_names)):
                            raise ValueError(f"Unexpected prediction class {cls} in {source}")
                        detections.append((int(cls), float(conf), tuple(map(float, box))))
                if result.speed and "inference" in result.speed:
                    speed_values.append(float(result.speed["inference"]))
                for threshold in args.conf:
                    counts = match_detections(truth, detections, threshold,
                                              args.match_iou, len(class_names))
                    for idx, count in counts.items():
                        metrics_by_conf[threshold][idx].update(count)
            elapsed = time.perf_counter() - start
            if len(seen) != len(images):
                raise RuntimeError(f"Predicted {len(seen)} images, expected {len(images)}")
            inf_speed = sum(speed_values) / len(speed_values) if speed_values else math.nan
            print(f"  {size}px: {len(seen)} images, {inf_speed:.2f} ms/image forward pass, "
                  f"{elapsed:.1f}s wall time")

            map50 = map50_95 = math.nan
            if args.compute_map:
                print(f"  Calculating Ultralytics mAP at {size}px (conf=0.001)...")
                val_metrics = model.val(
                    data=str(yaml_path), split="val", imgsz=size,
                    batch=args.batch, device=args.device, workers=0,
                    conf=0.001, iou=args.nms_iou, rect=False, plots=False,
                    project=str(RUNS_DIR), name=f"{run_dir.name}_map_{size}",
                    exist_ok=False, verbose=False,
                )
                map50 = float(val_metrics.box.map50)
                map50_95 = float(val_metrics.box.map)

            for threshold in args.conf:
                total = Counter()
                for idx, counts in metrics_by_conf[threshold].items():
                    total.update(counts)
                    class_rows.append({
                        "model": name, "training_run": run_dir.name,
                        "dataset": number, "imgsz": size,
                        "confidence": threshold, "class_id": idx,
                        "class_name": class_names[idx], **measures(counts),
                    })
                summary_rows.append({
                    "model": name, "training_run": run_dir.name,
                    "dataset": number, "imgsz": size, "confidence": threshold,
                    "images": len(images), **measures(total),
                    "inference_ms_per_image": round(inf_speed, 4),
                    "wall_seconds": round(elapsed, 3),
                    "map50_low_conf": map50,
                    "map50_95_low_conf": map50_95,
                })

    summary_columns = ["model", "training_run", "dataset", "imgsz", "confidence",
                       "images", "tp", "fp", "fn", "precision", "recall", "f1",
                       "inference_ms_per_image", "wall_seconds", "map50_low_conf", "map50_95_low_conf"]
    class_columns = ["model", "training_run", "dataset", "imgsz", "confidence",
                     "class_id", "class_name", "tp", "fp", "fn", "precision", "recall", "f1"]
    save_csv(out_dir / "optimisation_results.csv", summary_rows, summary_columns)
    save_csv(out_dir / "per_class_results.csv", class_rows, class_columns)
    plot_confidence(summary_rows, out_dir / "confidence_comparison.png")
    reference_conf = min(args.conf, key=lambda c: abs(c - 0.25))
    plot_resolution(summary_rows, out_dir / "resolution_comparison.png", reference_conf)
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_number": number,
        "dataset_yaml": str(yaml_path),
        "dataset_description": description,
        "evaluation_split": "val", "images": len(images),
        "class_names": class_names,
        "trained_checkpoints": used_checkpoints,
        "models": args.models, "sizes": args.sizes, "confidence_thresholds": args.conf,
        "batch_size": args.batch, "device": args.device,
        "gt_match_iou": args.match_iou, "nms_iou": args.nms_iou,
        "predict_rect": False, "max_det": 300,
        "mAP_evaluated": args.compute_map,
        "speed_measurement": "Mean Ultralytics Results.speed['inference'] (forward-pass ms/image). "
                             "Same value repeated across thresholds for a given model/size.",
        "matching_rule": "Class-aware, highest-confidence-first, one-to-one box matching at ground-truth IoU>=match_iou; "
                         "micro precision/recall, including negative images.",
        "note": "Validation-only; do not use test data for configuration selection. "
                "mAP uses separate low-conf validation and is not threshold-specific. "
                "Repeating optimisation overwrites the CSV/plots under this dataset slug.",
    }
    (out_dir / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nSaved {len(summary_rows)} configurations to {out_dir / 'optimisation_results.csv'}")
    print(f"Per-class results: {out_dir / 'per_class_results.csv'}")
    print(f"Graphs and metadata: {out_dir}")
    print("TIP: Choose the final configuration using validation data; reserve the test set.")
    return out_dir


def main() -> None:
    """Entry point for command-line experiments."""
    run_experiment(parse_args())


if __name__ == "__main__":
    main()
