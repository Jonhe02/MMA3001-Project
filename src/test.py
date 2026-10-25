"""One-time held-out test of the selected pork-packaging detector.

The final configuration was chosen on the VALIDATION split and is intentionally
fixed here. This script runs only on dataset 4's TEST split, does not train, and
refuses to overwrite an existing report.

From the repository root:
    python src/test.py

Outputs are kept in results/final_test/yolo11s_2class_v2_640px_conf020/.
Requires experiment_config.py and optimise_inference.py in the same src/ folder.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml
from ultralytics import YOLO

from experiment_config import PROJECT_ROOT, RUNS_DIR, find_training_checkpoint, select_dataset, training_run_base
from optimise_inference import IMAGE_EXTENSIONS, match_detections, measures, read_ground_truth

# Freeze these settings BEFORE evaluating the held-out test images.
DATASET_NUMBER = 4
MODEL_SIZE = "s"
TRAINING_RUN = "yolo11s_2class_v2"
IMAGE_SIZE = 640
CONFIDENCE = 0.20
MATCH_IOU = 0.50  # A prediction is correct when class matches and box IoU >= 0.50.
NMS_IOU = 0.70
BATCH = 4
DEVICE = "0"   # Change to "cpu" only if CUDA is unavailable; record the change.
MAX_DETECTIONS = 300
RESULTS_SUBDIR = "yolo11s_2class_v2_640px_conf020"


def locate_test_images(dataset_yaml: Path) -> tuple[Path, list[str]]:
    """Read the dataset YAML to locate the test split and its ordered classes."""
    meta = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    names = meta.get("names")
    if isinstance(names, list):
        class_names = [str(v) for v in names]
    elif isinstance(names, dict):
        by_id = {int(k): str(v) for k, v in names.items()}
        if set(by_id) != set(range(len(by_id))):
            raise ValueError("Non-consecutive class IDs in data.yaml")
        class_names = [by_id[i] for i in range(len(by_id))]
    else:
        raise ValueError("data.yaml has no valid 'names' mapping")

    base = dataset_yaml.resolve().parent
    path_entry = meta.get("path")
    if path_entry:
        candidate = Path(str(path_entry))
        if not candidate.is_absolute():
            local = base / candidate
            project_relative = PROJECT_ROOT / candidate
            candidate = local if local.exists() else project_relative
        base = candidate.resolve()

    test_entry = meta.get("test")
    if not isinstance(test_entry, str) or not test_entry.strip():
        raise ValueError("Dataset YAML must specify a 'test' images directory")
    test_dir = Path(test_entry)
    if not test_dir.is_absolute():
        test_dir = base / test_dir
    test_dir = test_dir.resolve()
    if not test_dir.is_dir() or test_dir.name != "images":
        raise FileNotFoundError(f"Cannot find YOLO test/images directory: {test_dir}")
    if not (test_dir.parent / "labels").is_dir():
        raise FileNotFoundError(f"Cannot find YOLO test/labels directory beside {test_dir}")
    return test_dir, class_names


def sha256_file(path: Path) -> str:
    """Identify the exact saved checkpoint used for the final assessment."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def f2_score(tp: int, fp: int, fn: int) -> float:
    """F2 rewards recall more than precision, using raw detection counts."""
    denominator = 5 * tp + 4 * fn + fp
    return 5 * tp / denominator if denominator else 0.0


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    """Save machine-readable reports for the project repository."""
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Run the frozen configuration once on the independent test split."""
    number, yaml_path, slug, description = select_dataset(DATASET_NUMBER)
    test_dir, class_names = locate_test_images(yaml_path)
    images = sorted(p for p in test_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise RuntimeError(f"No test images found in {test_dir}")
    image_stems = [p.stem for p in images]
    if len(set(image_stems)) != len(image_stems):
        raise ValueError("Two test images have the same stem; their labels would be ambiguous")

    base_name = training_run_base(MODEL_SIZE, slug)
    run_dir, checkpoint = find_training_checkpoint(base_name, requested_run=TRAINING_RUN)
    expected_names = [str(name) for name in class_names]
    model = YOLO(str(checkpoint))
    model_names = [str(model.names[i]) for i in range(len(model.names))]
    if model_names != expected_names:
        raise ValueError(f"Class mismatch: checkpoint {model_names}, dataset {expected_names}")

    output_dir = PROJECT_ROOT / "results" / "test" / RESULTS_SUBDIR
    if output_dir.exists():
        raise FileExistsError(
            f"Final test outputs already exist: {output_dir}\n"
            "Do not re-select settings using the test results or silently overwrite a prior test."
        )
    output_dir.mkdir(parents=True)  # create only after all paths/checkpoint are checked

    print("FINAL HELD-OUT TEST (no training, no further tuning)")
    print(f"Dataset: {number} ({description}), {len(images)} test images")
    print(f"Weights: {checkpoint}")
    print(f"Frozen settings: imgsz={IMAGE_SIZE}, conf={CONFIDENCE}, NMS IoU={NMS_IOU}, match IoU={MATCH_IOU}")

    counts_per_class = {idx: Counter() for idx in range(len(class_names))}
    image_rows: list[dict] = []
    seen = set()
    speeds = []
    negative_images = negative_images_flagged = 0
    flagged_images = images_with_defects = defect_images_flagged = 0

    start = time.perf_counter()
    predictions = model.predict(
        source=str(test_dir), imgsz=IMAGE_SIZE, conf=CONFIDENCE,
        iou=NMS_IOU, max_det=MAX_DETECTIONS, batch=BATCH,
        device=DEVICE, rect=False, stream=True, save=False, verbose=False,
    )
    for result in predictions:
        path = Path(result.path)
        stem = path.stem
        if stem not in image_stems or stem in seen:
            raise RuntimeError(f"Unexpected or duplicate prediction for {path}")
        seen.add(stem)
        height, width = result.orig_shape
        truth = read_ground_truth(test_dir.parent / "labels" / f"{stem}.txt", width, height, len(class_names))

        detections = []
        if result.boxes is not None and len(result.boxes):
            box_coords = result.boxes.xyxy.cpu().numpy()
            prediction_classes = result.boxes.cls.cpu().numpy().astype(int)
            scores = result.boxes.conf.cpu().numpy()
            for cls, score, box in zip(prediction_classes, scores, box_coords):
                if not (0 <= int(cls) < len(class_names)):
                    raise ValueError(f"Unexpected predicted class {cls} for {path}")
                detections.append((int(cls), float(score), tuple(map(float, box))))

        counts = match_detections(truth, detections, CONFIDENCE, MATCH_IOU, len(class_names))
        subtotal = Counter()
        for idx, class_counts in counts.items():
            counts_per_class[idx].update(class_counts)
            subtotal.update(class_counts)

        is_defective = bool(truth)
        flagged = bool(detections)
        images_with_defects += int(is_defective)
        flagged_images += int(flagged)
        defect_images_flagged += int(is_defective and flagged)
        if not is_defective:
            negative_images += 1
            negative_images_flagged += int(flagged)
        if result.speed and "inference" in result.speed:
            speeds.append(float(result.speed["inference"]))
        image_rows.append({
            "image": path.name, "ground_truth_defects": len(truth),
            "predicted_defects": len(detections),
            "tp": subtotal["tp"], "fp": subtotal["fp"], "fn": subtotal["fn"],
        })

    wall_seconds = time.perf_counter() - start
    if len(seen) != len(images):
        raise RuntimeError(f"Processed {len(seen)} images; expected {len(images)}")

    total_counts = Counter()
    class_rows = []
    for idx, class_counts in counts_per_class.items():
        total_counts.update(class_counts)
        class_rows.append({
            "class_id": idx, "class_name": class_names[idx], **measures(class_counts),
            "f2": f2_score(class_counts["tp"], class_counts["fp"], class_counts["fn"]),
        })
    summary = {
        "model": f"yolo11{MODEL_SIZE}", "training_run": run_dir.name,
        "dataset": number, "split": "test", "images": len(images),
        "imgsz": IMAGE_SIZE, "confidence": CONFIDENCE,
        **measures(total_counts),
        "f2": f2_score(total_counts["tp"], total_counts["fp"], total_counts["fn"]),
        "inference_ms_per_image": sum(speeds) / len(speeds) if speeds else None,
        "prediction_wall_seconds": wall_seconds,
        "negative_images": negative_images,
        "negative_images_flagged": negative_images_flagged,
        "negative_image_false_alarm_rate": (
            negative_images_flagged / negative_images if negative_images else None
        ),
        "images_with_labelled_defects": images_with_defects,
        "images_with_labelled_defects_flagged": defect_images_flagged,
        "flagged_images": flagged_images,
    }

    # These are standard AP metrics across confidence cut-offs, computed in a
    # separate low-confidence pass. They are NOT mAP at the chosen 0.20 threshold.
    print("Computing standard mAP50 and mAP50-95 on the test split (conf=0.001)...")
    ap_metrics = model.val(
        data=str(yaml_path), split="test", imgsz=IMAGE_SIZE,
        batch=BATCH, device=DEVICE, workers=0, conf=0.001,
        iou=NMS_IOU, rect=False, plots=True,
        project=str(RUNS_DIR), name=f"{run_dir.name}_test_map_{IMAGE_SIZE}",
        exist_ok=False, verbose=False,
    )
    summary["map50_low_conf"] = float(ap_metrics.box.map50)
    summary["map50_95_low_conf"] = float(ap_metrics.box.map)

    write_csv(output_dir / "test_summary.csv", [summary], list(summary))
    write_csv(output_dir / "test_per_class.csv", class_rows,
              ["class_id", "class_name", "tp", "fp", "fn", "precision", "recall", "f1", "f2"])
    write_csv(output_dir / "test_image_results.csv", image_rows,
              ["image", "ground_truth_defects", "predicted_defects", "tp", "fp", "fn"])

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_basis": "Settings selected from dataset 4 validation experiments BEFORE opening test results",
        "dataset_yaml": str(yaml_path), "dataset_description": description,
        "evaluation_split": "test", "class_names": class_names,
        "model": f"yolo11{MODEL_SIZE}", "training_run": run_dir.name,
        "checkpoint_path": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
        "imgsz": IMAGE_SIZE, "confidence_threshold": CONFIDENCE,
        "matching_iou": MATCH_IOU, "nms_iou": NMS_IOU, "batch": BATCH,
        "device": DEVICE, "rect": False, "max_det": MAX_DETECTIONS,
        "matching_rule": "Class-aware, highest-confidence-first, one-to-one box IoU>=0.5; micro P/R/F1/F2",
        "map_rule": "Separate Ultralytics model.val(split=test, conf=0.001); mAP is not confidence-0.20-specific",
        "model_val_save_dir": str(ap_metrics.save_dir),
        "operating_point": summary,
        "notes": [
            "Results are per bounding-box annotation, not package acceptance accuracy.",
            "Negative-image false-alarm rate counts images with no labelled defects that receive at least one detection.",
            "Do not use test outcomes to tune thresholds, model selection or image resolution.",
            "Possible near-duplicate images across dataset splits should be assessed when interpreting generalisation.",
        ],
    }
    (output_dir / "test_metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    print("\nFINAL TEST RESULTS (fixed confidence / IoU 0.50)")
    for key in ("tp", "fp", "fn", "precision", "recall", "f1", "f2",
                "map50_low_conf", "map50_95_low_conf", "inference_ms_per_image"):
        val = summary[key]
        print(f"  {key}: {val:.4f}" if isinstance(val, float) else f"  {key}: {val}")
    print("\nPer-class recall:")
    for row in class_rows:
        print(f"  {row['class_name']}: {row['recall']:.4f} ({row['tp']} TP, {row['fn']} FN)")
    print(f"\nReports: {output_dir}")
    print(f"Ultralytics mAP plots: {ap_metrics.save_dir}")
    print("This is the final held-out test; do not use its results for another optimisation round.")


if __name__ == "__main__":
    main()
