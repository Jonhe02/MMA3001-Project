"""Validate a trained YOLO11 model on a numbered packaging dataset.

Examples:
    python src/evaluate.py                           # interactive prompts
    python src/evaluate.py --dataset 4 --model n
    python src/evaluate.py --dataset 4 --model s
    python src/evaluate.py --dataset 4 --model s --split test
"""

"""DATASETS = {
    1: ("1 - five_class", "5class_aug", "Original augmented, five classes"),
    2: ("2 - two_class", "2class_v1", "First two-class dataset"),
    3: ("3 - raw_unaug", "5class_unaug", "Rebalanced, no new offline augmentation, five classes"),
    4: ("4 - two_class_v2", "2class_v2", "Rebalanced two-class dataset"),
}"""

import argparse
import json
from datetime import datetime, timezone

import yaml
from ultralytics import YOLO

if __package__:
    from .experiment_config import (
        RUNS_DIR, select_dataset, select_model, training_run_base,
        unused_run_name, find_training_checkpoint,
    )
else:
    from experiment_config import (
        RUNS_DIR, select_dataset, select_model, training_run_base,
        unused_run_name, find_training_checkpoint,
    )


def _class_names(data):
    """Read class labels from a YOLO dataset YAML as an ordered list."""
    names = data.get("names")
    if isinstance(names, list):
        return [str(name) for name in names]
    if isinstance(names, dict):
        return [str(value) for _, value in sorted(names.items(), key=lambda pair: int(pair[0]))]
    raise ValueError("Dataset YAML must contain 'names' as a list or mapping.")


def main():
    """Evaluate the appropriate saved checkpoint on validation or test images."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=int, choices=(1, 2, 3, 4))
    parser.add_argument("--model", choices=("n", "s"))
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--run", help="Optional exact training run folder, e.g. yolo11n_2class_v2_run2")
    args = parser.parse_args()

    number, dataset_yaml, slug, description = select_dataset(args.dataset)
    size = select_model(args.model)
    base = training_run_base(size, slug)
    run_dir, weights = find_training_checkpoint(base, args.run)

    model = YOLO(str(weights))
    with dataset_yaml.open("r", encoding="utf-8") as stream:
        dataset_info = yaml.safe_load(stream)
    expected_names = _class_names(dataset_info)
    actual_names = [str(model.names[index]) for index in range(len(model.names))]
    if actual_names != expected_names:
        raise ValueError(
            "Checkpoint/dataset class mismatch!\n"
            f"Dataset classes: {expected_names}\n"
            f"Model classes:   {actual_names}\n"
            "Check --dataset, --model and --run."
        )

    output_name = unused_run_name(f"{run_dir.name}_{args.split}")
    print(f"\nDataset {number}: {description}")
    print(f"Trained model: {weights}")
    print(f"Evaluation split: {args.split}")
    print(f"Saving to: {RUNS_DIR / output_name}")

    metrics = model.val(
        data=str(dataset_yaml),
        split=args.split,
        imgsz=640,
        batch=4,
        device=0,
        workers=0,
        plots=True,
        project=str(RUNS_DIR),
        name=output_name,
        exist_ok=False,
    )

    print("\n--- Overall detection metrics ---")
    print(f"Precision:  {metrics.box.mp:.4f}")
    print(f"Recall:     {metrics.box.mr:.4f}")
    print(f"mAP50:      {metrics.box.map50:.4f}")
    print(f"mAP50-95:   {metrics.box.map:.4f}")
    print("\n--- Per-class mAP50-95 ---")
    for class_id, class_name in enumerate(expected_names):
        print(f"{class_name}: {metrics.box.maps[class_id]:.4f}")
    print("\n--- Speed (ms/image) ---")
    for stage, milliseconds in metrics.speed.items():
        print(f"{stage}: {milliseconds:.2f}")

    output_dir = RUNS_DIR / output_name
    if output_dir.is_dir():
        metadata = {
            "dataset_number": number,
            "dataset_yaml": str(dataset_yaml),
            "model": f"yolo11{size}",
            "training_run": run_dir.name,
            "trained_checkpoint": str(weights),
            "evaluation_split": args.split,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "precision": metrics.box.mp,
            "recall": metrics.box.mr,
            "map50": metrics.box.map50,
            "map50_95": metrics.box.map,
            "speed_ms_per_image": metrics.speed,
        }
        (output_dir / "evaluation_summary.json").write_text(
            json.dumps(metadata, indent=2, default=float) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
