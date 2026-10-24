"""Train YOLO11n or YOLO11s on a numbered packaging-defect dataset.

Examples:
    python src/train.py                       # interactive prompts
    python src/train.py --dataset 4 --model n
    python src/train.py --dataset 4 --model s --batch 2
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

from ultralytics import YOLO

if __package__:
    from .experiment_config import (
        PROJECT_ROOT, RUNS_DIR, select_dataset, select_model,
        training_run_base, unused_run_name,
    )
else:
    from experiment_config import (
        PROJECT_ROOT, RUNS_DIR, select_dataset, select_model,
        training_run_base, unused_run_name,
    )


def main():
    """Fine-tune a pretrained detector, keeping every run in its own folder."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=int, choices=(1, 2, 3, 4))
    parser.add_argument("--model", choices=("n", "s"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch", type=int, default=4)
    args = parser.parse_args()

    number, dataset_yaml, slug, description = select_dataset(args.dataset)
    size = select_model(args.model)
    base = training_run_base(size, slug)
    run_name = unused_run_name(base)

    pretrained_name = f"yolo11{size}.pt"
    in_weights = PROJECT_ROOT / "weights" / pretrained_name
    in_root = PROJECT_ROOT / pretrained_name
    # Prefer the project's locally saved weights before an automatic download.
    pretrained = (
        str(in_weights) if in_weights.is_file() else
        str(in_root) if in_root.is_file() else pretrained_name
    )

    settings = {
        "data": str(dataset_yaml),
        "epochs": args.epochs,
        "patience": args.patience,
        "imgsz": 640,
        "batch": args.batch,
        "workers": 0,
        "cache": False,
        "device": 0,
        "seed": 42,
        "save": True,
        "project": str(RUNS_DIR),
        "name": run_name,
        "exist_ok": False,
    }

    print(f"\nDataset {number}: {description}")
    print(f"YAML: {dataset_yaml}")
    print(f"Pretrained weights: {pretrained}")
    print(f"Saving to: {RUNS_DIR / run_name}")
    model = YOLO(str(PROJECT_ROOT / "weights" / f"yolo11{args.model}.pt"))
    model.train(**settings)

    # Save a compact experiment record alongside Ultralytics' args.yaml.
    run_dir = RUNS_DIR / run_name
    if run_dir.is_dir():
        metadata = {
            "dataset_number": number,
            "dataset_description": description,
            "dataset_yaml": str(dataset_yaml),
            "model": f"yolo11{size}",
            "initial_weights": pretrained,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "training_settings": settings,
        }
        (run_dir / "experiment.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Training results: {run_dir}")


if __name__ == "__main__":
    main()
