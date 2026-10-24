"""Shared dataset selection and experiment naming for YOLO training/evaluation."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT_ROOT / "runs"

# Dataset number: (folder name, run-name suffix, description).
DATASETS = {
    1: ("1 - five_class", "5class_aug", "Original augmented, five classes"),
    2: ("2 - two_class", "2class_v1", "First two-class dataset"),
    3: ("3 - raw_unaug", "5class_unaug", "Rebalanced, no new offline augmentation, five classes"),
    4: ("4 - two_class_v2", "2class_v2", "Rebalanced two-class dataset"),
}


def select_dataset(number=None):
    """Resolve a dataset number, prompting when none was supplied."""
    if number is None:
        print("\nAvailable datasets:")
        for key, (folder, _, description) in DATASETS.items():
            print(f"  {key}. {description}  [{folder}]")
        number = input("Dataset number (1-4): ").strip()

    try:
        number = int(number)
    except (TypeError, ValueError) as exc:
        raise ValueError("Dataset must be a number from 1 to 4.") from exc

    if number not in DATASETS:
        raise ValueError(f"Unknown dataset {number}; choose 1, 2, 3, or 4.")

    folder, slug, description = DATASETS[number]
    yaml_path = PROJECT_ROOT / "data" / folder / "data.yaml"
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {yaml_path}")
    return number, yaml_path, slug, description


def select_model(size=None):
    """Resolve YOLO11 model size, prompting when none was supplied."""
    if size is None:
        size = input("YOLO11 model size (n = nano, s = small): ").strip()
    size = str(size).lower().strip()
    if size not in ("n", "s"):
        raise ValueError("Model must be 'n' or 's'.")
    return size


def training_run_base(size, slug):
    """Produce an easily identifiable, deterministic training run name."""
    return f"yolo11{size}_{slug}"


def unused_run_name(base):
    """Allocate a distinct name rather than overwriting previous experiment results."""
    if not (RUNS_DIR / base).exists():
        return base
    number = 2
    while (RUNS_DIR / f"{base}_run{number}").exists():
        number += 1
    return f"{base}_run{number}"


def find_training_checkpoint(base, requested_run=None):
    """Find best.pt from a matching trained run (latest numbered run by default)."""
    if requested_run is not None:
        requested_run = str(requested_run)
        if Path(requested_run).name != requested_run:
            raise ValueError("--run must be a run folder name, not a path.")
        if requested_run != base and not (
            requested_run.startswith(f"{base}_run")
            and requested_run[len(base) + 4:].isdigit()
        ):
            raise ValueError(f"Run '{requested_run}' does not match model/dataset '{base}'.")
        run_dir = RUNS_DIR / requested_run
        weights = run_dir / "weights" / "best.pt"
        if not weights.is_file():
            raise FileNotFoundError(f"Trained checkpoint not found: {weights}")
        return run_dir, weights

    candidates = []
    for run_dir in RUNS_DIR.iterdir() if RUNS_DIR.exists() else []:
        if not run_dir.is_dir():
            continue
        if run_dir.name == base:
            run_number = 1
        elif run_dir.name.startswith(f"{base}_run"):
            suffix = run_dir.name[len(base) + 4:]
            if not suffix.isdigit():
                continue
            run_number = int(suffix)
        else:
            continue
        weights = run_dir / "weights" / "best.pt"
        if weights.is_file():
            candidates.append((run_number, run_dir, weights))

    if not candidates:
        raise FileNotFoundError(
            f"No trained best.pt found for '{base}' under {RUNS_DIR}. "
            "Train this model/dataset combination first."
        )
    _, run_dir, weights = max(candidates, key=lambda item: item[0])
    return run_dir, weights
