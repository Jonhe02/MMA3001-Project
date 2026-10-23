
"""Train a YOLO11n model to detect pork packaging defects."""

from pathlib import Path
from ultralytics import YOLO

# Get the project's root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    """Train YOLO11n using the revised two-class dataset."""

    # Load pretrained model weights
    model = YOLO(str(PROJECT_ROOT / "weights" / "yolo11n.pt"))

    # Train using the Roboflow annotations
    model.train(
        epochs=100,
        patience=10,
        imgsz=640,
        batch=4,
        workers=0,
        cache=False,
        device=0,
        seed=42,
        data=str(PROJECT_ROOT / "data" / "4 - two_class_v2" / "data.yaml"),
        name="yolo11n_2class_v2",
    )


if __name__ == "__main__":
    main()
