
"""Train YOLO11s for two-class packaging defect detection."""

from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    """Fine-tune YOLO11s on the revised two-class dataset."""

    # Load pretrained YOLO11s weights
    model = YOLO("yolo11s.pt")

    model.train(
        data=str(
            PROJECT_ROOT / "data"
            / "4 - two_class_v2"
            / "data.yaml"
        ),
        epochs=100,
        patience=10,
        imgsz=640,
        batch=4,
        workers=0,
        cache=False,
        device=0,
        seed=42,
        save=True,
        project=str(PROJECT_ROOT / "runs"),
        name="yolo11s_2class_v2"
    )


if __name__ == "__main__":
    main()
