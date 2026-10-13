
"""
YOLO11 Packaging Defect Detection - Training Module.

This module trains a YOLO11 object detection model
using the Roboflow Pork Rasher Error Packaging dataset.

The model detects five packaging defect categories:
loose-meat, packaging-error, twisted-meat,
unsealed, and wrinkle.
"""

from ultralytics import YOLO


def main():
    """
    Train the YOLO11n model using the packaging dataset.

    Loads pretrained YOLO11n weights and fine-tunes them
    on labelled pork packaging images.

    Training configuration:
        - Epochs: 10
        - Image resolution: 640 x 640
        - Batch size: 4
        - Device: CUDA GPU 0

    Returns:
        None

    Raises:
        FileNotFoundError:
            If the dataset or model weights cannot be found.
        RuntimeError:
            If training fails or CUDA is unavailable.
    """
    model = YOLO("weights/yolo11n.pt")

    model.train(
        data="data/data.yaml",
        epochs=10,
        imgsz=640,
        batch=4,
        device=0,
        project="runs",
        name="packaging_detector"
    )


if __name__ == "__main__":
    main()
