from ultralytics import YOLO

def main():
    model = YOLO("yolo11n.pt")

    model.train(
        data="data/data.yaml",
        epochs=10,
        imgsz=640,
        project="runs",
        name="packaging_detector"
    )

if __name__ == "__main__":
    main()

