from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best.pt"

model = YOLO(str(MODEL_PATH))


def detect(frame):
    results = model(frame)

    boxes = []

    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        confidence = float(box.conf[0])

        boxes.append({
            "bbox": (
                int(x1),
                int(y1),
                int(x2),
                int(y2)
            ),
            "confidence": confidence
        })

    return boxes, results