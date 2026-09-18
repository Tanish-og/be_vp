import json
import cv2
import numpy as np
from pathlib import Path

# Get current directory
BASE_DIR = Path(__file__).resolve().parent

# Path to slots.json
POLYGON_FILE = BASE_DIR / "polygons" / "slots.json"

# Load polygon coordinates
with open(POLYGON_FILE, "r") as f:
    SLOT_POLYGONS = json.load(f)

# Convert polygons to NumPy arrays
slot_polygons = {}

for slot_name, points in SLOT_POLYGONS.items():
    slot_polygons[slot_name] = np.array(points, dtype=np.int32)


def point_inside_polygon(point, polygon):
    result = cv2.pointPolygonTest(
        polygon,
        point,
        False
    )
    return result >= 0


def get_slot_status(detections):
    """
    detections = [
        {
            "bbox": (x1, y1, x2, y2),
            "confidence": 0.95
        },
        ...
    ]
    """

    # Initially every slot is empty
    status = {}

    for slot in slot_polygons:
        status[slot] = "Empty"

    # Check every detected vehicle
    for detection in detections:

        x1, y1, x2, y2 = detection["bbox"]

        center = (
            int((x1 + x2) / 2),
            int((y1 + y2) / 2)
        )

        for slot, polygon in slot_polygons.items():

            if point_inside_polygon(center, polygon):
                status[slot] = "Occupied"

    return status


if __name__ == "__main__":

    point = (170, 220)
    print(point_inside_polygon(point, slot_polygons["A1"]))