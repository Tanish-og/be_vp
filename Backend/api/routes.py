import json
from pathlib import Path
from fastapi import Body

from fastapi import APIRouter
import cv2
from database.mongo import is_mongo_connected, save_event, get_all_events

from detector import detect
from geometry import get_slot_status
from detector import MODEL_PATH

router = APIRouter()
BASE_DIR = Path(__file__).resolve().parent.parent
POLYGON_FILE = BASE_DIR / "polygons" / "slots.json"
previous_status = {}
latest_slots = {}


def load_polygons():
    with POLYGON_FILE.open(encoding="utf-8") as polygon_file:
        return json.load(polygon_file)


def build_live_slots():
    polygons = load_polygons()
    return [
        {
            "id": slot_id,
            "status": latest_slots.get(slot_id, "Unknown"),
            "points": points,
        }
        for slot_id, points in polygons.items()
    ]

def build_slots_payload():
    global previous_status

    cap = cv2.VideoCapture("parking.mp4")
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return {
            "success": False,
            "message": "Could not read frame from video."
        }

    detections, _ = detect(frame)
    status = get_slot_status(detections)

    with open(POLYGON_FILE, "r") as f:
        polygons = json.load(f)

    for slot, state in status.items():
        if previous_status.get(slot) != state:
            try:
                save_event(slot, state, len(detections))
            except Exception as e:
                print("MongoDB Error:", e)

    previous_status = status.copy()

    slots = []
    for slot_id, points in polygons.items():
        slots.append({
            "id": slot_id,
            "status": status.get(slot_id, "Unknown"),
            "points": points
        })

    return {
        "success": True,
        "detections": len(detections),
        "slots": slots
    }


@router.get("/")
def home():
    return {
        "message": "VisionPark Backend Running"
    }


@router.get("/health")
def health():
    return {
        "status": "healthy"
    }


@router.get("/health/model")
def model_health():
    try:
        loaded = MODEL_PATH.exists()
    except Exception:
        loaded = False

    return {
        "status": "healthy" if loaded else "unhealthy",
        "loaded": loaded,
        "model_path": str(MODEL_PATH)
    }


@router.get("/health/polygon")
def polygon_health():
    try:
        with open(POLYGON_FILE, "r") as f:
            polygons = json.load(f)
        loaded = bool(polygons)
        count = len(polygons)
    except Exception:
        loaded = False
        count = 0

    return {
        "status": "healthy" if loaded else "unhealthy",
        "loaded": loaded,
        "count": count
    }


@router.get("/health/mongo")
def mongo_health():
    connected = is_mongo_connected()

    return {
        "status": "healthy" if connected else "unhealthy",
        "connected": connected
    }

@router.get("/slots")
def get_slots():
    if latest_slots:
        return {
            "success": True,
            "detections": None,
            "slots": build_live_slots(),
        }
    return build_slots_payload()


@router.post("/parking-events")
def receive_parking_event(event: dict = Body(...)):
    """Accept an enter/exit event emitted by the bytesort service."""
    slot_id = event.get("slot_id")
    event_type = event.get("event_type", event.get("event"))

    if not slot_id or event_type not in {"enter", "exit"}:
        return {
            "success": False,
            "message": "slot_id and event_type (enter or exit) are required",
        }

    status = "Occupied" if event_type == "enter" else "Available"
    latest_slots[slot_id] = status

    try:
        save_event(slot_id, status, event.get("track_id"))
    except Exception as error:
        print("MongoDB Error:", error)

    return {
        "success": True,
        "slot_id": slot_id,
        "status": status,
    }


@router.get("/live-status")
def live_status():
    return {
        "success": True,
        "slots": latest_slots,
    }

@router.get("/history")
def history():

    events = get_all_events()

    return {
        "success": True,
        "count": len(events),
        "events": events
    }

@router.post("/config")
def upload_config(config: dict = Body(...)):

    with open(POLYGON_FILE, "w") as f:
        json.dump(config, f, indent=4)

    return {
        "success": True,
        "message": "Polygon configuration updated successfully."
    }


# @router.get("/slots")
# def get_slots():

#     # image = cv2.imread("test.jpg")

#     # detections, _ = detect(image)

#     # status = get_slot_status(detections)

#     return status