"""
api_server.py

FastAPI microservice for Parking Perception + ByteTrack + Occupancy Engine.
Exposes REST API endpoints so the backend team can integrate the model over HTTP.

Run server:
    uvicorn api_server:app --host 0.0.0.0 --port 8000
"""

import time
import cv2
import json
import numpy as np
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from detector import YOLODetector
from inference import ParkingPerceptionPipeline

app = FastAPI(title="Smart Parking Perception API", version="1.0.0")

SLOTS_PATH = Path(__file__).resolve().parents[1] / "polygons" / "slots.json"
with SLOTS_PATH.open(encoding="utf-8") as slots_file:
    DEFAULT_SLOTS_CONFIG = {
        "homography": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "slots": [
            {"id": slot_id, "corners": corners}
            for slot_id, corners in json.load(slots_file).items()
        ],
    }

# Hyperparameters tuned during evaluation
TRACKER_KWARGS = {
    "lost_track_buffer": 5,
    "track_activation_threshold": 0.25,
    "minimum_iou_threshold": 0.1,
}

# Initialize detector (using yolov8n.pt or ultimate_parking_model.pt)
detector = YOLODetector(weights_path=str(Path(__file__).with_name("yolov8n.pt")), conf=0.25)
pipeline = ParkingPerceptionPipeline(DEFAULT_SLOTS_CONFIG, detector, TRACKER_KWARGS)


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "Parking Perception API"}


@app.get("/status")
def get_slot_status():
    """Returns the current confirmed occupancy status for all slots."""
    return {"slots": pipeline.current_slot_states()}


@app.post("/process-frame")
async def process_frame(file: UploadFile = File(...)):
    """
    Upload a camera frame JPEG/PNG image over HTTP POST.
    Runs YOLO detection + ByteTrack + Occupancy state machine.
    
    Returns:
        events: List of new enter/exit events generated in this frame
        slot_states: Current confirmed occupancy state of all slots
    """
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image file uploaded.")

    now = time.time()
    events = pipeline.process_frame(frame, now=now)
    slot_states = pipeline.current_slot_states()

    return JSONResponse(content={
        "timestamp": now,
        "events": events,
        "slot_states": slot_states,
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
