"""
camera_ws.py
------------
WebSocket endpoint that accepts raw JPEG frames from the browser camera,
runs YOLO detection + geometry check, and streams back slot statuses with
annotated bounding boxes.

Message protocol
----------------
Client → Server : binary blob (JPEG bytes)
Server → Client : JSON  {
    "success": true,
    "slots": [{ "id": str, "status": str, "points": [[x,y],...] }, ...],
    "detections": int,
    "boxes": [{ "bbox": [x1,y1,x2,y2], "confidence": float }, ...]
}
"""

import asyncio
import json
import logging

import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from detector import detect
from geometry import get_slot_status, POLYGON_FILE

logger = logging.getLogger("camera_ws")

router = APIRouter()


def _decode_frame(raw: bytes) -> np.ndarray | None:
    """Decode raw JPEG/PNG bytes sent from the browser into a NumPy BGR array."""
    arr = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return frame


def _build_response(frame: np.ndarray) -> dict:
    """Run inference on *frame* and return the JSON-serialisable response dict."""
    try:
        detections, results = detect(frame)
    except Exception as exc:
        logger.error("Detection error: %s", exc)
        detections = []

    # Slot occupancy
    status = get_slot_status(detections)

    # Load polygon definitions for the response
    try:
        with open(POLYGON_FILE, "r") as f:
            polygons = json.load(f)
    except Exception:
        polygons = {}

    slots = [
        {
            "id": slot_id,
            "status": status.get(slot_id, "Unknown"),
            "points": points,
        }
        for slot_id, points in polygons.items()
    ]

    # Serialise bounding boxes
    boxes = [
        {
            "bbox": list(d["bbox"]),
            "confidence": round(d["confidence"], 3),
        }
        for d in detections
    ]

    return {
        "success": True,
        "detections": len(detections),
        "slots": slots,
        "boxes": boxes,
    }


@router.websocket("/ws/camera")
async def camera_websocket(websocket: WebSocket):
    """
    Accepts binary JPEG frames from the frontend browser camera.
    Responds with detection + slot-status JSON after each frame.
    """
    await websocket.accept()
    logger.info("[CameraWS] Client connected: %s", websocket.client)

    try:
        while True:
            # Receive a binary JPEG frame (sent via websocket.send() in JS)
            try:
                raw: bytes = await asyncio.wait_for(
                    websocket.receive_bytes(), timeout=10.0
                )
            except asyncio.TimeoutError:
                # Send a keep-alive / empty response so the client knows we're alive
                await websocket.send_json({"success": False, "message": "timeout"})
                continue

            if not raw:
                continue

            # Decode + run inference in a thread so we don't block the event loop
            loop = asyncio.get_event_loop()
            frame = await loop.run_in_executor(None, _decode_frame, raw)

            if frame is None:
                await websocket.send_json(
                    {"success": False, "message": "Could not decode frame"}
                )
                continue

            response = await loop.run_in_executor(None, _build_response, frame)
            await websocket.send_json(response)

    except WebSocketDisconnect:
        logger.info("[CameraWS] Client disconnected")
    except Exception as exc:
        logger.error("[CameraWS] Unexpected error: %s", exc)
    finally:
        logger.info("[CameraWS] Connection closed")
