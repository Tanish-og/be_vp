"""
tracker.py

Thin, reusable wrapper: feed it detections frame by frame, get back the
same detections with a persistent tracker_id attached. Used by both
evaluate_tracking.py (feeding it detections from a file) and
inference.py (feeding it live YOLO output).
"""

from typing import List, Tuple
import numpy as np
import supervision as sv
from trackers import ByteTrackTracker


def build_tracker(
    track_activation_threshold: float = 0.25,
    lost_track_buffer: int = 30,
    minimum_iou_threshold: float = 0.3,
    frame_rate: float = 10.0,
) -> ByteTrackTracker:
    """
    track_activation_threshold: min detection confidence to start a new
        track. Lower = catches more real vehicles but risks starting
        tracks on noise.
    lost_track_buffer: frames a track survives with no matching
        detection before being dropped for good - this is the knob that
        controls occlusion tolerance. Higher = survives longer
        occlusions but risks merging two different cars into one ID if
        a new car enters the same spot while the old track is still
        "remembered."
    minimum_iou_threshold: how much overlap is required to match a
        detection to an existing track between frames.
    """
    return ByteTrackTracker(
        track_activation_threshold=track_activation_threshold,
        lost_track_buffer=lost_track_buffer,
        minimum_iou_threshold=minimum_iou_threshold,
        frame_rate=frame_rate,
    )


def detections_from_boxes(boxes: List[Tuple[float, float, float, float]], confidences: List[float] = None) -> sv.Detections:
    """boxes: list of (x1, y1, x2, y2). Builds an sv.Detections object
    the tracker can consume, the same shape a real YOLO model's output
    would take after sv.Detections.from_ultralytics()."""
    if not boxes:
        return sv.Detections.empty()
    xyxy = np.array(boxes, dtype=np.float32)
    conf = np.array(confidences if confidences else [0.9] * len(boxes), dtype=np.float32)
    class_id = np.zeros(len(boxes), dtype=int)
    return sv.Detections(xyxy=xyxy, confidence=conf, class_id=class_id)
