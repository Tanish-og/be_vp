"""
inference.py

The actual production pipeline: detector -> ByteTrack -> ground-plane
occupancy engine -> the track-event contract (track_id, slot_id,
event_type, timestamp) the backend's duration/violation logic consumes,
per Week 3 of the ML roadmap.

One thing worth calling out: now that real ByteTrack is in the loop,
it supersedes the DIY CentroidTracker built earlier as a stand-in - real
tracking gives persistent, Kalman-smoothed positions per vehicle "for
free," so the same is_stationary logic (needed to reject drive-by
flicker) is now derived from ByteTrack's own track history instead of a
hand-rolled matcher.
"""

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import supervision as sv

from tracker import build_tracker
from occupancy_engine import (
    SlotTracker,
    SlotState,
    project_footprint_to_ground,
    overlap_ratio,
)

STATIONARY_SPEED_THRESHOLD = 3.0   # ground-plane units/sec
STATIONARY_CONFIRM_SEC = 1.5
OCCLUSION_GRAZE_MIN = 0.05


@dataclass
class TrackHistory:
    positions: deque = field(default_factory=lambda: deque(maxlen=30))  # (x, y, t)

    def push(self, pos, t):
        self.positions.append((pos[0], pos[1], t))

    def is_stationary(self, now: float) -> bool:
        recent = [p for p in self.positions if now - p[2] <= STATIONARY_CONFIRM_SEC]
        if len(recent) < 2:
            return False
        (x0, y0, t0), (x1, y1, t1) = recent[0], recent[-1]
        dt = max(t1 - t0, 1e-6)
        speed = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5 / dt
        return speed <= STATIONARY_SPEED_THRESHOLD


class ParkingPerceptionPipeline:
    """Wraps detector + tracker + per-slot occupancy engine, and emits
    (track_id, slot_id, event_type, timestamp) events on every enter/exit -
    the contract backend's Week 6 duration/violation logic depends on."""

    def __init__(self, slots_config: dict, detector, tracker_kwargs: dict = None):
        self.detector = detector
        self.tracker = build_tracker(**(tracker_kwargs or {}))

        self.homography = np.array(slots_config["homography"], dtype=np.float32)
        self.slot_polygons = {}
        for slot in slots_config["slots"]:
            pixel_pts = np.array(slot["corners"], dtype=np.float32).reshape(-1, 1, 2)
            ground_pts = cv2.perspectiveTransform(pixel_pts, self.homography).reshape(-1, 2)
            from shapely.geometry import Polygon
            self.slot_polygons[slot["id"]] = Polygon(ground_pts)

        self.slot_states = {sid: SlotTracker(slot_id=sid) for sid in self.slot_polygons}
        self.occupying_track: Dict[str, Optional[int]] = {sid: None for sid in self.slot_polygons}
        self.track_histories: Dict[int, TrackHistory] = defaultdict(TrackHistory)

    def process_frame(self, frame: np.ndarray, now: float = None) -> List[dict]:
        """Runs one frame through the full pipeline. Returns any new
        (track_id, slot_id, event_type, timestamp) events generated."""
        now = now if now is not None else time.time()

        raw_detections = self.detector.detect(frame)
        sv_detections = sv.Detections(
            xyxy=np.array([d.bbox_xyxy for d in raw_detections], dtype=np.float32) if raw_detections
                 else np.zeros((0, 4), dtype=np.float32),
            confidence=np.array([d.confidence for d in raw_detections], dtype=np.float32) if raw_detections
                       else np.zeros((0,), dtype=np.float32),
            class_id=np.zeros(len(raw_detections), dtype=int),
        )
        tracked = self.tracker.update(sv_detections, timestamp=now)

        # Project every confirmed track's footprint to ground-plane and
        # update its motion history.
        footprints, stationary_flags, track_ids = [], [], []
        for tid, bbox in zip(tracked.tracker_id, tracked.xyxy.tolist()):
            if tid == -1:
                continue
            footprint = project_footprint_to_ground(tuple(bbox), self.homography)
            centroid = (footprint.centroid.x, footprint.centroid.y)
            self.track_histories[tid].push(centroid, now)

            footprints.append(footprint)
            stationary_flags.append(self.track_histories[tid].is_stationary(now))
            track_ids.append(tid)

        events = []
        for slot_id, slot_polygon in self.slot_polygons.items():
            overlaps = [overlap_ratio(fp, slot_polygon) for fp in footprints]
            if overlaps:
                best_idx = max(range(len(overlaps)), key=lambda i: overlaps[i])
                max_overlap = overlaps[best_idx]
                best_track_id = track_ids[best_idx]
                best_stationary = stationary_flags[best_idx]
            else:
                max_overlap, best_track_id, best_stationary = 0.0, None, True

            occlusion_nearby = any(OCCLUSION_GRAZE_MIN < o < SlotTracker.OVERLAP_THRESHOLD for o in overlaps)

            prev_state = self.slot_states[slot_id].state
            new_state = self.slot_states[slot_id].update(
                overlap=max_overlap, now=now, occlusion_nearby=occlusion_nearby,
                verifier_prob=None, vehicle_stationary=best_stationary,
            )

            if prev_state != SlotState.OCCUPIED and new_state == SlotState.OCCUPIED:
                self.occupying_track[slot_id] = best_track_id
                events.append({"track_id": best_track_id, "slot_id": slot_id, "event_type": "enter", "timestamp": now})

            elif prev_state == SlotState.OCCUPIED and new_state != SlotState.OCCUPIED:
                exiting_track = self.occupying_track[slot_id]
                self.occupying_track[slot_id] = None
                events.append({"track_id": exiting_track, "slot_id": slot_id, "event_type": "exit", "timestamp": now})

        return events

    def current_slot_states(self) -> Dict[str, str]:
        return {sid: t.state.value for sid, t in self.slot_states.items()}


if __name__ == "__main__":
    # Minimal runnable example using the mock detector - swap in a real
    # YOLODetector (see detector.py in the smart_parking_app backend)
    # once you have trained weights.
    import json
    from detector import MockDetector, Detection

    def demo_script(frame_index):
        if frame_index < 5:
            return []
        if 5 <= frame_index < 30:
            return [Detection(bbox_xyxy=(60, 80, 140, 150), confidence=0.9)]
        return []

    slots_config = {
        "homography": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "slots": [{"id": "A1", "corners": [[50, 50], [150, 50], [150, 150], [50, 150]]}],
    }

    pipeline = ParkingPerceptionPipeline(slots_config, MockDetector(demo_script))

    sim_clock = 0.0
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for i in range(35):
        events = pipeline.process_frame(blank_frame, now=sim_clock)
        for e in events:
            print(f"t={sim_clock:.0f}s  {e}")
        sim_clock += 1.0

    print("\nFinal slot states:", pipeline.current_slot_states())
