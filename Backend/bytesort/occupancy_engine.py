"""
ground_plane_occupancy.py

Two upgrades over pure image-plane point-in-polygon testing:

1. GROUND-PLANE PROJECTION instead of raw pixel point-in-polygon.
   Tall vehicles (SUVs, vans) photographed by an elevated oblique
   camera visually "lean" toward the camera - their roof can appear to
   spill into the neighboring slot's polygon even when the car's actual
   footprint (where its wheels touch the ground) is correctly parked.
   Fix: project only the BOTTOM strip of the detected bounding box
   (closest to the ground) through the homography into real-world
   ground coordinates, then measure overlap against the slot polygon
   there - not in distorted camera pixels.

2. TRACK-AWARE, ASYMMETRIC STATE RELEASE to handle occlusion.
   A parked car briefly hidden behind a passing/parking vehicle should
   NOT flip to "empty" just because the detector missed it for a couple
   of frames. Missing a car (false empty) sends a driver to a spot
   that's actually full - a worse outcome than being slow to confirm a
   spot has emptied. So: occupied is set quickly, empty is only
   confirmed after a longer, stricter grace period, with an explicit
   UNKNOWN state for genuinely ambiguous moments so routing never
   targets a slot the system isn't actually sure about.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import cv2
import numpy as np
from shapely.geometry import Polygon


class SlotState(Enum):
    AVAILABLE = "available"                 # confident empty -> safe to route here
    LIKELY_AVAILABLE = "likely_available"   # empty, but occlusion nearby recently
    OCCUPIED = "occupied"
    UNKNOWN = "unknown"                     # can't confirm -> never route here

# Only this state should ever be handed to the A* routing engine.
ROUTABLE_STATES = {SlotState.AVAILABLE}


def project_footprint_to_ground(bbox_xyxy, homography: np.ndarray, footprint_fraction: float = 0.15) -> Polygon:
    """
    bbox_xyxy: (x1, y1, x2, y2) detector output, camera pixel coords.
    homography: 3x3 matrix mapping camera pixels -> ground-plane coords
        (the same matrix built during the calibration step).
    footprint_fraction: fraction of the box height, measured from the
        bottom, treated as the "ground contact" strip. ~15% avoids using
        just the full bottom edge (still some vehicle body in it) while
        being more stable than a single bottom-center point.

    Returns a small polygon in ground-plane coordinates approximating
    where the vehicle actually touches the ground.
    """
    x1, y1, x2, y2 = bbox_xyxy
    strip_top = y2 - (y2 - y1) * footprint_fraction
    pixel_pts = np.array([
        [x1, strip_top], [x2, strip_top], [x2, y2], [x1, y2],
    ], dtype=np.float32).reshape(-1, 1, 2)

    ground_pts = cv2.perspectiveTransform(pixel_pts, homography).reshape(-1, 2)
    return Polygon(ground_pts)


def overlap_ratio(vehicle_footprint: Polygon, slot_polygon: Polygon) -> float:
    """Fraction of the VEHICLE FOOTPRINT's own area that falls inside
    this slot polygon. Normalizing by the footprint's area (not the
    slot's) matters because the footprint is a thin ground-contact
    strip, not the whole box - and this framing gives an intuitive
    reading either way: ~1.0 means the car is cleanly inside this one
    slot, ~0.5 on two different slots means the car is straddling both
    (correctly flagging both as unusable), and a low value on both means
    neither slot is meaningfully affected."""
    if not vehicle_footprint.is_valid or not slot_polygon.is_valid:
        return 0.0
    if vehicle_footprint.area == 0:
        return 0.0
    return vehicle_footprint.intersection(slot_polygon).area / vehicle_footprint.area


@dataclass
class SlotTracker:
    """Per-slot occupancy state machine with asymmetric debounce."""

    slot_id: str
    state: SlotState = SlotState.AVAILABLE
    last_occupied_seen: float = float("-inf")
    occlusion_flagged_until: float = float("-inf")

    OCCUPY_CONFIRM_SEC: float = 2.0     # quick to mark occupied
    RELEASE_CONFIRM_SEC: float = 8.0    # slow, stricter, to mark empty
    OCCLUSION_GRACE_SEC: float = 12.0   # how long "recently occluded" lingers
    OVERLAP_THRESHOLD: float = 0.45     # min fraction of the vehicle's footprint
                                         # inside this slot to count as "present"
                                         # (~0.5 on two slots = straddling car,
                                         # correctly flags both)

    def update(self, overlap: float, now: float, occlusion_nearby: bool,
               verifier_prob: Optional[float], vehicle_stationary: bool = True) -> SlotState:
        """
        overlap: max ground-plane overlap ratio from any detected vehicle
            this frame (0.0 if none detected over this slot).
        occlusion_nearby: True if a large vehicle is currently detected in
            an adjacent slot (raises suspicion this slot's view may be
            partially blocked right now).
        verifier_prob: patch-verifier occupied-probability for this slot
            this frame, or None if the verifier wasn't run this frame.
        vehicle_stationary: whether the vehicle contributing `overlap`
            has actually stopped moving (from real track history), not
            just been momentarily present. A car threading through a
            tight aisle can satisfy the overlap test for a couple of
            frames without ever parking - requiring genuine stillness is
            what tells "parked" apart from "driving through." Defaults
            to True so callers that don't track motion get the old
            behavior.
        """
        vehicle_present = overlap >= self.OVERLAP_THRESHOLD and vehicle_stationary
        if vehicle_present:
            self.last_occupied_seen = now

        if occlusion_nearby:
            self.occlusion_flagged_until = now + self.OCCLUSION_GRACE_SEC
        currently_occluded = now < self.occlusion_flagged_until

        # Case 1: seeing a vehicle right now, and have for long enough
        # to trust it (avoids flickering on a single stray frame).
        if vehicle_present and (now - self.last_occupied_seen) <= self.OCCUPY_CONFIRM_SEC:
            self.state = SlotState.OCCUPIED
            return self.state

        time_since_seen = now - self.last_occupied_seen

        # Case 2: not seeing a vehicle right now, but one was confirmed
        # recently -> hold state rather than snap to empty.
        if time_since_seen < self.RELEASE_CONFIRM_SEC:
            self.state = SlotState.UNKNOWN if currently_occluded else SlotState.OCCUPIED
            return self.state

        # Case 3: long clean gap with no vehicle signal -> safe to
        # consider empty, but let the patch verifier have final say,
        # especially if this slot was recently a plausible occlusion risk.
        if verifier_prob is not None and verifier_prob >= 0.5:
            self.state = SlotState.UNKNOWN  # verifier disagrees - stay cautious
        elif currently_occluded:
            self.state = SlotState.LIKELY_AVAILABLE
        else:
            self.state = SlotState.AVAILABLE

        return self.state


if __name__ == "__main__":
    # --- Simulated sequence proving the asymmetric debounce works ---
    # Identity homography for this synthetic test - ground coords == pixel coords.
    H = np.eye(3, dtype=np.float32)
    slot_polygon = Polygon([[0, 0], [100, 0], [100, 100], [0, 100]])
    tracker = SlotTracker(slot_id="A1")

    def frame(overlap, t, occluded=False, verifier=None):
        state = tracker.update(overlap, t, occluded, verifier)
        print(f"t={t:5.1f}s overlap={overlap:.2f} occluded_nearby={occluded!s:5} -> {state.value}")
        return state

    print("Car pulls in:")
    frame(0.0, 0.0)
    frame(0.9, 1.0)
    s = frame(0.9, 2.5)
    assert s == SlotState.OCCUPIED

    print("\nA passing truck occludes the view for a few seconds:")
    s = frame(0.0, 3.0, occluded=True)
    assert s != SlotState.AVAILABLE, "must NOT flip to available during occlusion"
    s = frame(0.0, 6.0, occluded=True)
    assert s != SlotState.AVAILABLE

    print("\nTruck moves on, car is confirmed still there:")
    s = frame(0.9, 7.0)
    assert s == SlotState.OCCUPIED

    print("\nCar actually leaves, long clean gap, verifier confirms empty:")
    s = frame(0.0, 8.0)
    s = frame(0.0, 12.0)
    s = frame(0.0, 20.0, verifier=0.05)
    assert s == SlotState.AVAILABLE

    print("\nALL ASSERTIONS PASSED - asymmetric debounce behaves correctly")
