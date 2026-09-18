# ByteTrack Integration Pipeline

## Read this first: there is no training step

ByteTrack has no trainable weights. It's an algorithm — Kalman-filter
motion prediction + Hungarian/IoU matching — that runs entirely on top
of the detections your already-trained YOLOv8 model produces. Don't go
looking for a `train.py` here; there isn't one, and there shouldn't be.

What actually needed building instead:

1. **An evaluation dataset** (`prepare_eval_dataset.py`) — footage with
   ground-truth track IDs, so tracking quality can be *measured*, not
   assumed.
2. **A way to measure tracking quality** (`evaluate_tracking.py`) — ID
   switches, track fragmentation, MOTA, IDF1, using the real `motmetrics`
   library.
3. **Hyperparameter tuning** (`tune_hyperparameters.py`) — the closest
   legitimate equivalent to "training": a grid search over ByteTrack's
   handful of knobs (`lost_track_buffer`, `track_activation_threshold`,
   `minimum_iou_threshold`) against your evaluation set.
4. **Production inference** (`inference.py`) — the actual pipeline:
   detector → ByteTrack → your ground-plane occupancy engine → the
   `(track_id, slot_id, event_type, timestamp)` event contract from
   Week 3 of the ML roadmap.

## Two things worth knowing, found while building and testing this

- **`supervision`'s `ByteTrack` class is deprecated** (removal scheduled
  for v0.31.0). The maintainers moved tracking into a standalone
  `trackers` package (`pip install trackers`, `ByteTrackTracker`,
  `.update()` instead of `.update_with_detections()`). This code uses
  the current package so it doesn't rot in a few weeks.
- **The official UA-DETRAC download page is known to be unreliable**
  (confirmed via a live GitHub issue where users report broken links).
  `prepare_eval_dataset.py` does not fetch it for you — hand it a
  locally-extracted folder (get it from a Kaggle/Hugging Face mirror if
  you want it) or skip it entirely and use your own CVAT-labeled footage
  instead, which is the more useful evaluation set anyway.

## Files

- `prepare_eval_dataset.py` — converts UA-DETRAC XML *or* a CVAT "MOT
  1.1" export into standard MOTChallenge `gt.txt` format.
- `tracker.py` — thin wrapper around `ByteTrackTracker`.
- `evaluate_tracking.py` — runs the tracker against a sequence and
  scores it with real MOT metrics.
- `tune_hyperparameters.py` — grid search over tracker hyperparameters.
- `occupancy_engine.py` — the ground-plane occupancy state machine
  (from earlier in this build).
- `detector.py` — swap `MockDetector` for `YOLODetector` once you have
  trained weights.
- `inference.py` — the full production pipeline.

## Quickstart

```bash
pip install supervision trackers motmetrics shapely opencv-python-headless

# 1. Label a few clips of your own footage in CVAT (export as "MOT 1.1"),
#    or point at an extracted UA-DETRAC folder instead:
python prepare_eval_dataset.py cvat path/to/cvat_export eval_data --sequence-name lot_cam_01

# 2. Measure current tracking quality:
python evaluate_tracking.py eval_data/lot_cam_01

# 3. Tune hyperparameters against your own footage's occlusion patterns:
python tune_hyperparameters.py eval_data/lot_cam_01

# 4. Run the real pipeline (swap MockDetector for YOLODetector in detector.py first):
python inference.py
```

## Verified while building this

- `sv.ByteTrack` deprecation caught by actually importing it, not by
  memory — the old API silently doesn't accept the parameter names
  documented in older tutorials.
- A real bug: the first draft of `evaluate_tracking.py` only called
  `tracker.update()` on frames that had a detection, which meant
  ByteTrack's occlusion-tolerance countdown (`lost_track_buffer`) never
  actually elapsed during a simulated occlusion gap, silently hiding the
  exact failure mode this tool exists to catch. Fixed to call `update()`
  on every frame in range, including empty ones — confirmed with a
  before/after test showing 1 ID switch (IDF1 0.48) with too short a
  buffer vs. 0 switches (IDF1 0.91) with a correctly tuned one.
