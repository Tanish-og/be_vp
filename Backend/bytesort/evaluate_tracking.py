"""
evaluate_tracking.py

This is the actual "did tracking work" measurement - the Week 2
deliverable from the roadmap (ID-switch rate, track fragmentation).

By default it runs the tracker on the ground-truth boxes themselves
("oracle detections") rather than a real detector's output. That's
deliberate: it isolates the TRACKER's association logic from the
DETECTOR's accuracy, so an ID-switch you see here is genuinely a
tracking-algorithm problem (or a hyperparameter problem), not your
detector missing a car. Pass --detections to evaluate against a real
detector's output instead, once you have one recorded for this sequence.

Usage:
    python evaluate_tracking.py path/to/sequence_dir --lost-track-buffer 30
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import motmetrics as mm

from tracker import build_tracker, detections_from_boxes


def load_mot_file(path: Path) -> dict:
    """frame_num -> list of (track_id, x1, y1, x2, y2)"""
    frames = defaultdict(list)
    with open(path) as f:
        for row in csv.reader(f):
            frame, track_id, left, top, w, h = int(row[0]), int(row[1]), *map(float, row[2:6])
            frames[frame].append((track_id, left, top, left + w, top + h))
    return frames


def run_tracker_on_sequence(gt_frames: dict, tracker_kwargs: dict, detections_path: Path = None) -> dict:
    """Returns pred_frames in the same shape as load_mot_file's output,
    built by running our tracker over either the ground-truth boxes
    (oracle mode) or a separately-provided detections file.

    Critically, the tracker is called on EVERY frame in the sequence's
    range - including frames with zero detections - not just frames
    that happen to have a box. ByteTrack's occlusion tolerance
    (lost_track_buffer) counts elapsed calls to update(), so skipping
    empty frames silently breaks the buffer timing and hides exactly
    the failure mode this evaluation exists to catch."""
    input_frames = load_mot_file(detections_path) if detections_path else gt_frames
    tracker = build_tracker(**tracker_kwargs)

    all_frame_nums = range(min(gt_frames.keys()), max(gt_frames.keys()) + 1)

    pred_frames = {}
    for frame_num in all_frame_nums:
        boxes = [(x1, y1, x2, y2) for (_, x1, y1, x2, y2) in input_frames.get(frame_num, [])]
        dets = detections_from_boxes(boxes)
        tracked = tracker.update(dets)
        pred_frames[frame_num] = [
            (int(tid), *box) for tid, box in zip(tracked.tracker_id, tracked.xyxy.tolist())
            if tid != -1  # -1 = not yet confirmed as a stable track
        ]
    return pred_frames


def score(gt_frames: dict, pred_frames: dict):
    acc = mm.MOTAccumulator(auto_id=True)
    for frame_num in sorted(gt_frames.keys()):
        gt = gt_frames.get(frame_num, [])
        pred = pred_frames.get(frame_num, [])

        gt_ids = [g[0] for g in gt]
        pred_ids = [p[0] for p in pred]
        # centroid distance as the matching cost (simpler and dependency-light
        # compared to IoU distance here; swap for mm.distances.iou_matrix
        # if you want strict box-overlap-based matching instead)
        gt_centroids = [((g[1] + g[3]) / 2, (g[2] + g[4]) / 2) for g in gt]
        pred_centroids = [((p[1] + p[3]) / 2, (p[2] + p[4]) / 2) for p in pred]
        dist_matrix = mm.distances.norm2squared_matrix(gt_centroids, pred_centroids, max_d2=2500)

        acc.update(gt_ids, pred_ids, dist_matrix)

    mh = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=["num_frames", "mota", "idf1", "num_switches", "num_fragmentations", "num_misses", "num_false_positives"],
        name="sequence",
    )
    return summary.iloc[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sequence_dir", type=str, help="Folder containing gt/gt.txt (from prepare_eval_dataset.py)")
    parser.add_argument("--detections", type=str, default=None, help="Optional: a real detector's output file, same MOT format. Defaults to oracle mode (ground-truth boxes as input).")
    parser.add_argument("--track-activation-threshold", type=float, default=0.25)
    parser.add_argument("--lost-track-buffer", type=int, default=30)
    parser.add_argument("--minimum-iou-threshold", type=float, default=0.3)
    parser.add_argument("--frame-rate", type=float, default=10.0)
    args = parser.parse_args()

    gt_path = Path(args.sequence_dir) / "gt" / "gt.txt"
    gt_frames = load_mot_file(gt_path)

    tracker_kwargs = dict(
        track_activation_threshold=args.track_activation_threshold,
        lost_track_buffer=args.lost_track_buffer,
        minimum_iou_threshold=args.minimum_iou_threshold,
        frame_rate=args.frame_rate,
    )
    pred_frames = run_tracker_on_sequence(gt_frames, tracker_kwargs, Path(args.detections) if args.detections else None)
    result = score(gt_frames, pred_frames)

    print(f"Sequence: {args.sequence_dir}")
    print(f"  Frames:              {int(result['num_frames'])}")
    print(f"  MOTA:                {result['mota']:.3f}")
    print(f"  IDF1:                {result['idf1']:.3f}")
    print(f"  ID switches:         {int(result['num_switches'])}")
    print(f"  Fragmentations:      {int(result['num_fragmentations'])}")
    print(f"  Missed detections:   {int(result['num_misses'])}")
    print(f"  False positives:     {int(result['num_false_positives'])}")


if __name__ == "__main__":
    main()
