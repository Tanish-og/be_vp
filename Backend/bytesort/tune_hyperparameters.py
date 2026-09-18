"""
tune_hyperparameters.py

ByteTrack has no weights to train, but it does have a handful of
hyperparameters worth tuning against your own footage - this is the
closest legitimate equivalent to a "training" step. Grid-searches
lost_track_buffer, track_activation_threshold, and
minimum_iou_threshold against a prepared evaluation sequence, and picks
the combination that minimizes ID switches (breaking ties on IDF1).

Usage:
    python tune_hyperparameters.py path/to/sequence_dir --detections path/to/detections.txt
"""

import argparse
import itertools
from pathlib import Path

from evaluate_tracking import load_mot_file, run_tracker_on_sequence, score


def grid_search(gt_frames, detections_path, frame_rate):
    lost_track_buffers = [5, 15, 30, 60]
    activation_thresholds = [0.25, 0.4, 0.6]
    iou_thresholds = [0.1, 0.3, 0.5]

    results = []
    for buf, act_thresh, iou_thresh in itertools.product(lost_track_buffers, activation_thresholds, iou_thresholds):
        kwargs = dict(
            lost_track_buffer=buf,
            track_activation_threshold=act_thresh,
            minimum_iou_threshold=iou_thresh,
            frame_rate=frame_rate,
        )
        pred_frames = run_tracker_on_sequence(gt_frames, kwargs, detections_path)
        result = score(gt_frames, pred_frames)
        results.append({
            "lost_track_buffer": buf,
            "track_activation_threshold": act_thresh,
            "minimum_iou_threshold": iou_thresh,
            "num_switches": int(result["num_switches"]),
            "idf1": float(result["idf1"]),
            "mota": float(result["mota"]),
        })

    # Primary objective: fewest ID switches (the thing occlusion breaks).
    # Tie-break on IDF1, since several configs can tie on switch count.
    results.sort(key=lambda r: (r["num_switches"], -r["idf1"]))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sequence_dir", type=str)
    parser.add_argument("--detections", type=str, default=None)
    parser.add_argument("--frame-rate", type=float, default=10.0)
    args = parser.parse_args()

    gt_frames = load_mot_file(Path(args.sequence_dir) / "gt" / "gt.txt")
    detections_path = Path(args.detections) if args.detections else None

    results = grid_search(gt_frames, detections_path, args.frame_rate)

    print(f"{'buffer':>8} {'act_thr':>8} {'iou_thr':>8} {'switches':>9} {'idf1':>7} {'mota':>7}")
    for r in results:
        print(f"{r['lost_track_buffer']:>8} {r['track_activation_threshold']:>8} "
              f"{r['minimum_iou_threshold']:>8} {r['num_switches']:>9} "
              f"{r['idf1']:>7.3f} {r['mota']:>7.3f}")

    best = results[0]
    print(f"\nBest config: lost_track_buffer={best['lost_track_buffer']}, "
          f"track_activation_threshold={best['track_activation_threshold']}, "
          f"minimum_iou_threshold={best['minimum_iou_threshold']} "
          f"-> {best['num_switches']} switches, IDF1={best['idf1']:.3f}")


if __name__ == "__main__":
    main()
