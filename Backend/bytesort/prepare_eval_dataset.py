"""
prepare_eval_dataset.py

IMPORTANT CONTEXT: ByteTrack has no trainable weights, so there's no
"training set" in the usual sense. What this script prepares is an
EVALUATION set: footage with ground-truth track IDs, so you can measure
whether tracking is actually working (ID-switch rate, fragmentation) and
tune its few hyperparameters against real numbers.

Two sources, in the order you should actually use them:

1. YOUR OWN FOOTAGE, hand-labeled in CVAT (recommended, do this first).
   CVAT has a built-in "MOT 1.1" export format for track annotations -
   label a few short clips (a car parking, a car getting occluded by
   another vehicle, a car leaving) with persistent track IDs, export as
   MOT 1.1, and this script just validates/normalizes the result. This
   matters more than any public benchmark: a generic dataset won't tell
   you anything about how tracking behaves on YOUR camera's angle and
   occlusion patterns, which is the whole point of Week 2 in the roadmap.

2. UA-DETRAC, a public vehicle-tracking benchmark (optional, for a
   general sanity-check of the tracking code itself before you trust it
   on your own footage). One honest caveat: the official UA-DETRAC
   download page is known to be unreliable (see e.g. github.com/
   captaineven/fairmotvehicle/issues/17 - "annotations download links
   all return errors"). Get it from a mirror instead - it's indexed on
   Kaggle and Hugging Face under names like "detrac-dataset" or
   "UA-DETRAC" - then point this script at wherever you extracted it.
   This script does NOT fetch it for you; hand it a local folder.

Both paths converge on the same output: one gt.txt per sequence, in
standard MOTChallenge format, which evaluate_tracking.py and
tune_hyperparameters.py both consume.

MOTChallenge gt.txt column format (comma-separated, no header):
    frame, id, bb_left, bb_top, width, height, confidence, class, visibility
"""

import argparse
import csv
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


def convert_ua_detrac_sequence(xml_path: Path, out_dir: Path):
    """Parse one UA-DETRAC sequence XML and write it as gt.txt.

    UA-DETRAC's annotation XML nests <frame num="N"><target_list>
    <target id="ID"><box left top width height/></target></target_list>
    </frame> inside a <sequence> root. Mirrors occasionally repackage
    this slightly differently (e.g. a wrapping <sequence_list>), so this
    parser searches for <frame> anywhere in the tree rather than
    assuming a fixed depth.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    sequence_name = root.attrib.get("name", xml_path.stem)

    rows = []
    for frame_elem in root.iter("frame"):
        frame_num = int(frame_elem.attrib["num"])
        for target in frame_elem.iter("target"):
            track_id = int(target.attrib["id"])
            box = target.find("box")
            if box is None:
                continue
            left = float(box.attrib["left"])
            top = float(box.attrib["top"])
            width = float(box.attrib["width"])
            height = float(box.attrib["height"])
            # MOTChallenge: frame, id, left, top, w, h, conf, class, visibility, unused
            rows.append([frame_num, track_id, left, top, width, height, 1, 1, 1, -1])

    if not rows:
        raise ValueError(
            f"No <frame>/<target>/<box> elements found in {xml_path}. "
            "The mirror you downloaded may use a different XML layout - "
            "open the file and check the tag names against the ones this "
            "parser looks for (frame, target, box)."
        )

    rows.sort(key=lambda r: (r[0], r[1]))
    seq_dir = out_dir / sequence_name / "gt"
    seq_dir.mkdir(parents=True, exist_ok=True)
    with open(seq_dir / "gt.txt", "w", newline="") as f:
        csv.writer(f).writerows(rows)

    num_frames = len({r[0] for r in rows})
    print(f"{sequence_name}: {len(rows)} annotated boxes across {num_frames} frames -> {seq_dir / 'gt.txt'}")


def convert_ua_detrac_folder(annotations_dir: Path, out_dir: Path):
    xml_files = sorted(annotations_dir.glob("*.xml"))
    if not xml_files:
        raise SystemExit(f"No .xml files found under {annotations_dir}")
    for xml_path in xml_files:
        convert_ua_detrac_sequence(xml_path, out_dir)


def normalize_cvat_mot_export(cvat_export_dir: Path, out_dir: Path, sequence_name: str):
    """CVAT's 'MOT 1.1' export already produces a gt.txt in the right
    format, usually at <export_dir>/gt/gt.txt. This just copies it into
    our standard <out_dir>/<sequence_name>/gt/gt.txt layout so every
    sequence - CVAT-sourced or UA-DETRAC-sourced - looks the same to
    evaluate_tracking.py."""
    candidates = list(cvat_export_dir.rglob("gt.txt"))
    if not candidates:
        raise SystemExit(
            f"No gt.txt found under {cvat_export_dir}. Confirm you exported "
            "from CVAT using the 'MOT 1.1' format specifically (not COCO, "
            "not CVAT XML)."
        )
    src = candidates[0]
    dest_dir = out_dir / sequence_name / "gt"
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest_dir / "gt.txt")
    print(f"{sequence_name}: copied {src} -> {dest_dir / 'gt.txt'}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="source", required=True)

    ua = sub.add_parser("ua-detrac", help="Convert a locally-extracted UA-DETRAC annotations folder")
    ua.add_argument("annotations_dir", type=str, help="Folder containing UA-DETRAC *.xml sequence annotations")
    ua.add_argument("out_dir", type=str)

    cvat = sub.add_parser("cvat", help="Normalize a CVAT 'MOT 1.1' export of your own labeled footage")
    cvat.add_argument("export_dir", type=str, help="The folder CVAT exported")
    cvat.add_argument("out_dir", type=str)
    cvat.add_argument("--sequence-name", type=str, required=True)

    args = parser.parse_args()
    out_dir = Path(args.out_dir)

    if args.source == "ua-detrac":
        convert_ua_detrac_folder(Path(args.annotations_dir), out_dir)
    else:
        normalize_cvat_mot_export(Path(args.export_dir), out_dir, args.sequence_name)


if __name__ == "__main__":
    main()
