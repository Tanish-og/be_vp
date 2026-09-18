"""Run the parking pipeline against a local video or RTSP stream."""

import argparse
import json
import time
from urllib.error import URLError
from urllib.request import Request, urlopen
from pathlib import Path

import cv2

from detector import YOLODetector
from inference import ParkingPerceptionPipeline


def send_event(event: dict, backend_url: str):
    request = Request(
        backend_url,
        data=json.dumps(event).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=5) as response:
            if response.status >= 300:
                raise RuntimeError(f"Backend returned HTTP {response.status}")
    except (OSError, URLError) as error:
        print(f"Could not deliver event to backend: {error}", flush=True)


def load_slots(path: Path) -> dict:
    import json

    with path.open(encoding="utf-8") as slots_file:
        slot_data = json.load(slots_file)

    return {
        "homography": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "slots": [
            {"id": slot_id, "corners": corners}
            for slot_id, corners in slot_data.items()
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Run parking inference on video or RTSP")
    parser.add_argument("source", help="Video path or RTSP URL")
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument(
        "--slots",
        default=str(Path(__file__).resolve().parents[1] / "polygons" / "slots.json"),
    )
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument(
        "--backend-events-url",
        default="http://127.0.0.1:8002/parking-events",
        help="FastAPI endpoint receiving enter/exit events",
    )
    args = parser.parse_args()

    detector = YOLODetector(weights_path=args.weights, conf=0.25)
    pipeline = ParkingPerceptionPipeline(load_slots(Path(args.slots)), detector, {
        "lost_track_buffer": 5,
        "track_activation_threshold": 0.25,
        "minimum_iou_threshold": 0.1,
    })

    capture = cv2.VideoCapture(args.source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video or stream: {args.source}")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            events = pipeline.process_frame(frame, now=time.time())
            for event in events:
                print(event, flush=True)
                send_event(event, args.backend_events_url)

            if not args.no_display:
                cv2.imshow("Parking inference", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        capture.release()
        cv2.destroyAllWindows()

    print("Final slot states:", pipeline.current_slot_states())


if __name__ == "__main__":
    main()