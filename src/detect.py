import argparse
import os
from pathlib import Path
from typing import Any

import cv2
from dotenv import load_dotenv
from inference import get_model

import alert


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time driver phone detection")
    parser.add_argument(
        "--source",
        default="0",
        help="Video source: webcam index (e.g. 0) or path to video file",
    )
    parser.add_argument(
        "--rtsp",
        default=None,
        help="RTSP stream URL (overrides --source), e.g. rtsp://user:pass@host/path",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Confidence threshold for alerting",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without cv2.imshow (for servers)",
    )
    return parser.parse_args()


def _parse_source(source_arg: str) -> int | str:
    if isinstance(source_arg, int):
        return source_arg
    source_text = str(source_arg).strip()
    if source_text.isdigit():
        return int(source_text)
    return source_text


def _open_capture(source: int | str, is_rtsp: bool) -> cv2.VideoCapture:
    # RTSP streams are more reliable with explicit FFMPEG backend when available.
    if is_rtsp:
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap = cv2.VideoCapture(source)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap
    return cv2.VideoCapture(source)


def _prediction_value(prediction: Any, key: str, default: Any = None) -> Any:
    if isinstance(prediction, dict):
        return prediction.get(key, default)
    return getattr(prediction, key, default)


def _extract_predictions(result: Any) -> list[Any]:
    if isinstance(result, dict):
        predictions = result.get("predictions", [])
        return predictions if isinstance(predictions, list) else []
    predictions = getattr(result, "predictions", [])
    return predictions if isinstance(predictions, list) else []


def _draw_safe_label(frame) -> None:
    cv2.putText(
        frame,
        "SAFE",
        (15, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def main() -> int:
    args = _parse_args()

    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(dotenv_path=env_path)
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        print("Error: ROBOFLOW_API_KEY is missing. Add it to phone_detection/.env.")
        return 1

    source_input = args.rtsp if args.rtsp else args.source
    source = _parse_source(source_input)
    is_rtsp = isinstance(source, str) and source.lower().startswith("rtsp://")

    try:
        model = get_model(model_id="phone-using-dhjqe/3", api_key=api_key)
    except Exception as exc:
        print(f"Error: failed to load Roboflow model: {exc}")
        return 1

    cap = _open_capture(source=source, is_rtsp=is_rtsp)
    if not cap.isOpened():
        print(f"Error: unable to open source: {source_input}")
        if is_rtsp:
            print(
                "Hint: verify RTSP URL, camera credentials, and that your machine can reach the camera IP."
            )
        return 1

    print(f"Driver phone detection started on source: {source_input}")
    if not args.headless:
        print("Press 'q' to quit.")
    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            try:
                result = model.infer(image=frame)[0]
            except Exception as exc:
                print(f"Error: inference failed: {exc}")
                break

            predictions = _extract_predictions(result)
            thresholded = [
                p
                for p in predictions
                if float(_prediction_value(p, "confidence", 0.0)) >= float(args.conf)
            ]

            if thresholded:
                frame = alert.trigger(frame, detections=thresholded)
            else:
                _draw_safe_label(frame)

            if not args.headless:
                cv2.imshow("Driver Phone Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
