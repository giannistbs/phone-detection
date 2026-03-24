import os
from pathlib import Path
from typing import Any

import cv2
from dotenv import load_dotenv
from inference import get_model

import alert


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
    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(dotenv_path=env_path)
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        print("Error: ROBOFLOW_API_KEY is missing. Add it to phone_detection/.env.")
        return 1

    try:
        model = get_model(model_id="phone-using-dhjqe/3", api_key=api_key)
    except Exception as exc:
        print(f"Error: failed to load Roboflow model: {exc}")
        return 1

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: unable to open webcam.")
        return 1

    print("Driver phone detection started. Press 'q' to quit.")
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
                p for p in predictions
                if float(_prediction_value(p, "confidence", 0.0)) >= 0.5
            ]

            if thresholded:
                frame = alert.trigger(frame, detections=thresholded)
            else:
                _draw_safe_label(frame)

            cv2.imshow("Driver Phone Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
