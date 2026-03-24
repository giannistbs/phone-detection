import os
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from playsound import playsound

try:
    import supervision as sv
except Exception:  # pragma: no cover - optional dependency fallback
    sv = None

_BEEP_COOLDOWN_SECONDS = 3.0
_last_beep_time = 0.0
_warned_missing_sound = False


def _prediction_value(prediction: Any, key: str, default: Any = None) -> Any:
    """Read prediction values from object-style or dict-style payloads."""
    if isinstance(prediction, dict):
        return prediction.get(key, default)
    return getattr(prediction, key, default)


def _to_xyxy(prediction: Any) -> tuple[int, int, int, int]:
    """Convert center-width-height style box to xyxy."""
    x = float(_prediction_value(prediction, "x", 0))
    y = float(_prediction_value(prediction, "y", 0))
    width = float(_prediction_value(prediction, "width", 0))
    height = float(_prediction_value(prediction, "height", 0))

    x1 = int(x - (width / 2))
    y1 = int(y - (height / 2))
    x2 = int(x + (width / 2))
    y2 = int(y + (height / 2))
    return x1, y1, x2, y2


def _label_for_prediction(prediction: Any) -> str:
    class_name = _prediction_value(prediction, "class_name") or _prediction_value(
        prediction, "class", "phone"
    )
    confidence = float(_prediction_value(prediction, "confidence", 0.0))
    return f"{class_name} {confidence * 100:.1f}%"


def _default_sound_path() -> str | None:
    env_sound = os.getenv("ALERT_SOUND_PATH")
    if env_sound and Path(env_sound).exists():
        return env_sound

    candidates = []
    if os.name == "nt":
        candidates = [r"C:\Windows\Media\Windows Notify.wav"]
    elif os.uname().sysname == "Darwin":
        candidates = ["/System/Library/Sounds/Ping.aiff", "/System/Library/Sounds/Glass.aiff"]
    else:
        candidates = [
            "/usr/share/sounds/freedesktop/stereo/alarm-clock-elapsed.oga",
            "/usr/share/sounds/freedesktop/stereo/complete.oga",
        ]

    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    return None


def _play_beep_non_blocking() -> None:
    sound_path = _default_sound_path()
    global _warned_missing_sound
    if sound_path is None:
        if not _warned_missing_sound:
            print(
                "Warning: no alert sound file found. Set ALERT_SOUND_PATH in your environment to enable audio."
            )
            _warned_missing_sound = True
        return

    def _worker() -> None:
        try:
            playsound(sound_path)
        except Exception as exc:
            print(f"Warning: failed to play alert sound ({exc}).")

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()


def _draw_banner(frame: np.ndarray) -> np.ndarray:
    output = frame.copy()
    overlay = output.copy()
    banner_height = 70
    cv2.rectangle(overlay, (0, 0), (output.shape[1], banner_height), (0, 0, 255), -1)
    cv2.addWeighted(overlay, 0.4, output, 0.6, 0, output)
    cv2.putText(
        output,
        "PHONE DETECTED",
        (15, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def _draw_boxes_with_opencv(frame: np.ndarray, detections: list[Any]) -> np.ndarray:
    output = _draw_banner(frame)
    for prediction in detections:
        x1, y1, x2, y2 = _to_xyxy(prediction)
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            output,
            _label_for_prediction(prediction),
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    return output


def _draw_boxes_with_supervision(frame: np.ndarray, detections: list[Any]) -> np.ndarray:
    output = _draw_banner(frame)
    xyxy = np.array([_to_xyxy(prediction) for prediction in detections], dtype=np.float32)
    confidence = np.array(
        [float(_prediction_value(prediction, "confidence", 0.0)) for prediction in detections],
        dtype=np.float32,
    )
    class_names = [
        _prediction_value(prediction, "class_name") or _prediction_value(prediction, "class", "phone")
        for prediction in detections
    ]
    class_id = np.arange(len(detections), dtype=np.int32)
    sv_detections = sv.Detections(
        xyxy=xyxy,
        confidence=confidence,
        class_id=class_id,
        data={"class_name": class_names},
    )
    labels = [_label_for_prediction(prediction) for prediction in detections]
    box_annotator = sv.BoxAnnotator(color=sv.Color.RED, thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1, color=sv.Color.RED)
    output = box_annotator.annotate(scene=output, detections=sv_detections)
    output = label_annotator.annotate(scene=output, detections=sv_detections, labels=labels)
    return output


def trigger(frame: np.ndarray, detections: list[Any]) -> np.ndarray:
    """Annotate detections and play a debounced non-blocking alert beep."""
    global _last_beep_time

    now = time.time()
    if now - _last_beep_time >= _BEEP_COOLDOWN_SECONDS:
        _play_beep_non_blocking()
        _last_beep_time = now

    if sv is not None:
        try:
            return _draw_boxes_with_supervision(frame, detections)
        except Exception:
            # Supervision is optional; fallback keeps runtime resilient.
            return _draw_boxes_with_opencv(frame, detections)
    return _draw_boxes_with_opencv(frame, detections)
