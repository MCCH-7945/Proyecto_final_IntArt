"""YOLO-based ball detection and frame-level tracking."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


class VideoOpenError(RuntimeError):
    """Raised when a video cannot be opened."""


def get_video_metadata(video_path: str | Path) -> dict[str, Any]:
    """Read basic video metadata with OpenCV."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise VideoOpenError("Video could not be opened.")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    capture.release()
    return {
        "fps": float(fps),
        "frame_count": frame_count,
        "width": width,
        "height": height,
    }


def _load_yolo_model(model_path: str | Path | None):
    if not model_path:
        return None, "Ball model path missing."
    if not Path(model_path).exists():
        return None, "Ball model file missing."
    try:
        from ultralytics import YOLO
    except Exception as exc:
        return None, f"ultralytics could not be imported: {exc}"
    try:
        return YOLO(str(model_path)), None
    except Exception as exc:
        return None, f"Ball model could not be loaded: {exc}"


def _class_name_for_box(model, class_id: int) -> str | None:
    names = getattr(model, "names", None)
    if isinstance(names, dict):
        return str(names.get(class_id, "")).lower()
    if isinstance(names, list) and 0 <= class_id < len(names):
        return str(names[class_id]).lower()
    return None


def _candidate_detections(result, model) -> list[dict[str, float]]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    detections: list[dict[str, float]] = []
    for box in boxes:
        xyxy = box.xyxy[0].detach().cpu().numpy().astype(float)
        confidence = float(box.conf[0].detach().cpu().item()) if box.conf is not None else 0.0
        class_id = int(box.cls[0].detach().cpu().item()) if box.cls is not None else -1
        class_name = _class_name_for_box(model, class_id)
        is_ball = class_name is None or class_name in {"ball", "sports ball", "soccer ball"}
        if not is_ball:
            continue
        x1, y1, x2, y2 = xyxy
        detections.append(
            {
                "x": float((x1 + x2) / 2.0),
                "y": float((y1 + y2) / 2.0),
                "confidence": confidence,
                "width": float(x2 - x1),
                "height": float(y2 - y1),
            }
        )

    # Ball-specific custom models often expose only one unnamed class.
    if not detections and len(boxes) > 0:
        for box in boxes:
            xyxy = box.xyxy[0].detach().cpu().numpy().astype(float)
            confidence = float(box.conf[0].detach().cpu().item()) if box.conf is not None else 0.0
            x1, y1, x2, y2 = xyxy
            detections.append(
                {
                    "x": float((x1 + x2) / 2.0),
                    "y": float((y1 + y2) / 2.0),
                    "confidence": confidence,
                    "width": float(x2 - x1),
                    "height": float(y2 - y1),
                }
            )
    return detections


def _select_detection(
    detections: list[dict[str, float]],
    previous_position: tuple[float, float] | None,
    frame_shape: tuple[int, int],
) -> dict[str, float] | None:
    if not detections:
        return None
    if previous_position is None:
        return max(detections, key=lambda det: det["confidence"])

    height, width = frame_shape
    diagonal = max(float(np.hypot(width, height)), 1.0)
    px, py = previous_position

    def score(det: dict[str, float]) -> float:
        distance = float(np.hypot(det["x"] - px, det["y"] - py))
        trajectory_penalty = min(distance / diagonal, 1.0) * 0.25
        return det["confidence"] - trajectory_penalty

    return max(detections, key=score)


def detect_ball_positions(
    video_path: str | Path,
    model_path: str | Path | None = None,
    video_id: str | None = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Detect and track ball position for every frame in a video.

    The detector uses YOLO when a model is available. If the model is missing
    or cannot be loaded, the function still emits one row per frame with
    `ball_detected = False`, allowing the pipeline to fail transparently.
    """
    video_path = Path(video_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise VideoOpenError("Video could not be opened.")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    resolved_video_id = video_id or video_path.stem

    model, model_note = _load_yolo_model(model_path)
    rows: list[dict[str, Any]] = []
    previous_position: tuple[float, float] | None = None
    frame_idx = 0
    iterator = tqdm(total=frame_count if frame_count > 0 else None, desc="Tracking ball", disable=not show_progress)

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        selected = None
        if model is not None:
            results = model(frame, verbose=False)
            result = results[0] if isinstance(results, list) else results
            detections = _candidate_detections(result, model)
            selected = _select_detection(detections, previous_position, frame.shape[:2])

        if selected is None:
            row = {
                "video_id": resolved_video_id,
                "frame": frame_idx,
                "time_sec": float(frame_idx / fps),
                "ball_x": np.nan,
                "ball_y": np.nan,
                "ball_detected": False,
                "ball_confidence": 0.0,
            }
        else:
            previous_position = (selected["x"], selected["y"])
            row = {
                "video_id": resolved_video_id,
                "frame": frame_idx,
                "time_sec": float(frame_idx / fps),
                "ball_x": selected["x"],
                "ball_y": selected["y"],
                "ball_detected": True,
                "ball_confidence": selected["confidence"],
            }
        rows.append(row)
        frame_idx += 1
        iterator.update(1)

    iterator.close()
    capture.release()

    output = pd.DataFrame(rows)
    output.attrs["fps"] = float(fps)
    output.attrs["frame_count"] = frame_idx
    output.attrs["width"] = width
    output.attrs["height"] = height
    if model_note:
        output.attrs["tracking_note"] = model_note
    return output
