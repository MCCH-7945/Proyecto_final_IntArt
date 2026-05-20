"""Optional video and review-frame annotation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

try:
    from .io_utils import ensure_dir, ensure_parent_dir
except ImportError:  # pragma: no cover - allows running modules as scripts
    from io_utils import ensure_dir, ensure_parent_dir


def _draw_polygon(frame: np.ndarray, points: list[list[float]], color: tuple[int, int, int], thickness: int = 2) -> None:
    if not points:
        return
    pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=thickness)


def _draw_goal(frame: np.ndarray, goal_corners: dict[str, Any] | None) -> None:
    if not goal_corners:
        return
    order = ["bottom_left", "bottom_right", "top_right", "top_left"]
    if not all(key in goal_corners for key in order):
        return
    _draw_polygon(frame, [goal_corners[key] for key in order], (255, 140, 0), 2)


def _draw_text(frame: np.ndarray, text: str, origin: tuple[int, int], scale: float = 0.55) -> None:
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def _is_finite_number(value: Any) -> bool:
    try:
        return bool(np.isfinite(value))
    except (TypeError, ValueError):
        return False


def _row_for_frame(frame_df: pd.DataFrame, frame_number: int) -> pd.Series | None:
    rows = frame_df[frame_df["frame"] == frame_number]
    if rows.empty:
        return None
    return rows.iloc[0]


def _draw_frame_annotations(
    frame: np.ndarray,
    frame_number: int,
    frame_df: pd.DataFrame,
    shot_row: dict[str, Any],
    config: dict[str, Any] | None,
    trail: int = 15,
) -> np.ndarray:
    output = frame.copy()
    if config:
        _draw_polygon(output, config.get("penalty_area_polygon"), (0, 180, 0), 2)
        _draw_goal(output, config.get("goal_corners"))

    start = max(0, frame_number - trail)
    trail_df = frame_df[(frame_df["frame"] >= start) & (frame_df["frame"] <= frame_number)]
    points = []
    for _, row in trail_df.iterrows():
        x = row.get("ball_x_smooth", row.get("ball_x"))
        y = row.get("ball_y_smooth", row.get("ball_y"))
        if np.isfinite(x) and np.isfinite(y):
            points.append((int(round(x)), int(round(y))))
    for p0, p1 in zip(points[:-1], points[1:]):
        cv2.line(output, p0, p1, (0, 220, 255), 2)

    row = _row_for_frame(frame_df, frame_number)
    if row is not None:
        x = row.get("ball_x_smooth", row.get("ball_x"))
        y = row.get("ball_y_smooth", row.get("ball_y"))
        if np.isfinite(x) and np.isfinite(y):
            cv2.circle(output, (int(round(x)), int(round(y))), 6, (0, 0, 255), -1)
        speed = row.get("ball_speed")
        candidate = bool(row.get("is_candidate_shot_frame", False))
        selected = bool(row.get("is_selected_shot_frame", False))
        play_end = bool(row.get("is_play_end_frame", False))
        _draw_text(output, f"Frame {frame_number}  t={row.get('time_sec', 0):.2f}s", (18, 30))
        _draw_text(output, f"speed={speed:.1f}px/s" if _is_finite_number(speed) else "speed=NA", (18, 56))
        if candidate:
            _draw_text(output, "candidate shot frame", (18, 82), scale=0.5)
        if selected:
            cv2.rectangle(output, (8, 8), (output.shape[1] - 8, output.shape[0] - 8), (0, 0, 255), 4)
            _draw_text(output, "SELECTED SHOT", (18, 108), scale=0.6)
        if play_end:
            cv2.rectangle(output, (16, 16), (output.shape[1] - 16, output.shape[0] - 16), (255, 0, 255), 3)
            outcome = shot_row.get("play_outcome", "unknown")
            _draw_text(output, f"PLAY END: {outcome}", (18, 134 if selected else 108), scale=0.6)

    entry_x = shot_row.get("goal_entry_x_px")
    entry_y = shot_row.get("goal_entry_y_px")
    if entry_x is not None and entry_y is not None and _is_finite_number(entry_x) and _is_finite_number(entry_y):
        center = (int(round(entry_x)), int(round(entry_y)))
        cv2.drawMarker(output, center, (255, 0, 255), cv2.MARKER_CROSS, markerSize=18, thickness=2)
        u = shot_row.get("goal_entry_u")
        v = shot_row.get("goal_entry_v")
        conf = shot_row.get("goal_entry_confidence")
        if u is not None and v is not None and _is_finite_number(u) and _is_finite_number(v):
            _draw_text(output, f"goal u={u:.2f} v={v:.2f} conf={conf:.2f}", (18, output.shape[0] - 22), scale=0.5)

    shot_conf = shot_row.get("shot_confidence")
    if shot_conf is not None and _is_finite_number(shot_conf):
        _draw_text(output, f"shot conf={shot_conf:.2f}", (18, output.shape[0] - 48), scale=0.5)
    return output


def annotate_video(
    video_path: str | Path,
    frame_df: pd.DataFrame,
    shot_row: dict[str, Any],
    config: dict[str, Any] | None,
    output_path: str | Path,
    trail: int = 15,
) -> None:
    """Save an annotated video for manual inspection."""
    ensure_parent_dir(output_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError("Video could not be opened for annotation.")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_number = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        annotated = _draw_frame_annotations(frame, frame_number, frame_df, shot_row, config, trail=trail)
        writer.write(annotated)
        frame_number += 1

    writer.release()
    capture.release()


def save_review_frames(
    video_path: str | Path,
    frame_df: pd.DataFrame,
    shot_row: dict[str, Any],
    config: dict[str, Any] | None,
    output_dir: str | Path,
    frames_each_side: int = 15,
) -> None:
    """Save annotated still frames around the selected shot frame."""
    shot_frame = shot_row.get("shot_frame")
    if shot_frame is None or not _is_finite_number(shot_frame):
        return

    ensure_dir(output_dir)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError("Video could not be opened for review frames.")

    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames_to_save: set[int] = set()
    event_frames = [int(shot_frame)]
    play_end_frame = shot_row.get("play_end_frame")
    if play_end_frame is not None and _is_finite_number(play_end_frame):
        event_frames.append(int(play_end_frame))

    for event_frame in event_frames:
        start = max(0, event_frame - frames_each_side)
        end = event_frame + frames_each_side
        if total > 0:
            end = min(total - 1, end)
        frames_to_save.update(range(start, end + 1))

    for frame_number in sorted(frames_to_save):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, frame = capture.read()
        if not ok:
            continue
        annotated = _draw_frame_annotations(frame, frame_number, frame_df, shot_row, config, trail=frames_each_side)
        output_path = Path(output_dir) / f"frame_{frame_number:06d}.jpg"
        cv2.imwrite(str(output_path), annotated)

    capture.release()
