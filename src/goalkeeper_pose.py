"""Goalkeeper five-point pose schema and goal-relative normalization.

This module does not run pose estimation yet. It defines the data contract and
geometry helpers that Stage 2 can use once keeper keypoints are detected or
annotated.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:
    from .goal_mapping import compute_goal_homography, map_point_to_goal_uv
except ImportError:  # pragma: no cover
    from goal_mapping import compute_goal_homography, map_point_to_goal_uv


BODY_POINTS = ["head", "left_hand", "right_hand", "left_foot", "right_foot"]
POSE_BASE_COLUMNS = [
    "video_id",
    "frame",
    "time_sec",
    "shot_frame",
    "frame_relative_to_shot",
    "play_end_frame",
    "is_shot_frame",
    "is_play_end_frame",
    "keeper_detected",
    "keeper_confidence",
    "keeper_pose_confidence",
    "pose_valid_points_count",
    "pose_missing_points_count",
]


def pose_point_columns(include_goal_uv: bool = True, include_confidence: bool = True) -> list[str]:
    """Return expected pose columns for the five keeper body points."""
    columns: list[str] = []
    for point in BODY_POINTS:
        columns.extend([f"{point}_x", f"{point}_y"])
        if include_goal_uv:
            columns.extend([f"{point}_u", f"{point}_v"])
        if include_confidence:
            columns.append(f"{point}_confidence")
    return columns


def keeper_pose_columns() -> list[str]:
    """Return the complete frame-level keeper-pose schema."""
    return POSE_BASE_COLUMNS + pose_point_columns()


def empty_keeper_pose_frame(video_id: str | None = None) -> pd.DataFrame:
    """Create an empty keeper pose dataframe with the expected columns."""
    df = pd.DataFrame(columns=keeper_pose_columns())
    if video_id is not None:
        df["video_id"] = pd.Series(dtype="object")
    return df


def build_pose_window_index(
    shot_row: dict[str, Any] | pd.Series,
    fps: float,
    pre_frames: int = 30,
    post_frames: int = 30,
) -> pd.DataFrame:
    """Create a frame index around a shot for future keeper-pose extraction."""
    shot_frame = shot_row.get("shot_frame")
    if shot_frame is None or not np.isfinite(shot_frame):
        return empty_keeper_pose_frame(str(shot_row.get("video_id", "")))

    video_id = str(shot_row.get("video_id", ""))
    shot_frame = int(shot_frame)
    play_end_frame = shot_row.get("play_end_frame")
    if play_end_frame is not None and np.isfinite(play_end_frame):
        end_frame = max(shot_frame, int(play_end_frame))
    else:
        end_frame = shot_frame + int(post_frames)
    start_frame = max(0, shot_frame - int(pre_frames))

    frames = list(range(start_frame, end_frame + 1))
    df = pd.DataFrame(
        {
            "video_id": video_id,
            "frame": frames,
            "time_sec": [frame / fps if fps > 0 else np.nan for frame in frames],
            "shot_frame": shot_frame,
            "frame_relative_to_shot": [frame - shot_frame for frame in frames],
            "play_end_frame": play_end_frame,
            "is_shot_frame": [frame == shot_frame for frame in frames],
            "is_play_end_frame": [play_end_frame is not None and frame == int(play_end_frame) for frame in frames],
            "keeper_detected": False,
            "keeper_confidence": 0.0,
            "keeper_pose_confidence": 0.0,
            "pose_valid_points_count": 0,
            "pose_missing_points_count": len(BODY_POINTS),
        }
    )
    for column in pose_point_columns():
        df[column] = np.nan
    return df[keeper_pose_columns()]


def add_goal_relative_pose_coordinates(
    pose_df: pd.DataFrame,
    goal_corners: dict[str, Any] | None,
) -> pd.DataFrame:
    """Map keeper keypoints from image pixels to goal-normalized `(u, v)`.

    Values outside `[0, 1]` are preserved because they can encode useful keeper
    positioning relative to the goal frame.
    """
    output = pose_df.copy()
    for point in BODY_POINTS:
        output[f"{point}_u"] = np.nan
        output[f"{point}_v"] = np.nan

    if not goal_corners or output.empty:
        return output

    homography = compute_goal_homography(goal_corners)
    for index, row in output.iterrows():
        for point in BODY_POINTS:
            x = row.get(f"{point}_x")
            y = row.get(f"{point}_y")
            if x is None or y is None or not np.isfinite(x) or not np.isfinite(y):
                continue
            u, v = map_point_to_goal_uv((float(x), float(y)), homography)
            output.at[index, f"{point}_u"] = u
            output.at[index, f"{point}_v"] = v
    return output


def add_keeper_pose_quality(
    pose_df: pd.DataFrame,
    min_point_confidence: float = 0.25,
) -> pd.DataFrame:
    """Add per-frame pose quality columns from point-level confidences.

    The confidence is penalized when keypoints are missing: a frame with three
    high-confidence points is useful, but less complete than one with all five.
    """
    output = pose_df.copy()
    if output.empty:
        for column in ["keeper_pose_confidence", "pose_valid_points_count", "pose_missing_points_count"]:
            output[column] = pd.Series(dtype="float64")
        return output

    pose_confidences: list[float] = []
    valid_counts: list[int] = []
    missing_counts: list[int] = []
    for _, row in output.iterrows():
        point_confidences: list[float] = []
        valid_count = 0
        for point in BODY_POINTS:
            confidence = row.get(f"{point}_confidence")
            try:
                confidence = float(confidence)
            except (TypeError, ValueError):
                confidence = np.nan
            if np.isfinite(confidence):
                point_confidences.append(confidence)
                if confidence >= min_point_confidence:
                    valid_count += 1

        finite_mean = float(np.mean(point_confidences)) if point_confidences else 0.0
        completeness = valid_count / len(BODY_POINTS)
        keeper_confidence = row.get("keeper_confidence", 1.0)
        try:
            keeper_confidence = float(keeper_confidence)
        except (TypeError, ValueError):
            keeper_confidence = 1.0
        if not np.isfinite(keeper_confidence):
            keeper_confidence = 1.0

        pose_confidences.append(float(np.clip(finite_mean * completeness * keeper_confidence, 0.0, 1.0)))
        valid_counts.append(valid_count)
        missing_counts.append(len(BODY_POINTS) - valid_count)

    output["keeper_pose_confidence"] = pose_confidences
    output["pose_valid_points_count"] = valid_counts
    output["pose_missing_points_count"] = missing_counts
    output["keeper_detected"] = output["pose_valid_points_count"] > 0
    return output


def extract_goalkeeper_pose(*args, **kwargs):
    """Future extension point for model-backed keeper pose estimation."""
    raise NotImplementedError(
        "Goalkeeper pose extraction is not implemented yet. Use this module's schema helpers "
        "to prepare annotation or model outputs for Stage 2."
    )
