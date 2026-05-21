"""Simplified goalkeeper body polygon features."""

from __future__ import annotations

import numpy as np
import pandas as pd
from shapely.geometry import Polygon

try:
    from .goalkeeper_pose import BODY_POINTS
except ImportError:  # pragma: no cover
    from goalkeeper_pose import BODY_POINTS


POLYGON_ORDER = ["left_hand", "head", "right_hand", "right_foot", "left_foot"]
POLYGON_FEATURE_COLUMNS = [
    "keeper_pose_valid",
    "keeper_polygon_confidence",
    "keeper_polygon_area",
    "keeper_polygon_area_uv",
    "keeper_body_width",
    "keeper_body_height",
    "keeper_hand_span",
    "keeper_foot_span",
    "keeper_center_x",
    "keeper_center_y",
    "keeper_center_u",
    "keeper_center_v",
    "keeper_body_aspect_ratio",
]


def _finite_pair(row: pd.Series, point: str, suffix_x: str, suffix_y: str) -> tuple[float, float] | None:
    x = row.get(f"{point}_{suffix_x}")
    y = row.get(f"{point}_{suffix_y}")
    if x is None or y is None:
        return None
    if not np.isfinite(x) or not np.isfinite(y):
        return None
    return float(x), float(y)


def _distance(a: tuple[float, float] | None, b: tuple[float, float] | None) -> float:
    if a is None or b is None:
        return np.nan
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _point_confidence(row: pd.Series, point: str) -> float:
    confidence = row.get(f"{point}_confidence")
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        return np.nan
    return confidence if np.isfinite(confidence) else np.nan


def _polygon_confidence(row: pd.Series, polygon_valid: bool, min_point_confidence: float = 0.25) -> float:
    if not polygon_valid:
        return 0.0
    confidences = [_point_confidence(row, point) for point in POLYGON_ORDER]
    finite = [confidence for confidence in confidences if np.isfinite(confidence)]
    if not finite:
        return 0.0
    valid_count = sum(confidence >= min_point_confidence for confidence in finite)
    completeness = valid_count / len(POLYGON_ORDER)
    return float(np.clip(float(np.mean(finite)) * completeness, 0.0, 1.0))


def build_keeper_polygon(
    row: pd.Series | dict,
    coordinate_space: str = "px",
) -> Polygon | None:
    """Build the observed five-point keeper polygon.

    `coordinate_space="px"` uses `*_x`, `*_y`. `coordinate_space="uv"` uses
    `*_u`, `*_v` in goal-normalized coordinates.
    """
    series = pd.Series(row)
    if coordinate_space == "px":
        suffix_x, suffix_y = "x", "y"
    elif coordinate_space == "uv":
        suffix_x, suffix_y = "u", "v"
    else:
        raise ValueError("coordinate_space must be 'px' or 'uv'.")

    points = [_finite_pair(series, point, suffix_x, suffix_y) for point in POLYGON_ORDER]
    if any(point is None for point in points):
        return None

    polygon = Polygon(points)
    if polygon.is_empty:
        return None
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if polygon.is_empty:
        return None
    return polygon


def polygon_feature_row(row: pd.Series | dict) -> dict[str, float | bool]:
    """Compute geometric keeper-polygon features for one pose row."""
    series = pd.Series(row)
    px_points = {point: _finite_pair(series, point, "x", "y") for point in BODY_POINTS}
    uv_points = {point: _finite_pair(series, point, "u", "v") for point in BODY_POINTS}
    px_polygon = build_keeper_polygon(series, coordinate_space="px")
    uv_polygon = build_keeper_polygon(series, coordinate_space="uv")

    valid = px_polygon is not None
    finite_px = [point for point in px_points.values() if point is not None]
    finite_uv = [point for point in uv_points.values() if point is not None]

    if finite_px:
        xs = [point[0] for point in finite_px]
        ys = [point[1] for point in finite_px]
        width = float(max(xs) - min(xs))
        height = float(max(ys) - min(ys))
        center_x = float(np.mean(xs))
        center_y = float(np.mean(ys))
    else:
        width = height = center_x = center_y = np.nan

    if finite_uv:
        center_u = float(np.mean([point[0] for point in finite_uv]))
        center_v = float(np.mean([point[1] for point in finite_uv]))
    else:
        center_u = center_v = np.nan

    aspect_ratio = width / height if np.isfinite(width) and np.isfinite(height) and height > 0 else np.nan
    polygon_confidence = _polygon_confidence(series, valid)
    return {
        "keeper_pose_valid": bool(valid),
        "keeper_polygon_confidence": polygon_confidence,
        "keeper_polygon_area": float(px_polygon.area) if px_polygon is not None else np.nan,
        "keeper_polygon_area_uv": float(uv_polygon.area) if uv_polygon is not None else np.nan,
        "keeper_body_width": width,
        "keeper_body_height": height,
        "keeper_hand_span": _distance(px_points["left_hand"], px_points["right_hand"]),
        "keeper_foot_span": _distance(px_points["left_foot"], px_points["right_foot"]),
        "keeper_center_x": center_x,
        "keeper_center_y": center_y,
        "keeper_center_u": center_u,
        "keeper_center_v": center_v,
        "keeper_body_aspect_ratio": aspect_ratio,
    }


def add_keeper_polygon_features(pose_df: pd.DataFrame) -> pd.DataFrame:
    """Append keeper polygon features to a pose dataframe."""
    output = pose_df.copy()
    if output.empty:
        for column in POLYGON_FEATURE_COLUMNS:
            output[column] = pd.Series(dtype="float64")
        return output

    features = pd.DataFrame([polygon_feature_row(row) for _, row in output.iterrows()], index=output.index)
    for column in POLYGON_FEATURE_COLUMNS:
        output[column] = features[column]
    return output


def keeper_pose_ml_feature_columns(include_ball_context: bool = True) -> list[str]:
    """Return a recommended normalized feature vector for predictive models."""
    columns: list[str] = []
    for point in POLYGON_ORDER:
        columns.extend([f"{point}_u", f"{point}_v"])
    columns.extend(
        [
            "keeper_polygon_area_uv",
            "keeper_polygon_confidence",
            "keeper_center_u",
            "keeper_center_v",
            "keeper_body_aspect_ratio",
        ]
    )
    if include_ball_context:
        columns.extend(
            [
                "goal_entry_u",
                "goal_entry_v",
                "shot_ball_speed",
                "time_ball_to_goal",
            ]
        )
    return columns
