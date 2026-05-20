"""Goal-frame homography, goal entry estimation, and zone classification."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiPoint, Point, Polygon
from shapely.ops import nearest_points


GOAL_WIDTH_M = 7.32
GOAL_HEIGHT_M = 2.44
GOAL_CORNER_ORDER = ["bottom_left", "bottom_right", "top_right", "top_left"]


def _corner_array(goal_corners: dict[str, Any]) -> np.ndarray:
    missing = [name for name in GOAL_CORNER_ORDER if name not in goal_corners]
    if missing:
        raise ValueError(f"Missing goal corner(s): {', '.join(missing)}")
    return np.array([goal_corners[name] for name in GOAL_CORNER_ORDER], dtype=np.float32)


def compute_goal_homography(goal_corners: dict[str, Any]) -> np.ndarray:
    """Compute image-pixel to normalized-goal homography."""
    src = _corner_array(goal_corners)
    dst = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(src, dst)
    if homography is None or not np.isfinite(homography).all():
        raise ValueError("Could not compute goal homography.")
    return homography


def map_point_to_goal_uv(
    point_px: tuple[float, float] | list[float] | None,
    homography: np.ndarray,
) -> tuple[float | None, float | None]:
    """Map an image point to normalized goal coordinates `(u, v)`."""
    if point_px is None:
        return None, None
    x, y = point_px
    if not np.isfinite(x) or not np.isfinite(y):
        return None, None

    point = np.array([float(x), float(y), 1.0], dtype=float)
    mapped = homography @ point
    if not np.isfinite(mapped).all() or abs(mapped[2]) < 1e-12:
        return None, None
    return float(mapped[0] / mapped[2]), float(mapped[1] / mapped[2])


def classify_goal_zone(
    goal_entry_u: float | None,
    goal_entry_v: float | None,
) -> dict[str, str]:
    """Derive discrete goal zones from continuous normalized coordinates."""
    if goal_entry_u is None or not np.isfinite(goal_entry_u):
        horizontal = "unknown"
    elif goal_entry_u < 1.0 / 3.0:
        horizontal = "left"
    elif goal_entry_u < 2.0 / 3.0:
        horizontal = "center"
    else:
        horizontal = "right"

    if goal_entry_v is None or not np.isfinite(goal_entry_v):
        vertical = "unknown"
    elif goal_entry_v < 1.0 / 3.0:
        vertical = "low"
    elif goal_entry_v < 2.0 / 3.0:
        vertical = "middle"
    else:
        vertical = "high"

    return {
        "goal_zone_horizontal": horizontal,
        "goal_zone_vertical": vertical,
    }


def goal_polygon_from_corners(goal_corners: dict[str, Any]) -> Polygon:
    """Create a shapely polygon for the visible goal frame."""
    corners = _corner_array(goal_corners)
    return Polygon([(float(x), float(y)) for x, y in corners])


def _extract_first_point(geometry, reference: Point | None = None) -> Point | None:
    """Return a representative point from a shapely intersection geometry."""
    if geometry.is_empty:
        return None
    if isinstance(geometry, Point):
        return geometry
    if isinstance(geometry, MultiPoint):
        points = list(geometry.geoms)
    elif hasattr(geometry, "geoms"):
        points = []
        for geom in geometry.geoms:
            point = _extract_first_point(geom, reference)
            if point is not None:
                points.append(point)
    elif isinstance(geometry, LineString):
        coords = list(geometry.coords)
        points = [Point(coords[0]), Point(coords[-1])] if coords else []
    else:
        return None

    if not points:
        return None
    if reference is None:
        return points[0]
    return min(points, key=lambda point: point.distance(reference))


def _position_columns(ball_df: pd.DataFrame) -> tuple[str, str]:
    if "ball_x_smooth" in ball_df.columns and "ball_y_smooth" in ball_df.columns:
        return "ball_x_smooth", "ball_y_smooth"
    return "ball_x", "ball_y"


def _goal_size_px(goal_corners: dict[str, Any]) -> float:
    corners = _corner_array(goal_corners)
    width = np.linalg.norm(corners[1] - corners[0])
    height = np.linalg.norm(corners[3] - corners[0])
    return float(max(width, height))


def _uv_confidence_factor(u: float | None, v: float | None) -> float:
    if u is None or v is None:
        return 0.0
    max_outside = max(0.0, -u, u - 1.0, -v, v - 1.0)
    if max_outside <= 0.0:
        return 1.0
    if max_outside <= 0.10:
        return 0.65
    if max_outside <= 0.25:
        return 0.35
    return 0.10


def estimate_goal_entry_point(
    ball_df: pd.DataFrame,
    shot_frame: int | None,
    goal_corners: dict[str, Any] | None,
) -> dict[str, float | str | None]:
    """Estimate where the post-shot ball trajectory crosses the goal frame.

    The function prefers direct ball detections inside the goal frame, then
    line-segment intersections with the frame boundary, then a low-confidence
    nearest projection only when the trajectory passes very close to the goal.
    """
    if shot_frame is None or not goal_corners:
        return {
            "goal_entry_x_px": None,
            "goal_entry_y_px": None,
            "goal_entry_frame": None,
            "goal_entry_confidence": 0.0,
            "method": "missing_goal_or_shot",
        }

    try:
        goal_polygon = goal_polygon_from_corners(goal_corners)
        homography = compute_goal_homography(goal_corners)
    except Exception:
        return {
            "goal_entry_x_px": None,
            "goal_entry_y_px": None,
            "goal_entry_frame": None,
            "goal_entry_confidence": 0.0,
            "method": "invalid_goal_corners",
        }

    if goal_polygon.is_empty or not goal_polygon.is_valid:
        goal_polygon = goal_polygon.buffer(0)
    if goal_polygon.is_empty:
        return {
            "goal_entry_x_px": None,
            "goal_entry_y_px": None,
            "goal_entry_frame": None,
            "goal_entry_confidence": 0.0,
            "method": "invalid_goal_polygon",
        }

    x_col, y_col = _position_columns(ball_df)
    post_df = ball_df[ball_df["frame"] >= int(shot_frame)].copy()
    post_df = post_df[np.isfinite(post_df[x_col]) & np.isfinite(post_df[y_col])]
    if post_df.empty:
        return {
            "goal_entry_x_px": None,
            "goal_entry_y_px": None,
            "goal_entry_frame": None,
            "goal_entry_confidence": 0.0,
            "method": "no_post_shot_positions",
        }

    # Direct observation inside the visible goal frame.
    for _, row in post_df.iterrows():
        point = Point(float(row[x_col]), float(row[y_col]))
        if goal_polygon.covers(point):
            u, v = map_point_to_goal_uv((point.x, point.y), homography)
            confidence = 0.85 * _uv_confidence_factor(u, v)
            return {
                "goal_entry_x_px": float(point.x),
                "goal_entry_y_px": float(point.y),
                "goal_entry_frame": int(row["frame"]),
                "goal_entry_confidence": float(confidence),
                "method": "direct_ball_inside_goal_frame",
            }

    # Segment intersection between consecutive post-shot positions.
    rows = list(post_df.itertuples(index=False))
    for prev, curr in zip(rows[:-1], rows[1:]):
        prev_frame = getattr(prev, "frame")
        curr_frame = getattr(curr, "frame")
        if int(curr_frame) - int(prev_frame) > 8:
            continue
        p0 = Point(float(getattr(prev, x_col)), float(getattr(prev, y_col)))
        p1 = Point(float(getattr(curr, x_col)), float(getattr(curr, y_col)))
        segment = LineString([p0, p1])
        if not segment.intersects(goal_polygon):
            continue
        intersection = segment.intersection(goal_polygon.boundary)
        point = _extract_first_point(intersection, reference=p0)
        if point is None and goal_polygon.intersects(segment):
            point = _extract_first_point(segment.intersection(goal_polygon), reference=p0)
        if point is None:
            continue
        u, v = map_point_to_goal_uv((point.x, point.y), homography)
        confidence = 0.75 * _uv_confidence_factor(u, v)
        return {
                "goal_entry_x_px": float(point.x),
                "goal_entry_y_px": float(point.y),
                "goal_entry_frame": int(curr_frame),
                "goal_entry_confidence": float(confidence),
                "method": "trajectory_goal_frame_intersection",
            }

    # Low-confidence fallback if the ball path passes near the goal frame.
    near_threshold = max(8.0, _goal_size_px(goal_corners) * 0.06)
    best_point = None
    best_distance = float("inf")
    for prev, curr in zip(rows[:-1], rows[1:]):
        p0 = Point(float(getattr(prev, x_col)), float(getattr(prev, y_col)))
        p1 = Point(float(getattr(curr, x_col)), float(getattr(curr, y_col)))
        segment = LineString([p0, p1])
        distance = segment.distance(goal_polygon)
        if distance < best_distance:
            best_distance = float(distance)
            _, best_point = nearest_points(segment, goal_polygon)

    if best_point is not None and best_distance <= near_threshold:
        u, v = map_point_to_goal_uv((best_point.x, best_point.y), homography)
        distance_factor = max(0.0, 1.0 - best_distance / near_threshold)
        confidence = 0.35 * distance_factor * _uv_confidence_factor(u, v)
        return {
            "goal_entry_x_px": float(best_point.x),
            "goal_entry_y_px": float(best_point.y),
            "goal_entry_frame": None,
            "goal_entry_confidence": float(confidence),
            "method": "nearest_projection_to_goal_frame",
        }

    return {
        "goal_entry_x_px": None,
        "goal_entry_y_px": None,
        "goal_entry_frame": None,
        "goal_entry_confidence": 0.0,
        "method": "no_clear_goal_crossing",
    }
