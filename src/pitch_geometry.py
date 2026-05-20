"""Pitch and penalty-area geometry helpers."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon


def _is_valid_point(point: Iterable[float] | None) -> bool:
    if point is None:
        return False
    values = list(point)
    if len(values) != 2:
        return False
    return bool(np.isfinite(values[0]) and np.isfinite(values[1]))


def _is_finite_number(value) -> bool:
    try:
        return bool(np.isfinite(value))
    except (TypeError, ValueError):
        return False


def is_shot_inside_penalty_area(
    ball_pos: tuple[float, float] | list[float] | None,
    penalty_area_polygon: list[list[float]] | None,
) -> bool | None:
    """Return whether the ball position is inside the penalty-area polygon.

    `None` is returned when either the ball point or polygon is unavailable.
    Boundary points count as inside because a shot exactly on the line should
    not be discarded by geometric precision alone.
    """
    if not penalty_area_polygon or not _is_valid_point(ball_pos):
        return None

    try:
        polygon = Polygon(penalty_area_polygon)
    except Exception:
        return None

    if polygon.is_empty or not polygon.is_valid:
        polygon = polygon.buffer(0)
    if polygon.is_empty:
        return None

    point = Point(float(ball_pos[0]), float(ball_pos[1]))
    return bool(polygon.covers(point))


def add_event_inside_penalty_area_column(
    ball_df: pd.DataFrame,
    penalty_area_polygon: list[list[float]] | None,
    event_frames: list[int | None],
    x_col: str = "ball_x_smooth",
    y_col: str = "ball_y_smooth",
) -> pd.DataFrame:
    """Add nullable `inside_penalty_area`, evaluated only at event frames."""
    output = ball_df.copy()
    output["inside_penalty_area"] = pd.NA
    output.attrs.update(ball_df.attrs)
    if not penalty_area_polygon:
        return output

    if x_col not in output.columns or y_col not in output.columns:
        x_col, y_col = "ball_x", "ball_y"

    clean_event_frames = {int(frame) for frame in event_frames if frame is not None and _is_finite_number(frame)}
    for index, row in output[output["frame"].isin(clean_event_frames)].iterrows():
        point = (row.get(x_col), row.get(y_col))
        output.at[index, "inside_penalty_area"] = is_shot_inside_penalty_area(point, penalty_area_polygon)
    return output


def add_inside_penalty_area_column(
    ball_df: pd.DataFrame,
    penalty_area_polygon: list[list[float]] | None,
    x_col: str = "ball_x_smooth",
    y_col: str = "ball_y_smooth",
) -> pd.DataFrame:
    """Deprecated compatibility helper: evaluate penalty area for every frame."""
    event_frames = ball_df["frame"].dropna().astype(int).tolist() if "frame" in ball_df.columns else []
    return add_event_inside_penalty_area_column(ball_df, penalty_area_polygon, event_frames, x_col=x_col, y_col=y_col)
