"""Ball trajectory smoothing, velocity, and acceleration."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _smooth_series(series: pd.Series, max_gap: int, window: int) -> pd.Series:
    interpolated = series.interpolate(method="linear", limit=max_gap, limit_area="inside")
    valid_mask = interpolated.notna()
    if valid_mask.sum() < 3:
        return interpolated

    window = max(3, int(window))
    if window % 2 == 0:
        window += 1
    smoothed = interpolated.rolling(window=window, center=True, min_periods=1).median()
    smoothed[~valid_mask] = np.nan
    return smoothed


def smooth_ball_trajectory(
    ball_df: pd.DataFrame,
    max_gap: int = 5,
    window: int = 7,
    min_valid_detections: int = 5,
) -> pd.DataFrame:
    """Add smoothed ball coordinates without filling long gaps."""
    output = ball_df.copy()
    valid = output["ball_detected"].fillna(False).astype(bool)
    valid_count = int(valid.sum())

    output["ball_x_smooth"] = output["ball_x"]
    output["ball_y_smooth"] = output["ball_y"]
    output["tracking_quality"] = "ok" if valid_count >= min_valid_detections else "low"
    output.attrs.update(ball_df.attrs)

    if valid_count < min_valid_detections:
        return output

    x = output["ball_x"].where(valid)
    y = output["ball_y"].where(valid)
    output["ball_x_smooth"] = _smooth_series(x, max_gap=max_gap, window=window)
    output["ball_y_smooth"] = _smooth_series(y, max_gap=max_gap, window=window)
    output.attrs.update(ball_df.attrs)
    return output


def estimate_ball_velocity(ball_df: pd.DataFrame, fps: float) -> pd.DataFrame:
    """Add velocity, speed, and acceleration columns in pixels per second."""
    output = ball_df.copy()
    fps = float(fps) if fps and np.isfinite(fps) and fps > 0 else 30.0

    x_col = "ball_x_smooth" if "ball_x_smooth" in output.columns else "ball_x"
    y_col = "ball_y_smooth" if "ball_y_smooth" in output.columns else "ball_y"

    output["ball_vx"] = output[x_col].diff() * fps
    output["ball_vy"] = output[y_col].diff() * fps
    invalid_velocity = output[x_col].isna() | output[y_col].isna() | output[x_col].shift(1).isna() | output[y_col].shift(1).isna()
    output.loc[invalid_velocity, ["ball_vx", "ball_vy"]] = np.nan

    output["ball_speed"] = np.sqrt(output["ball_vx"] ** 2 + output["ball_vy"] ** 2)
    output["ball_acceleration"] = output["ball_speed"].diff() * fps
    invalid_acceleration = output["ball_speed"].isna() | output["ball_speed"].shift(1).isna()
    output.loc[invalid_acceleration, "ball_acceleration"] = np.nan
    output.attrs.update(ball_df.attrs)
    return output
