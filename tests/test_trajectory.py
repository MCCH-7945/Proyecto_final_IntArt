import numpy as np
import pandas as pd

from trajectory import estimate_ball_velocity, smooth_ball_trajectory


def test_smooth_ball_trajectory_fills_short_gap_but_not_long_gap():
    df = pd.DataFrame(
        {
            "ball_x": [0.0, 1.0, np.nan, 3.0, np.nan, np.nan, np.nan, np.nan, np.nan, 9.0],
            "ball_y": [0.0, 1.0, np.nan, 3.0, np.nan, np.nan, np.nan, np.nan, np.nan, 9.0],
            "ball_detected": [True, True, False, True, False, False, False, False, False, True],
        }
    )
    out = smooth_ball_trajectory(df, max_gap=2, window=3, min_valid_detections=3)
    assert np.isfinite(out.loc[2, "ball_x_smooth"])
    assert np.isnan(out.loc[6, "ball_x_smooth"])


def test_estimate_ball_velocity_uses_pixels_per_second():
    df = pd.DataFrame(
        {
            "ball_x_smooth": [0.0, 2.0, 4.0],
            "ball_y_smooth": [0.0, 0.0, 0.0],
        }
    )
    out = estimate_ball_velocity(df, fps=10)
    assert np.isclose(out.loc[1, "ball_vx"], 20.0)
    assert np.isclose(out.loc[1, "ball_speed"], 20.0)

