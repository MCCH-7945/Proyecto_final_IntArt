import numpy as np
import pandas as pd

from shot_detector import detect_shot_frame, estimate_play_end_event
from trajectory import estimate_ball_velocity


def test_detect_shot_frame_from_sudden_speed_increase():
    fps = 10
    frames = list(range(30))
    x = []
    pos = 0.0
    for frame in frames:
        step = 1.0 if frame < 10 else 7.0
        pos += step
        x.append(pos)
    df = pd.DataFrame(
        {
            "frame": frames,
            "time_sec": [frame / fps for frame in frames],
            "ball_x_smooth": x,
            "ball_y_smooth": np.zeros(len(frames)),
        }
    )
    df = estimate_ball_velocity(df, fps=fps)
    result = detect_shot_frame(df, fps=fps, threshold_z=1.5, speed_percentile=80)
    assert result["shot_frame"] is not None
    assert 9 <= result["shot_frame"] <= 11
    assert result["shot_confidence"] > 0


def test_estimate_play_end_event_prefers_goal_crossing():
    fps = 10
    df = pd.DataFrame(
        {
            "frame": list(range(8)),
            "time_sec": [frame / fps for frame in range(8)],
            "ball_x_smooth": np.arange(8, dtype=float),
            "ball_y_smooth": np.zeros(8),
            "ball_speed": [np.nan, 10, 20, 30, 40, 50, 60, 70],
        }
    )
    goal_entry = {
        "goal_entry_frame": 5,
        "goal_entry_confidence": 0.7,
    }
    result = estimate_play_end_event(df, 2, goal_entry, fps=fps)
    assert result["play_end_frame"] == 5
    assert result["play_outcome"] == "goal"
