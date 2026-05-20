import pandas as pd

from pitch_geometry import add_event_inside_penalty_area_column, is_shot_inside_penalty_area


def test_is_shot_inside_penalty_area_includes_boundary():
    polygon = [[0, 0], [10, 0], [10, 10], [0, 10]]
    assert is_shot_inside_penalty_area((5, 5), polygon) is True
    assert is_shot_inside_penalty_area((0, 5), polygon) is True
    assert is_shot_inside_penalty_area((12, 5), polygon) is False


def test_is_shot_inside_penalty_area_missing_returns_none():
    assert is_shot_inside_penalty_area((5, 5), None) is None
    assert is_shot_inside_penalty_area(None, [[0, 0], [1, 0], [1, 1]]) is None


def test_add_event_inside_penalty_area_only_evaluates_event_frames():
    df = pd.DataFrame(
        {
            "frame": [0, 1, 2],
            "ball_x_smooth": [5.0, 12.0, 6.0],
            "ball_y_smooth": [5.0, 5.0, 5.0],
        }
    )
    polygon = [[0, 0], [10, 0], [10, 10], [0, 10]]
    out = add_event_inside_penalty_area_column(df, polygon, [1])
    assert pd.isna(out.loc[0, "inside_penalty_area"])
    assert out.loc[1, "inside_penalty_area"] is False
    assert pd.isna(out.loc[2, "inside_penalty_area"])
