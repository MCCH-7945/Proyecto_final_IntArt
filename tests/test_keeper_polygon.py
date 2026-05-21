import numpy as np
import pandas as pd

from keeper_polygon import add_keeper_polygon_features, build_keeper_polygon


def test_build_keeper_polygon_from_five_points():
    row = {
        "left_hand_x": 0.0,
        "left_hand_y": 1.0,
        "head_x": 1.0,
        "head_y": 0.0,
        "right_hand_x": 2.0,
        "right_hand_y": 1.0,
        "right_foot_x": 2.0,
        "right_foot_y": 3.0,
        "left_foot_x": 0.0,
        "left_foot_y": 3.0,
    }
    polygon = build_keeper_polygon(row)
    assert polygon is not None
    assert polygon.area > 0


def test_add_keeper_polygon_features():
    df = pd.DataFrame(
        [
            {
                "left_hand_x": 0.0,
                "left_hand_y": 1.0,
                "head_x": 1.0,
                "head_y": 0.0,
                "right_hand_x": 2.0,
                "right_hand_y": 1.0,
                "right_foot_x": 2.0,
                "right_foot_y": 3.0,
                "left_foot_x": 0.0,
                "left_foot_y": 3.0,
                "left_hand_u": 0.1,
                "left_hand_v": 0.5,
                "head_u": 0.5,
                "head_v": 0.9,
                "right_hand_u": 0.9,
                "right_hand_v": 0.5,
                "right_foot_u": 0.8,
                "right_foot_v": 0.0,
                "left_foot_u": 0.2,
                "left_foot_v": 0.0,
                "left_hand_confidence": 0.9,
                "head_confidence": 0.9,
                "right_hand_confidence": 0.9,
                "right_foot_confidence": 0.9,
                "left_foot_confidence": 0.9,
            }
        ]
    )
    out = add_keeper_polygon_features(df)
    assert bool(out.loc[0, "keeper_pose_valid"]) is True
    assert np.isfinite(out.loc[0, "keeper_polygon_area"])
    assert np.isfinite(out.loc[0, "keeper_polygon_area_uv"])
    assert out.loc[0, "keeper_polygon_confidence"] > 0.8

