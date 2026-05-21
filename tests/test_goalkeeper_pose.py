import pandas as pd

from goalkeeper_pose import BODY_POINTS, add_keeper_pose_quality, build_pose_window_index, keeper_pose_columns


def test_keeper_pose_columns_include_five_points():
    columns = keeper_pose_columns()
    for point in BODY_POINTS:
        assert f"{point}_x" in columns
        assert f"{point}_y" in columns
        assert f"{point}_u" in columns
        assert f"{point}_v" in columns
        assert f"{point}_confidence" in columns
    assert "keeper_pose_confidence" in columns
    assert "pose_valid_points_count" in columns


def test_build_pose_window_index_uses_shot_and_play_end():
    shot_row = pd.Series(
        {
            "video_id": "clip_001",
            "shot_frame": 100,
            "play_end_frame": 112,
        }
    )
    df = build_pose_window_index(shot_row, fps=25, pre_frames=5, post_frames=5)
    assert int(df["frame"].min()) == 95
    assert int(df["frame"].max()) == 112
    assert bool(df.loc[df["frame"] == 100, "is_shot_frame"].iloc[0]) is True


def test_add_keeper_pose_quality_penalizes_missing_points():
    df = pd.DataFrame(
        [
            {
                "keeper_confidence": 0.9,
                "head_confidence": 0.9,
                "left_hand_confidence": 0.8,
                "right_hand_confidence": 0.7,
                "left_foot_confidence": 0.0,
                "right_foot_confidence": 0.0,
            }
        ]
    )
    out = add_keeper_pose_quality(df, min_point_confidence=0.25)
    assert int(out.loc[0, "pose_valid_points_count"]) == 3
    assert int(out.loc[0, "pose_missing_points_count"]) == 2
    assert 0 < out.loc[0, "keeper_pose_confidence"] < 0.9
