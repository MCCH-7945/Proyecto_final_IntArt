import numpy as np

from pff_ingestion import (
    build_pff_save_probability_rows,
    event_to_save_probability_row,
    hydrate_rows_with_pff_tracking,
    infer_attacking_goal_sign,
    is_shot_event,
    normalize_field_point_to_goal_uv,
    normalize_goalmouth_point_to_goal_uv,
)


def test_pff_shot_event_with_freeze_frame_maps_to_model_row():
    event = {
        "event_type": "Shot",
        "game_id": 99,
        "event_id": "shot-1",
        "team_id": "ARG",
        "player_name": "Forward",
        "start": {"x": 41.0, "y": 0.0},
        "shot": {"end_location": [52.5, 1.0, 0.8], "outcome": {"name": "Saved"}},
        "freeze_frame": [
            {"team_id": "FRA", "position": {"name": "Goalkeeper"}, "location": [51.0, -0.4]},
            {"team_id": "ARG", "position": {"name": "Forward"}, "location": [41.0, 0.0]},
        ],
        "shot_speed": 27.0,
    }
    row = event_to_save_probability_row(event)
    assert row["save_label"] == 1.0
    assert row["keeper_source"] == "event_freeze_frame"
    assert np.isclose(row["goal_entry_u"], (1.0 + 3.66) / 7.32)
    assert np.isclose(row["goal_entry_v"], 0.8 / 2.44)
    assert np.isclose(row["ball_position_u"], 0.5)
    assert row["time_to_goal_source"] == "distance_over_speed"
    assert row["shot_ball_speed"] == 27.0


def test_pff_negative_goal_side_is_oriented_consistently():
    goal_sign = infer_attacking_goal_sign(
        {},
        start=(-42.0, 1.0, np.nan),
        end=(-52.5, -1.0, 0.6),
    )
    assert goal_sign == -1.0
    target_u, target_v = normalize_goalmouth_point_to_goal_uv((-52.5, -1.0, 0.6), goal_sign)
    keeper_u, keeper_v = normalize_field_point_to_goal_uv((-51.0, 0.4, np.nan), goal_sign)
    assert np.isclose(target_u, (1.0 + 3.66) / 7.32)
    assert np.isclose(target_v, 0.6 / 2.44)
    assert np.isclose(keeper_u, (-0.4 + 3.66) / 7.32)
    assert keeper_v < 0


def test_build_rows_filters_unlabeled_by_default():
    events = [
        {
            "event_type": "Shot",
            "start": [42.0, 0.0],
            "end_location": [52.5, 0.0, 0.2],
            "outcome": "Goal",
        },
        {
            "event_type": "Pass",
            "start": [10.0, 2.0],
        },
        {
            "event_type": "Shot",
            "start": [40.0, 1.0],
            "end_location": [52.5, 7.0, 0.1],
            "outcome": "Off Target",
        },
    ]
    labeled = build_pff_save_probability_rows(events)
    assert len(labeled) == 1
    assert labeled.iloc[0]["save_label"] == 0.0
    all_shots = build_pff_save_probability_rows(events, include_unlabeled=True)
    assert len(all_shots) == 2
    assert is_shot_event(events[0])
    assert not is_shot_event(events[1])


def test_hydrate_rows_with_tracking_replaces_keeper_fallback(tmp_path):
    events = [
        {
            "event_type": "Shot",
            "game_id": "game-1",
            "frame_id": 10,
            "team_id": "ARG",
            "start": [42.0, 0.0],
            "end_location": [52.5, 0.0, 0.2],
            "outcome": "Goal",
        }
    ]
    rows = build_pff_save_probability_rows(events)
    assert rows.iloc[0]["keeper_source"] == "fallback_center"

    tracking_dir = tmp_path / "tracking"
    tracking_dir.mkdir()
    (tracking_dir / "game-1.jsonl").write_text(
        (
            '{"frame_id": 10, "players": ['
            '{"team_id": "FRA", "position": {"name": "Goalkeeper"}, "location": [51.0, -0.4]},'
            '{"team_id": "ARG", "position": {"name": "Forward"}, "location": [42.0, 0.0]}'
            '], "ball": [42.0, 0.0, 0.0]}\n'
        ),
        encoding="utf-8",
    )
    hydrated = hydrate_rows_with_pff_tracking(rows, tracking_dir)
    assert hydrated.iloc[0]["keeper_source"] == "tracking"
    assert hydrated.iloc[0]["tracking_used"]
    assert hydrated.iloc[0]["ball_source"] == "tracking"
    assert np.isclose(hydrated.iloc[0]["keeper_center_u"], (-0.4 + 3.66) / 7.32)
