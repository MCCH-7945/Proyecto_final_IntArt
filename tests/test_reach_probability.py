import numpy as np
import pandas as pd

from reach_probability import (
    SAVE_PROBABILITY_FEATURES,
    add_derived_save_features,
    generate_synthetic_save_probability_data,
    goal_area_bounds,
    normalize_outcome_to_save_label,
    pitch_area_bounds,
    prepare_save_probability_dataset,
)


def test_normalize_outcome_to_save_label():
    assert normalize_outcome_to_save_label("save") == 1.0
    assert normalize_outcome_to_save_label("stopped_or_save") == 1.0
    assert normalize_outcome_to_save_label("goal") == 0.0
    assert np.isnan(normalize_outcome_to_save_label("out_or_miss"))


def test_add_derived_save_features_computes_distance():
    df = pd.DataFrame(
        [
            {
                "keeper_center_u": 0.5,
                "keeper_center_v": 0.2,
                "goal_entry_u": 1.0,
                "goal_entry_v": 0.2,
            }
        ]
    )
    out = add_derived_save_features(df)
    assert np.isclose(out.loc[0, "lateral_delta_m"], 3.66)
    assert np.isclose(out.loc[0, "vertical_delta_m"], 0.0)
    assert np.isclose(out.loc[0, "distance_keeper_to_target_m"], 3.66)


def test_prepare_save_probability_dataset_from_synthetic_data():
    df = generate_synthetic_save_probability_data(n=80, random_state=3)
    prepared = prepare_save_probability_dataset(df)
    assert len(prepared.labels) == 80
    assert prepared.features.columns.tolist() == SAVE_PROBABILITY_FEATURES
    assert set(prepared.labels.unique()).issubset({0, 1})


def test_synthetic_actors_stay_inside_penalty_and_goal_areas():
    df = generate_synthetic_save_probability_data(n=800, random_state=5)
    out = add_derived_save_features(df)
    pen = pitch_area_bounds()
    box = goal_area_bounds()
    for col_u, col_v in [("keeper_center_u", "keeper_center_v"), ("ball_position_u", "ball_position_v")]:
        assert out[col_u].between(pen["u_min"], pen["u_max"]).all()
        assert out[col_v].between(pen["v_min"], pen["v_max"]).all()
    in_small = (
        out["ball_position_u"].between(box["u_min"], box["u_max"])
        & out["ball_position_v"].between(box["v_min"], box["v_max"])
    )
    assert int(in_small.sum()) > 120


def test_synthetic_data_matches_apeste_ph_stylized_facts():
    df = generate_synthetic_save_probability_data(n=5000, random_state=9)
    shootout = df.loc[df["is_shootout_context"] >= 0.5]
    open_play = df.loc[df["is_shootout_context"] < 0.5]
    assert len(shootout) > 2000
    goal_rate = 1.0 - shootout["save_label"].mean()
    assert 0.66 <= goal_rate <= 0.82
    assert 0.74 <= open_play["empirical_goal_probability"].mean() <= 0.86
    assert shootout["time_ball_to_goal"].between(0.28, 0.42).mean() > 0.85
    first = shootout.loc[shootout["team_kicks_first"] >= 0.5]
    second_lag = shootout.loc[(shootout["team_kicks_first"] < 0.5) & (shootout["partial_score"] < 0)]
    assert (1.0 - first["save_label"].mean()) > (1.0 - second_lag["save_label"].mean()) - 0.05
    prepared = prepare_save_probability_dataset(df)
    assert len(prepared.features.columns) == 26
