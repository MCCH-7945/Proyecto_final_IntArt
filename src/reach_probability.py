"""Supervised save-probability helpers.

This module is intentionally tabular. It can train from a future SoccerNet- or
annotation-derived table with columns such as keeper position, shot target, ball
speed, time-to-goal, and outcome. For demos, it can also generate synthetic
examples that follow a geometric proxy; those synthetic examples are useful for
showing the pipeline, but they are not evidence learned from SoccerNet.

Synthetic shoot-out scenarios are loosely calibrated to stylized facts from
Apesteguia and Palacios-Huerta, "Psychological Pressure in Competitive
Environments: Evidence from a Randomized Natural Experiment," American Economic
Review, 2010 (often discussed alongside field evidence in behavioral economics).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

GOAL_WIDTH_M = 7.32
GOAL_HEIGHT_M = 2.44
PENALTY_WIDTH_M = 40.32
PENALTY_DEPTH_M = 16.5
GOAL_AREA_WIDTH_M = 18.32
GOAL_AREA_DEPTH_M = 5.5

PENALTY_HALF_WIDTH_U = (PENALTY_WIDTH_M / GOAL_WIDTH_M) / 2.0
PENALTY_DEPTH_V = PENALTY_DEPTH_M / GOAL_HEIGHT_M
GOAL_AREA_HALF_WIDTH_U = (GOAL_AREA_WIDTH_M / GOAL_WIDTH_M) / 2.0
GOAL_AREA_DEPTH_V = GOAL_AREA_DEPTH_M / GOAL_HEIGHT_M
PENALTY_MARK_DISTANCE_M = 11.0

# Stylized facts from Apesteguia & Palacios-Huerta (AER 2010), penalty shoot-outs.
APESTE_PH_2010 = {
    "ball_flight_time_sec": (0.30, 0.40),
    "shootout_scoring_rate_all": 0.731,
    "shootout_scoring_rate_first_team": 0.763,
    "shootout_scoring_rate_second_team": 0.697,
    "in_game_penalty_scoring_rate": 0.80,
    "shootout_first_mover_win_rate": 0.605,
    "second_lagging_scoring_early_rounds": 0.77,
    "second_lagging_scoring_late_rounds": 0.64,
    "first_team_scoring_band": (0.72, 0.78),
}

SAVE_LABEL_COLUMN = "save_label"
SAVE_PROBABILITY_FEATURES = [
    "keeper_center_u",
    "keeper_center_v",
    "keeper_body_width",
    "keeper_body_height",
    "keeper_hand_span",
    "keeper_foot_span",
    "keeper_polygon_area_uv",
    "keeper_pose_confidence",
    "ball_position_u",
    "ball_position_v",
    "goal_entry_u",
    "goal_entry_v",
    "shot_ball_speed",
    "time_ball_to_goal",
    "reaction_time",
    "lateral_delta_m",
    "vertical_delta_m",
    "distance_keeper_to_target_m",
    "distance_ball_to_target_m",
    "distance_keeper_to_ball_m",
    "keeper_outside_goal",
    "team_kicks_first",
    "shootout_round",
    "partial_score",
    "kick_importance",
    "is_shootout_context",
]

POSITIVE_OUTCOMES = {
    "save",
    "saved",
    "stopped",
    "stopped_or_save",
    "keeper_save",
    "blocked_by_keeper",
}
NEGATIVE_OUTCOMES = {
    "goal",
    "scored",
    "conceded",
}


def pitch_area_bounds() -> dict[str, float]:
    """Penalty area in goal-normalized coordinates (v=0 is the goal line)."""
    return {
        "u_min": 0.5 - PENALTY_HALF_WIDTH_U,
        "u_max": 0.5 + PENALTY_HALF_WIDTH_U,
        "v_min": -PENALTY_DEPTH_V,
        "v_max": 0.0,
    }


def goal_area_bounds() -> dict[str, float]:
    """Six-yard box inside the penalty area."""
    return {
        "u_min": 0.5 - GOAL_AREA_HALF_WIDTH_U,
        "u_max": 0.5 + GOAL_AREA_HALF_WIDTH_U,
        "v_min": -GOAL_AREA_DEPTH_V,
        "v_max": 0.0,
    }


def clip_pitch_uv(u: float | np.ndarray, v: float | np.ndarray) -> tuple[Any, Any]:
    """Clamp positions to the penalty area (includes the small box)."""
    bounds = pitch_area_bounds()
    u = np.clip(u, bounds["u_min"], bounds["u_max"])
    v = np.clip(v, bounds["v_min"], bounds["v_max"])
    return u, v


def _sample_uniform_in_bounds(rng: np.random.Generator, n: int, bounds: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    u = rng.uniform(bounds["u_min"], bounds["u_max"], n)
    v = rng.uniform(bounds["v_min"], bounds["v_max"], n)
    return u, v


def _sample_pitch_actor_uv(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample keeper/ball positions inside the small box or the wider penalty area."""
    goal_bounds = goal_area_bounds()
    penalty_bounds = pitch_area_bounds()
    in_small_box = rng.random(n) < 0.42
    u = np.empty(n, dtype=float)
    v = np.empty(n, dtype=float)
    if int(in_small_box.sum()):
        u[in_small_box], v[in_small_box] = _sample_uniform_in_bounds(rng, int(in_small_box.sum()), goal_bounds)
    if int((~in_small_box).sum()):
        u[~in_small_box], v[~in_small_box] = _sample_uniform_in_bounds(rng, int((~in_small_box).sum()), penalty_bounds)
    return clip_pitch_uv(u, v)


@dataclass(frozen=True)
class SaveProbabilityData:
    """Prepared supervised dataset."""

    features: pd.DataFrame
    labels: pd.Series
    frame: pd.DataFrame


def _safe_float_series(df: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce").astype(float)


def normalize_outcome_to_save_label(value: Any) -> float:
    """Map an outcome string/value to 1 for save, 0 for goal, NaN otherwise."""
    if value is None:
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value):
        if float(value) in {0.0, 1.0}:
            return float(value)
    clean = str(value).strip().lower()
    if clean in POSITIVE_OUTCOMES:
        return 1.0
    if clean in NEGATIVE_OUTCOMES:
        return 0.0
    return np.nan


def add_derived_save_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add geometric features needed by the save-probability model."""
    output = df.copy()
    keeper_u = _safe_float_series(output, "keeper_center_u", 0.5)
    keeper_v = _safe_float_series(output, "keeper_center_v", 0.2)
    target_u = _safe_float_series(output, "goal_entry_u")
    target_v = _safe_float_series(output, "goal_entry_v")
    if "ball_position_u" in output.columns:
        ball_u = _safe_float_series(output, "ball_position_u")
    else:
        ball_u = target_u
    if "ball_position_v" in output.columns:
        ball_v = _safe_float_series(output, "ball_position_v")
    else:
        ball_v = target_v

    output["ball_position_u"] = ball_u
    output["ball_position_v"] = ball_v
    output["lateral_delta_m"] = (target_u - keeper_u) * GOAL_WIDTH_M
    output["vertical_delta_m"] = (target_v - keeper_v) * GOAL_HEIGHT_M
    output["distance_keeper_to_target_m"] = np.hypot(output["lateral_delta_m"], output["vertical_delta_m"])
    output["distance_ball_to_target_m"] = np.hypot(
        (target_u - ball_u) * GOAL_WIDTH_M,
        (target_v - ball_v) * GOAL_HEIGHT_M,
    )
    output["distance_keeper_to_ball_m"] = np.hypot(
        (ball_u - keeper_u) * GOAL_WIDTH_M,
        (ball_v - keeper_v) * GOAL_HEIGHT_M,
    )
    lateral_outside = np.maximum(0.0, np.abs(keeper_u - 0.5) - 0.5)
    vertical_outside = np.maximum(0.0, np.maximum(keeper_v - 1.0, -keeper_v))
    output["keeper_outside_goal"] = np.clip(lateral_outside * 2.0 + vertical_outside, 0.0, 1.5)

    if "reaction_time" not in output.columns:
        output["reaction_time"] = 0.32
    if "keeper_pose_confidence" not in output.columns and "keeper_polygon_confidence" in output.columns:
        output["keeper_pose_confidence"] = output["keeper_polygon_confidence"]
    if "keeper_pose_confidence" not in output.columns:
        output["keeper_pose_confidence"] = 0.65
    if "team_kicks_first" not in output.columns:
        output["team_kicks_first"] = 0.5
    if "shootout_round" not in output.columns:
        output["shootout_round"] = 0.0
    if "partial_score" not in output.columns:
        output["partial_score"] = 0.0
    if "kick_importance" not in output.columns:
        output["kick_importance"] = 0.35
    if "is_shootout_context" not in output.columns:
        output["is_shootout_context"] = 0.0
    lagging = output["partial_score"] < -0.5
    late = output["shootout_round"] >= 3
    output["kicker_under_pressure"] = (
        lagging.astype(float) * 0.55
        + (late & lagging).astype(float) * 0.25
        + (1 - output["team_kicks_first"]) * lagging.astype(float) * 0.15
    ).clip(0.0, 1.0)
    return output


def prepare_save_probability_dataset(
    df: pd.DataFrame,
    outcome_col: str = "outcome",
    feature_columns: list[str] | None = None,
) -> SaveProbabilityData:
    """Prepare feature/label matrices from a shot-level training table."""
    if df.empty:
        raise ValueError("Training dataframe is empty.")
    output = add_derived_save_features(df)
    feature_columns = feature_columns or SAVE_PROBABILITY_FEATURES

    if SAVE_LABEL_COLUMN in output.columns:
        labels = pd.to_numeric(output[SAVE_LABEL_COLUMN], errors="coerce")
    elif outcome_col in output.columns:
        labels = output[outcome_col].map(normalize_outcome_to_save_label)
    elif "play_outcome" in output.columns:
        labels = output["play_outcome"].map(normalize_outcome_to_save_label)
    else:
        raise ValueError("Training data needs `save_label`, `outcome`, or `play_outcome`.")

    features = output.reindex(columns=feature_columns)
    valid = labels.notna()
    if int(valid.sum()) < 20:
        raise ValueError("Need at least 20 labeled save/goal examples to train.")
    features = features.loc[valid].astype(float)
    labels = labels.loc[valid].astype(int)
    frame = output.loc[valid].copy()
    return SaveProbabilityData(features=features, labels=labels, frame=frame)


def _sample_shootout_context(rng: np.random.Generator, n: int) -> pd.DataFrame:
    """Sample penalty-shoot-out states consistent with Apesteguia & Palacios-Huerta (2010)."""
    team_kicks_first = rng.binomial(1, 0.5, n).astype(float)
    shootout_round = rng.integers(1, 6, n).astype(float)
    partial_score = np.zeros(n, dtype=float)
    lagging_second = rng.random(n) < 0.62
    partial_score[(team_kicks_first < 0.5) & lagging_second] = -1.0
    tied_first = rng.random(n) < 0.55
    partial_score[(team_kicks_first > 0.5) & tied_first] = 0.0
    leading = rng.random(n) < 0.18
    partial_score[leading & (team_kicks_first > 0.5)] = 1.0
    partial_score[leading & (team_kicks_first < 0.5)] = -1.0

    kick_importance = np.clip(0.22 + 0.11 * (shootout_round - 1) + 0.18 * (partial_score < 0), 0.15, 0.95)
    kick_importance += np.where((team_kicks_first < 0.5) & (partial_score < 0), 0.12, 0.0)
    kick_importance = np.clip(kick_importance, 0.15, 0.98)

    ball_u = 0.5 + rng.normal(0.0, 0.04, n)
    ball_v = np.zeros(n, dtype=float)
    keeper_u = 0.5 + rng.normal(0.0, 0.22, n) + np.where(partial_score < 0, -0.08, 0.05)
    keeper_v = rng.uniform(-1.05, -0.18, n)
    flight = rng.uniform(*APESTE_PH_2010["ball_flight_time_sec"], n)
    speed_ms = PENALTY_MARK_DISTANCE_M / np.maximum(flight, 0.22)
    shot_speed = np.clip(speed_ms * 3.6, 58, 118)

    return pd.DataFrame(
        {
            "team_kicks_first": team_kicks_first,
            "shootout_round": shootout_round,
            "partial_score": partial_score,
            "kick_importance": kick_importance,
            "is_shootout_context": np.ones(n, dtype=float),
            "ball_position_u": ball_u,
            "ball_position_v": ball_v,
            "keeper_center_u": keeper_u,
            "keeper_center_v": keeper_v,
            "time_ball_to_goal": flight,
            "shot_ball_speed": shot_speed,
            "reaction_time": np.clip(rng.normal(0.30, 0.04, n), 0.22, 0.40),
        }
    )


def _apeste_ph_goal_probability(row: pd.Series) -> float:
    """Stylized scoring probability P(goal) from Apesteguia & Palacios-Huerta (2010)."""
    if float(row.get("is_shootout_context", 0.0)) < 0.5:
        return float(APESTE_PH_2010["in_game_penalty_scoring_rate"])

    first = float(row["team_kicks_first"]) >= 0.5
    lagging = float(row["partial_score"]) < -0.5
    rnd = float(row.get("shootout_round", 1.0))

    if first and not lagging:
        base = APESTE_PH_2010["shootout_scoring_rate_first_team"]
    elif (not first) and lagging:
        if rnd >= 3:
            base = APESTE_PH_2010["second_lagging_scoring_late_rounds"]
        else:
            base = APESTE_PH_2010["second_lagging_scoring_early_rounds"]
    elif not first:
        base = APESTE_PH_2010["shootout_scoring_rate_second_team"]
    else:
        base = APESTE_PH_2010["shootout_scoring_rate_all"]

    importance = float(row.get("kick_importance", 0.35))
    if lagging and not first:
        base -= 0.04 * importance
    if first and float(row["partial_score"]) >= 0.5:
        low, high = APESTE_PH_2010["first_team_scoring_band"]
        base = 0.5 * (low + high)
    return float(np.clip(base, 0.52, 0.84))


def _synthetic_probability(row: pd.Series) -> float:
    """Generate a plausible save probability for synthetic demo data."""
    keeper_x = row["keeper_center_u"] * GOAL_WIDTH_M
    keeper_z = row["keeper_center_v"] * GOAL_HEIGHT_M
    target_x = row["goal_entry_u"] * GOAL_WIDTH_M
    target_z = row["goal_entry_v"] * GOAL_HEIGHT_M
    ball_x = row["ball_position_u"] * GOAL_WIDTH_M
    ball_z = row["ball_position_v"] * GOAL_HEIGHT_M
    dx = target_x - keeper_x
    # When the keeper stands inside the area in front of the goal, vertical reach is
    # measured from the goal line rather than from their depth on the grass.
    reach_keeper_z = max(keeper_z, 0.0)
    dz = target_z - reach_keeper_z
    reach_x = max(0.25, row["keeper_hand_span"] * 0.42 + 0.38)
    reach_z = max(0.25, row["keeper_body_height"] * 0.58 + 0.22)
    normalized_distance = np.sqrt((dx / reach_x) ** 2 + (dz / reach_z) ** 2)
    movement_time = 0.17 + 0.22 * normalized_distance + 0.08 * abs(dx) / GOAL_WIDTH_M
    time_ratio = row["time_ball_to_goal"] / max(0.08, row["reaction_time"] + movement_time)
    speed_penalty = np.clip((row["shot_ball_speed"] - 72.0) / 58.0, 0.0, 1.0)
    pose_bonus = 0.75 * (row["keeper_pose_confidence"] - 0.55)
    ball_travel = np.hypot(target_x - ball_x, target_z - ball_z)
    ball_setup_penalty = np.clip(ball_travel / 9.5, 0.0, 1.0) * 0.42
    outside_posts = np.clip(float(row["keeper_outside_goal"]), 0.0, 1.5) * 0.38
    keeper_ball_dist = float(row.get("distance_keeper_to_ball_m", np.hypot(ball_x - keeper_x, ball_z - keeper_z)))
    in_pitch = ball_z <= 0.0 and keeper_z <= 0.0
    area_bonus = 0.0
    if in_pitch:
        area_bonus = 0.28 * np.clip(1.0 - keeper_ball_dist / 7.5, 0.0, 1.0)
        area_bonus += 0.12 * np.clip(-ball_z / GOAL_HEIGHT_M / 4.0, 0.0, 1.0)
    pressure_penalty = 0.0
    if float(row.get("is_shootout_context", 0.0)) >= 0.5:
        pressure_penalty = 0.55 * float(row.get("kicker_under_pressure", 0.0))
        pressure_penalty += 0.08 * float(row.get("kick_importance", 0.0)) * (1.0 - float(row["team_kicks_first"]))

    logit = (
        3.75 * (1.05 - normalized_distance)
        + 1.9 * (time_ratio - 1.0)
        - 0.95 * speed_penalty
        + pose_bonus
        - ball_setup_penalty
        - outside_posts
        + area_bonus
        - pressure_penalty
    )
    return float(1.0 / (1.0 + np.exp(-logit)))


def _synthetic_save_probability(row: pd.Series) -> float:
    """Blend geometry with Apesteguia-Palacios-Huerta stylized scoring rates."""
    p_save_geom = _synthetic_probability(row)
    p_goal_empirical = _apeste_ph_goal_probability(row)
    p_goal_blend = 0.42 * p_goal_empirical + 0.58 * (1.0 - p_save_geom)
    return float(np.clip(1.0 - p_goal_blend, 0.06, 0.94))


def generate_synthetic_save_probability_data(n: int = 1600, random_state: int = 7) -> pd.DataFrame:
    """Generate a synthetic supervised dataset for demos and tests."""
    rng = np.random.default_rng(random_state)
    n_shootout = int(round(n * 0.58))
    n_open_play = n - n_shootout

    shootout = _sample_shootout_context(rng, n_shootout)
    shootout["goal_entry_u"] = rng.beta(1.35, 1.35, n_shootout)
    shootout["goal_entry_v"] = rng.beta(1.15, 2.15, n_shootout)

    keeper_u, keeper_v = _sample_pitch_actor_uv(rng, n_open_play)
    ball_u, ball_v = _sample_pitch_actor_uv(rng, n_open_play)
    open_play = pd.DataFrame(
        {
            "keeper_center_u": keeper_u,
            "keeper_center_v": keeper_v,
            "ball_position_u": ball_u,
            "ball_position_v": ball_v,
            "goal_entry_u": rng.beta(1.35, 1.35, n_open_play),
            "goal_entry_v": rng.beta(1.15, 2.15, n_open_play),
            "shot_ball_speed": np.clip(rng.normal(82, 18, n_open_play), 42, 132),
            "time_ball_to_goal": np.clip(rng.normal(0.55, 0.18, n_open_play), 0.16, 1.25),
            "reaction_time": np.clip(rng.normal(0.32, 0.07, n_open_play), 0.16, 0.62),
            "team_kicks_first": np.full(n_open_play, 0.5),
            "shootout_round": np.zeros(n_open_play),
            "partial_score": np.zeros(n_open_play),
            "kick_importance": np.clip(rng.normal(0.32, 0.08, n_open_play), 0.12, 0.55),
            "is_shootout_context": np.zeros(n_open_play),
        }
    )

    body_cols = {
        "keeper_body_width": np.clip(rng.normal(0.78, 0.14, n), 0.35, 1.25),
        "keeper_body_height": np.clip(rng.normal(1.62, 0.18, n), 1.05, 2.10),
        "keeper_hand_span": np.clip(rng.normal(1.82, 0.32, n), 0.90, 2.80),
        "keeper_foot_span": np.clip(rng.normal(0.72, 0.16, n), 0.30, 1.25),
        "keeper_polygon_area_uv": np.clip(rng.normal(0.12, 0.04, n), 0.03, 0.26),
        "keeper_pose_confidence": rng.beta(5.0, 2.0, n),
    }
    df = pd.concat([shootout, open_play], ignore_index=True)
    for key, values in body_cols.items():
        df[key] = values
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    df = add_derived_save_features(df)
    probabilities = df.apply(_synthetic_save_probability, axis=1).to_numpy()
    probabilities = np.clip(probabilities, 0.04, 0.96)
    labels = rng.binomial(1, probabilities)
    df["save_label"] = labels
    df["outcome"] = np.where(labels == 1, "save", "goal")
    df["synthetic_probability"] = probabilities
    df["synthetic_source"] = "apeste_ph_2010_stylized"
    df["empirical_goal_probability"] = df.apply(_apeste_ph_goal_probability, axis=1)
    return df


def estimate_reach_probability(*args, **kwargs):
    """Future extension point kept for API compatibility."""
    raise NotImplementedError(
        "Use `scripts/train_save_probability_model.py` for the first supervised save-probability model."
    )
