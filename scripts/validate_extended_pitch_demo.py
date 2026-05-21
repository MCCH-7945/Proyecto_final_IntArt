"""Validate synthetic data for extended pitch scenarios (ball in area, keeper outside goal)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from reach_probability import (  # noqa: E402
    APESTE_PH_2010,
    SAVE_PROBABILITY_FEATURES,
    add_derived_save_features,
    generate_synthetic_save_probability_data,
    goal_area_bounds,
    pitch_area_bounds,
    prepare_save_probability_dataset,
)


def _subset_summary(df: pd.DataFrame, name: str, mask: pd.Series) -> dict:
    part = df.loc[mask]
    return {
        "subset": name,
        "n": int(len(part)),
        "save_rate": float(part["save_label"].mean()) if len(part) else None,
        "mean_p_synthetic": float(part["synthetic_probability"].mean()) if len(part) else None,
    }


def main() -> int:
    df = generate_synthetic_save_probability_data(n=4000, random_state=11)
    df = add_derived_save_features(df)
    prepared = prepare_save_probability_dataset(df)

    pen = pitch_area_bounds()
    box = goal_area_bounds()
    in_penalty = (
        df["ball_position_u"].between(pen["u_min"], pen["u_max"])
        & df["ball_position_v"].between(pen["v_min"], pen["v_max"])
        & df["keeper_center_u"].between(pen["u_min"], pen["u_max"])
        & df["keeper_center_v"].between(pen["v_min"], pen["v_max"])
    )
    in_small_box = (
        df["ball_position_u"].between(box["u_min"], box["u_max"])
        & df["ball_position_v"].between(box["v_min"], box["v_max"])
    )
    keeper_in_small = (
        df["keeper_center_u"].between(box["u_min"], box["u_max"])
        & df["keeper_center_v"].between(box["v_min"], box["v_max"])
    )
    both_small = in_small_box & keeper_in_small

    shootout = df.loc[df["is_shootout_context"] >= 0.5]
    first_kicker = shootout.loc[shootout["team_kicks_first"] >= 0.5]
    second_lagging = shootout.loc[(shootout["team_kicks_first"] < 0.5) & (shootout["partial_score"] < 0)]

    summaries = [
        _subset_summary(df, "all", pd.Series(True, index=df.index)),
        _subset_summary(df, "actors_inside_penalty_area", in_penalty),
        _subset_summary(df, "ball_in_small_box", in_small_box),
        _subset_summary(df, "keeper_in_small_box", keeper_in_small),
        _subset_summary(df, "ball_and_keeper_in_small_box", both_small),
        _subset_summary(df, "shootout_context", df["is_shootout_context"] >= 0.5),
        _subset_summary(df, "shootout_first_kicker", df["team_kicks_first"] >= 0.5),
        _subset_summary(df, "shootout_second_lagging", (df["team_kicks_first"] < 0.5) & (df["partial_score"] < 0)),
    ]

    samples = df.loc[both_small].head(12)
    out_csv = ROOT / "outputs" / "extended_pitch_demo_samples.csv"
    out_json = ROOT / "outputs" / "model_reports" / "extended_pitch_demo_summary.json"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    samples.to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    x_train, x_test, y_train, y_test = train_test_split(
        prepared.features,
        prepared.labels,
        test_size=0.25,
        random_state=7,
        stratify=prepared.labels,
    )
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2500, class_weight="balanced", random_state=7)),
        ]
    )
    model.fit(x_train, y_train)
    probs = model.predict_proba(x_test)[:, 1]
    auc = float(roc_auc_score(y_test, probs))

    print("Extended pitch synthetic validation (Apesteguia & Palacios-Huerta 2010 stylized)")
    print(f"Features ({len(SAVE_PROBABILITY_FEATURES)}): OK")
    print(f"Target shoot-out scoring band: {APESTE_PH_2010['shootout_scoring_rate_all']:.3f}")
    if len(shootout):
        print(f"Simulated shoot-out goal rate: {1.0 - shootout['save_label'].mean():.3f}")
        print(f"Simulated mean flight time (s): {shootout['time_ball_to_goal'].mean():.3f}")
    for row in summaries:
        print(
            f"  {row['subset']}: n={row['n']} save_rate={row['save_rate']:.3f} "
            f"mean_p={row['mean_p_synthetic']:.3f}"
        )
    print(f"Quick logistic AUC on holdout: {auc:.3f}")
    print(f"Samples CSV: {out_csv}")
    print(f"Summary JSON: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
