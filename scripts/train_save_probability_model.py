"""Train a first supervised goalkeeper save-probability model.

The preferred input is a real annotated table with:

    keeper_state + shot_state + outcome

For a presentation/demo, `--demo-synthetic` trains the same ML pipeline on
synthetic labels generated from a geometric proxy. That proves the wiring works,
but it must not be described as SoccerNet-trained.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from reach_probability import (  # noqa: E402
    SAVE_PROBABILITY_FEATURES,
    add_derived_save_features,
    generate_synthetic_save_probability_data,
    prepare_save_probability_dataset,
)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".json":
        return pd.read_json(path)
    return pd.read_csv(path)


def _build_model(random_state: int):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )


def _evaluate_model(model, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics: dict[str, Any] = {
        "n_test": int(len(y_test)),
        "positive_rate_test": float(np.mean(y_test)),
        "accuracy_0_50": float(accuracy_score(y_test, predictions)),
        "brier_score": float(brier_score_loss(y_test, probabilities)),
        "log_loss": float(log_loss(y_test, probabilities, labels=[0, 1])),
    }
    if len(set(y_test.astype(int))) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_test, probabilities))
    else:
        metrics["roc_auc"] = None
    return metrics


def _pipeline_to_web_json(model, metrics: dict[str, Any], source: str, feature_columns: list[str]) -> dict[str, Any]:
    imputer = model.named_steps["imputer"]
    scaler = model.named_steps["scaler"]
    classifier = model.named_steps["classifier"]
    return {
        "model_kind": "logistic_regression",
        "source": source,
        "trained_on_synthetic_data": source == "demo_synthetic",
        "positive_label": "save",
        "feature_columns": feature_columns,
        "imputer_median": [float(value) for value in imputer.statistics_],
        "scaler_mean": [float(value) for value in scaler.mean_],
        "scaler_scale": [float(value) if value != 0 else 1.0 for value in scaler.scale_],
        "coef": [float(value) for value in classifier.coef_[0]],
        "intercept": float(classifier.intercept_[0]),
        "metrics": metrics,
        "disclaimer": (
            "Synthetic demo model. Replace with SoccerNet/custom annotated shots before making performance claims."
            if source == "demo_synthetic"
            else "Trained from user-provided annotated shot table."
        ),
    }


def _default_grid_rows() -> pd.DataFrame:
    values = []
    for u in np.linspace(0.02, 0.98, 49):
        for v in np.linspace(0.02, 0.98, 25):
            keeper_u = 0.50
            keeper_v = 0.22
            values.append(
                {
                    "keeper_center_u": keeper_u,
                    "keeper_center_v": keeper_v,
                    "ball_position_u": 0.5,
                    "ball_position_v": 0.0,
                    "team_kicks_first": 1.0,
                    "shootout_round": 3.0,
                    "partial_score": -1.0,
                    "kick_importance": 0.72,
                    "is_shootout_context": 1.0,
                    "keeper_body_width": 0.78,
                    "keeper_body_height": 1.62,
                    "keeper_hand_span": 1.82,
                    "keeper_foot_span": 0.72,
                    "keeper_polygon_area_uv": 0.12,
                    "keeper_pose_confidence": 0.72,
                    "goal_entry_u": u,
                    "goal_entry_v": v,
                    "shot_ball_speed": 82,
                    "time_ball_to_goal": 0.35,
                    "reaction_time": 0.32,
                    "lateral_delta_m": (u - keeper_u) * 7.32,
                    "vertical_delta_m": (v - keeper_v) * 2.44,
                    "distance_keeper_to_target_m": float(np.hypot((u - keeper_u) * 7.32, (v - keeper_v) * 2.44)),
                }
            )
    return pd.DataFrame(values)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a save-probability model from keeper/shot examples.")
    parser.add_argument("--data", type=Path, default=None, help="Annotated CSV/JSON with keeper_state + shot_state + outcome.")
    parser.add_argument("--out-model", type=Path, default=Path("models/save_probability/save_probability_model.pkl"))
    parser.add_argument("--out-report", type=Path, default=Path("outputs/model_reports/save_probability_metrics.json"))
    parser.add_argument("--web-json", type=Path, default=Path("web_demo/save_probability_model.json"))
    parser.add_argument("--out-training-csv", type=Path, default=None, help="Optional copy of the training table used.")
    parser.add_argument("--out-grid-csv", type=Path, default=Path("outputs/model_reports/save_probability_grid_demo.csv"))
    parser.add_argument("--outcome-col", default="outcome", help="Outcome column if not using save_label.")
    parser.add_argument("--demo-synthetic", action="store_true", help="Train on synthetic demo data if no real table is ready.")
    parser.add_argument("--n-synthetic", type=int, default=2200)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=7)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.data:
        df = _read_table(args.data)
        source = str(args.data)
    elif args.demo_synthetic:
        df = generate_synthetic_save_probability_data(n=args.n_synthetic, random_state=args.random_state)
        source = "demo_synthetic"
    else:
        print("[ERROR] Provide --data annotated_shots.csv or use --demo-synthetic for a presentation-only model.")
        return 1

    prepared = prepare_save_probability_dataset(df, outcome_col=args.outcome_col)

    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(
        prepared.features,
        prepared.labels,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=prepared.labels if prepared.labels.nunique() == 2 else None,
    )
    model = _build_model(random_state=args.random_state)
    model.fit(x_train, y_train)
    metrics = _evaluate_model(model, x_test, y_test)
    metrics.update(
        {
            "n_total": int(len(prepared.labels)),
            "n_train": int(len(y_train)),
            "positive_rate_total": float(prepared.labels.mean()),
            "feature_columns": SAVE_PROBABILITY_FEATURES,
            "source": source,
            "trained_on_synthetic_data": source == "demo_synthetic",
        }
    )

    _ensure_parent(args.out_model)
    with args.out_model.open("wb") as file:
        pickle.dump(model, file)

    _ensure_parent(args.out_report)
    args.out_report.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    web_json = _pipeline_to_web_json(model, metrics, source, SAVE_PROBABILITY_FEATURES)
    _ensure_parent(args.web_json)
    args.web_json.write_text(json.dumps(web_json, indent=2), encoding="utf-8")

    if args.out_training_csv:
        _ensure_parent(args.out_training_csv)
        prepared.frame.to_csv(args.out_training_csv, index=False)

    if args.out_grid_csv:
        grid = add_derived_save_features(_default_grid_rows())
        grid["p_save"] = model.predict_proba(grid[SAVE_PROBABILITY_FEATURES])[:, 1]
        _ensure_parent(args.out_grid_csv)
        grid.to_csv(args.out_grid_csv, index=False)

    print(f"Model: {args.out_model}")
    print(f"Report: {args.out_report}")
    print(f"Web JSON: {args.web_json}")
    print(f"Rows: {metrics['n_total']} | AUC: {metrics['roc_auc']} | Brier: {metrics['brier_score']:.4f}")
    if source == "demo_synthetic":
        print("WARNING: trained on synthetic demo data, not SoccerNet/custom annotations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
