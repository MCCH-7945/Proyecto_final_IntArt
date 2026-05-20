"""Input/output helpers for the shot analysis pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


FRAME_LEVEL_COLUMNS = [
    "video_id",
    "frame",
    "time_sec",
    "ball_x",
    "ball_y",
    "ball_detected",
    "ball_confidence",
    "ball_x_smooth",
    "ball_y_smooth",
    "ball_vx",
    "ball_vy",
    "ball_speed",
    "ball_acceleration",
    "event_type",
    "inside_penalty_area",
    "is_candidate_shot_frame",
    "is_selected_shot_frame",
    "is_play_end_frame",
]


SHOT_LEVEL_COLUMNS = [
    "video_id",
    "shot_frame",
    "shot_time_sec",
    "shot_ball_x",
    "shot_ball_y",
    "shot_ball_vx",
    "shot_ball_vy",
    "shot_ball_speed",
    "inside_penalty_area",
    "play_end_frame",
    "play_end_time_sec",
    "play_end_ball_x",
    "play_end_ball_y",
    "play_end_ball_speed",
    "end_inside_penalty_area",
    "play_outcome",
    "play_end_confidence",
    "goal_entry_frame",
    "goal_entry_x_px",
    "goal_entry_y_px",
    "goal_entry_u",
    "goal_entry_v",
    "goal_entry_x_m",
    "goal_entry_z_m",
    "goal_zone_horizontal",
    "goal_zone_vertical",
    "goal_entry_confidence",
    "shot_confidence",
    "pre_shot_start_frame",
    "pre_shot_end_frame",
    "post_shot_start_frame",
    "post_shot_end_frame",
    "status",
    "notes",
]


def ensure_parent_dir(path: str | Path) -> None:
    """Create the parent directory for a file path."""
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def ensure_dir(path: str | Path) -> None:
    """Create a directory if needed."""
    Path(path).expanduser().resolve().mkdir(parents=True, exist_ok=True)


def load_pitch_config(config_path: str | Path | None) -> tuple[dict[str, Any] | None, str | None]:
    """Load the pitch configuration JSON.

    Returns `(config, note)`. If the file is missing or invalid, `config` is
    `None` and `note` explains the issue.
    """
    if not config_path:
        return None, "Pitch config missing."

    path = Path(config_path)
    if not path.exists():
        return None, "Pitch config missing."

    try:
        with path.open("r", encoding="utf-8") as file:
            config = json.load(file)
    except json.JSONDecodeError as exc:
        return None, f"Pitch config could not be parsed: {exc}"

    return config, None


def get_video_id(video_path: str | Path, config: dict[str, Any] | None = None) -> str:
    """Resolve a stable video id from config or file name."""
    if config and config.get("video_id"):
        return str(config["video_id"])
    return Path(video_path).stem


def ensure_columns(df: pd.DataFrame, required_columns: list[str]) -> pd.DataFrame:
    """Add missing columns and order the required columns first."""
    output = df.copy()
    for column in required_columns:
        if column not in output.columns:
            output[column] = pd.NA
    extras = [column for column in output.columns if column not in required_columns]
    return output[required_columns + extras]


def write_frame_level_csv(frame_df: pd.DataFrame, output_path: str | Path) -> None:
    """Write frame-level output with the expected columns."""
    ensure_parent_dir(output_path)
    frame_df = ensure_columns(frame_df, FRAME_LEVEL_COLUMNS)
    frame_df.to_csv(output_path, index=False)


def write_shot_level(shot_df: pd.DataFrame, output_path: str | Path) -> None:
    """Write shot-level output as CSV or JSON based on extension."""
    ensure_parent_dir(output_path)
    shot_df = ensure_columns(shot_df, SHOT_LEVEL_COLUMNS)
    suffix = Path(output_path).suffix.lower()
    if suffix == ".json":
        shot_df.to_json(output_path, orient="records", indent=2)
    else:
        shot_df.to_csv(output_path, index=False)
