"""Stage 1 pipeline: ball tracking, shot detection, and goal-entry mapping."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from .ball_tracking import VideoOpenError, detect_ball_positions
    from .goal_mapping import (
        GOAL_HEIGHT_M,
        GOAL_WIDTH_M,
        classify_goal_zone,
        compute_goal_homography,
        estimate_goal_entry_point,
        map_point_to_goal_uv,
    )
    from .io_utils import (
        FRAME_LEVEL_COLUMNS,
        SHOT_LEVEL_COLUMNS,
        ensure_columns,
        get_video_id,
        load_pitch_config,
        write_frame_level_csv,
        write_shot_level,
    )
    from .pitch_geometry import add_event_inside_penalty_area_column
    from .trajectory import estimate_ball_velocity, smooth_ball_trajectory
    from .video_annotation import annotate_video, save_review_frames
except ImportError:  # pragma: no cover - allows `python src/shot_detector.py`
    from ball_tracking import VideoOpenError, detect_ball_positions
    from goal_mapping import (
        GOAL_HEIGHT_M,
        GOAL_WIDTH_M,
        classify_goal_zone,
        compute_goal_homography,
        estimate_goal_entry_point,
        map_point_to_goal_uv,
    )
    from io_utils import (
        FRAME_LEVEL_COLUMNS,
        SHOT_LEVEL_COLUMNS,
        ensure_columns,
        get_video_id,
        load_pitch_config,
        write_frame_level_csv,
        write_shot_level,
    )
    from pitch_geometry import add_event_inside_penalty_area_column
    from trajectory import estimate_ball_velocity, smooth_ball_trajectory
    from video_annotation import annotate_video, save_review_frames


def _finite_series(series: pd.Series) -> pd.Series:
    return series[np.isfinite(series)]


def _robust_zscore(series: pd.Series) -> pd.Series:
    values = series.astype(float)
    finite = _finite_series(values)
    if len(finite) < 3:
        return pd.Series(np.nan, index=series.index)
    median = float(finite.median())
    mad = float((finite - median).abs().median())
    if mad > 1e-9:
        z = 0.6745 * (values - median) / mad
    else:
        std = float(finite.std(ddof=0))
        if std <= 1e-9:
            return pd.Series(np.nan, index=series.index)
        z = (values - float(finite.mean())) / std
    return z


def _direction_consistency(ball_df: pd.DataFrame, frame: int, post_window: int) -> float:
    window = ball_df[(ball_df["frame"] > frame) & (ball_df["frame"] <= frame + post_window)]
    vectors = window[["ball_vx", "ball_vy"]].to_numpy(dtype=float)
    speeds = np.linalg.norm(vectors, axis=1)
    valid = np.isfinite(speeds) & (speeds > 1e-6)
    vectors = vectors[valid]
    speeds = speeds[valid]
    if len(vectors) < 2:
        return 0.0
    unit = vectors / speeds[:, None]
    mean_vector = unit.mean(axis=0)
    return float(np.clip(np.linalg.norm(mean_vector), 0.0, 1.0))


def _valid_detection_ratio(ball_df: pd.DataFrame, frame: int, window: int) -> float:
    local = ball_df[(ball_df["frame"] >= frame - window) & (ball_df["frame"] <= frame + window)]
    if local.empty:
        return 0.0
    valid = local["ball_x_smooth"].notna() & local["ball_y_smooth"].notna()
    return float(valid.mean())


def detect_shot_frame(
    ball_df: pd.DataFrame,
    fps: float,
    threshold_z: float = 2.0,
    speed_percentile: float = 90.0,
    pre_window: int | None = None,
    post_window: int | None = None,
) -> dict[str, Any]:
    """Detect the most likely shot frame from ball kinematics."""
    if ball_df.empty or "ball_speed" not in ball_df.columns or "ball_acceleration" not in ball_df.columns:
        return {
            "shot_frame": None,
            "shot_time_sec": None,
            "shot_confidence": 0.0,
            "candidate_frames": [],
            "reason": "Missing velocity or acceleration columns.",
        }

    fps = float(fps) if fps and np.isfinite(fps) and fps > 0 else 30.0
    pre_window = pre_window or max(4, int(round(fps * 0.35)))
    post_window = post_window or max(5, int(round(fps * 0.45)))

    output = ball_df.copy()
    output["ball_acceleration_zscore"] = _robust_zscore(output["ball_acceleration"])
    valid_speed = _finite_series(output["ball_speed"])
    valid_acc = _finite_series(output["ball_acceleration"])
    if len(valid_speed) < 6 or len(valid_acc) < 4:
        return {
            "shot_frame": None,
            "shot_time_sec": None,
            "shot_confidence": 0.0,
            "candidate_frames": [],
            "reason": "Not enough valid ball detections to estimate shot reliably.",
        }

    speed_threshold = float(np.nanpercentile(valid_speed, speed_percentile))
    candidates = output[
        (output["ball_acceleration_zscore"] > threshold_z)
        & (output["ball_speed"] >= speed_threshold)
        & output["ball_x_smooth"].notna()
        & output["ball_y_smooth"].notna()
    ].copy()

    if candidates.empty:
        return {
            "shot_frame": None,
            "shot_time_sec": None,
            "shot_confidence": 0.0,
            "candidate_frames": [],
            "reason": "No clear sudden and persistent speed increase found.",
        }

    scored: list[dict[str, Any]] = []
    for _, candidate in candidates.iterrows():
        frame = int(candidate["frame"])
        pre = output[(output["frame"] >= frame - pre_window) & (output["frame"] < frame)]["ball_speed"]
        post = output[(output["frame"] >= frame) & (output["frame"] <= frame + post_window)]["ball_speed"]
        pre_mean = float(np.nanmean(pre)) if np.isfinite(pre).any() else 0.0
        post_mean = float(np.nanmean(post)) if np.isfinite(post).any() else 0.0
        post_valid_count = int(np.isfinite(post).sum())
        if post_valid_count < max(3, post_window // 3):
            continue

        speed_gain = post_mean / max(pre_mean, 1.0)
        persistence = post_mean > max(pre_mean * 1.15, speed_threshold * 0.75)
        direction = _direction_consistency(output, frame, post_window)
        valid_ratio = _valid_detection_ratio(output, frame, max(pre_window, post_window))

        if not persistence or direction < 0.45 or valid_ratio < 0.45:
            continue

        z_component = float(np.clip((candidate["ball_acceleration_zscore"] - threshold_z) / 4.0, 0.0, 1.0))
        speed_component = float(np.clip(candidate["ball_speed"] / max(speed_threshold, 1.0) - 1.0, 0.0, 1.0))
        gain_component = float(np.clip((speed_gain - 1.0) / 3.0, 0.0, 1.0))
        confidence = 0.25 + 0.25 * z_component + 0.20 * speed_component + 0.20 * direction + 0.10 * gain_component
        confidence *= float(np.clip(valid_ratio, 0.0, 1.0))

        scored.append(
            {
                "frame": frame,
                "time_sec": float(candidate["time_sec"]),
                "confidence": float(np.clip(confidence, 0.0, 0.98)),
                "score": float(confidence),
            }
        )

    candidate_frames = [int(frame) for frame in candidates["frame"].tolist()]
    if not scored:
        return {
            "shot_frame": None,
            "shot_time_sec": None,
            "shot_confidence": 0.0,
            "candidate_frames": candidate_frames,
            "reason": "Candidates were not persistent or directionally consistent enough.",
        }

    best = max(scored, key=lambda item: item["score"])
    if best["confidence"] < 0.45:
        return {
            "shot_frame": None,
            "shot_time_sec": None,
            "shot_confidence": best["confidence"],
            "candidate_frames": candidate_frames,
            "reason": "Best candidate confidence below threshold.",
        }

    return {
        "shot_frame": int(best["frame"]),
        "shot_time_sec": float(best["time_sec"]),
        "shot_confidence": float(best["confidence"]),
        "candidate_frames": candidate_frames,
        "reason": "Sudden and persistent speed increase detected.",
    }


def _is_finite_number(value: Any) -> bool:
    try:
        return bool(np.isfinite(value))
    except (TypeError, ValueError):
        return False


def _frame_time(frame_df: pd.DataFrame, frame: int | None, fps: float) -> float | None:
    if frame is None:
        return None
    rows = frame_df[frame_df["frame"] == int(frame)]
    if not rows.empty and _is_finite_number(rows.iloc[0].get("time_sec")):
        return float(rows.iloc[0]["time_sec"])
    return float(frame / fps) if fps > 0 else None


def _row_at_frame(frame_df: pd.DataFrame, frame: int | None) -> pd.Series | None:
    if frame is None:
        return None
    rows = frame_df[frame_df["frame"] == int(frame)]
    if rows.empty:
        return None
    return rows.iloc[0]


def estimate_play_end_event(
    ball_df: pd.DataFrame,
    shot_frame: int | None,
    goal_entry: dict[str, Any] | None,
    fps: float,
    frame_width: int | None = None,
    frame_height: int | None = None,
) -> dict[str, Any]:
    """Estimate when the shot sequence concludes.

    This is intentionally conservative. Goal is inferred from a goal-frame
    crossing. Save/stoppage is inferred from a sustained speed drop. Out/miss is
    inferred from leaving the visible frame or disappearing after fast movement.
    """
    if shot_frame is None:
        return {
            "play_end_frame": None,
            "play_end_time_sec": None,
            "play_outcome": "unknown",
            "play_end_confidence": 0.0,
            "reason": "No shot frame available.",
        }

    fps = float(fps) if fps and np.isfinite(fps) and fps > 0 else 30.0
    candidates: list[dict[str, Any]] = []

    if goal_entry:
        goal_frame = goal_entry.get("goal_entry_frame")
        goal_confidence = float(goal_entry.get("goal_entry_confidence") or 0.0)
        if goal_frame is not None and goal_confidence >= 0.25:
            candidates.append(
                {
                    "play_end_frame": int(goal_frame),
                    "play_outcome": "goal",
                    "play_end_confidence": float(min(0.98, max(0.50, goal_confidence))),
                    "reason": "Ball trajectory crossed the configured goal frame.",
                }
            )

    post_df = ball_df[ball_df["frame"] >= int(shot_frame)].copy()
    post_df = post_df[post_df["ball_x_smooth"].notna() & post_df["ball_y_smooth"].notna()]
    if post_df.empty:
        return {
            "play_end_frame": None,
            "play_end_time_sec": None,
            "play_outcome": "unknown",
            "play_end_confidence": 0.0,
            "reason": "No post-shot ball positions available.",
        }

    valid_speeds = _finite_series(post_df["ball_speed"]) if "ball_speed" in post_df.columns else pd.Series(dtype=float)
    reference_speed = float(np.nanpercentile(valid_speeds, 70)) if len(valid_speeds) else 0.0
    stop_threshold = max(30.0, reference_speed * 0.20)
    min_after_frames = max(3, int(round(fps * 0.25)))
    stop_streak = max(3, int(round(fps * 0.25)))
    candidate_stop_frames = post_df[post_df["frame"] >= int(shot_frame) + min_after_frames].copy()
    low_speed = candidate_stop_frames["ball_speed"].fillna(np.inf) <= stop_threshold
    streak_count = 0
    for (_, row), is_low in zip(candidate_stop_frames.iterrows(), low_speed.tolist()):
        streak_count = streak_count + 1 if is_low else 0
        if streak_count >= stop_streak:
            candidates.append(
                {
                    "play_end_frame": int(row["frame"]),
                    "play_outcome": "stopped_or_save",
                    "play_end_confidence": 0.55,
                    "reason": "Sustained post-shot ball speed drop detected.",
                }
            )
            break

    if frame_width and frame_height:
        margin = max(8.0, min(float(frame_width), float(frame_height)) * 0.03)
        for _, row in post_df[post_df["frame"] >= int(shot_frame) + min_after_frames].iterrows():
            x = float(row["ball_x_smooth"])
            y = float(row["ball_y_smooth"])
            near_edge = x <= margin or x >= frame_width - margin or y <= margin or y >= frame_height - margin
            if near_edge:
                candidates.append(
                    {
                        "play_end_frame": int(row["frame"]),
                        "play_outcome": "out_or_miss",
                        "play_end_confidence": 0.45,
                        "reason": "Ball reached the visible frame boundary after the shot.",
                    }
                )
                break

    max_gap = max(6, int(round(fps * 0.25)))
    previous_row = None
    for _, row in post_df.iterrows():
        if previous_row is not None:
            gap = int(row["frame"]) - int(previous_row["frame"])
            previous_speed = previous_row.get("ball_speed")
            was_fast = _is_finite_number(previous_speed) and float(previous_speed) >= max(stop_threshold * 2.0, reference_speed * 0.6)
            if gap > max_gap and was_fast:
                candidates.append(
                    {
                        "play_end_frame": int(previous_row["frame"]),
                        "play_outcome": "out_or_miss",
                        "play_end_confidence": 0.40,
                        "reason": "Ball disappeared after fast post-shot movement.",
                    }
                )
                break
        previous_row = row

    if candidates:
        candidates = [candidate for candidate in candidates if int(candidate["play_end_frame"]) >= int(shot_frame)]
        best = min(candidates, key=lambda item: int(item["play_end_frame"]))
        best["play_end_time_sec"] = _frame_time(ball_df, best["play_end_frame"], fps)
        return best

    last_valid = post_df.iloc[-1]
    end_frame = int(last_valid["frame"])
    return {
        "play_end_frame": end_frame,
        "play_end_time_sec": _frame_time(ball_df, end_frame, fps),
        "play_outcome": "unknown",
        "play_end_confidence": 0.10,
        "reason": "No goal, stoppage, or out-of-frame event could be inferred.",
    }


def _empty_shot_row(video_id: str, status: str, notes: str) -> dict[str, Any]:
    row = {column: None for column in SHOT_LEVEL_COLUMNS}
    row.update(
        {
            "video_id": video_id,
            "goal_zone_horizontal": "unknown",
            "goal_zone_vertical": "unknown",
            "goal_entry_confidence": 0.0,
            "shot_confidence": 0.0,
            "status": status,
            "notes": notes,
        }
    )
    return row


def _value_or_none(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        if not np.isfinite(value):
            return None
    except TypeError:
        return value
    return value


def _append_note(notes: list[str], note: str | None) -> None:
    if note and note not in notes:
        notes.append(note)


def build_shot_level_row(
    frame_df: pd.DataFrame,
    shot_result: dict[str, Any],
    goal_corners: dict[str, Any] | None,
    goal_entry: dict[str, Any],
    play_end_event: dict[str, Any],
    video_id: str,
    status: str,
    notes: list[str],
    window_frames: int = 30,
) -> dict[str, Any]:
    """Build the one-row shot-level output."""
    shot_frame = shot_result.get("shot_frame")
    if shot_frame is None:
        return _empty_shot_row(video_id, status=status, notes=" ".join(notes).strip())

    rows = frame_df[frame_df["frame"] == int(shot_frame)]
    if rows.empty:
        return _empty_shot_row(video_id, status="no_reliable_shot_detected", notes="Shot frame not present in frame data.")
    row = rows.iloc[0]

    shot_x = _value_or_none(row.get("ball_x_smooth", row.get("ball_x")))
    shot_y = _value_or_none(row.get("ball_y_smooth", row.get("ball_y")))

    entry_x = goal_entry.get("goal_entry_x_px")
    entry_y = goal_entry.get("goal_entry_y_px")
    entry_frame = goal_entry.get("goal_entry_frame")
    entry_confidence = float(goal_entry.get("goal_entry_confidence") or 0.0)

    goal_entry_u = None
    goal_entry_v = None
    if goal_corners and entry_x is not None and entry_y is not None:
        try:
            homography = compute_goal_homography(goal_corners)
            goal_entry_u, goal_entry_v = map_point_to_goal_uv((entry_x, entry_y), homography)
        except Exception:
            goal_entry_u, goal_entry_v = None, None

    if goal_corners is None:
        _append_note(notes, "Goal corners missing.")
    elif goal_entry.get("method") == "no_clear_goal_crossing":
        _append_note(notes, "No clear goal crossing detected.")
    if play_end_event.get("play_outcome") == "unknown":
        _append_note(notes, "Play end could not be inferred confidently.")

    zones = classify_goal_zone(goal_entry_u, goal_entry_v)
    play_end_frame = play_end_event.get("play_end_frame")
    play_end_row = _row_at_frame(frame_df, play_end_frame)
    row_dict = {
        "video_id": video_id,
        "shot_frame": int(shot_frame),
        "shot_time_sec": _value_or_none(shot_result.get("shot_time_sec")),
        "shot_ball_x": shot_x,
        "shot_ball_y": shot_y,
        "shot_ball_vx": _value_or_none(row.get("ball_vx")),
        "shot_ball_vy": _value_or_none(row.get("ball_vy")),
        "shot_ball_speed": _value_or_none(row.get("ball_speed")),
        "inside_penalty_area": _value_or_none(row.get("inside_penalty_area")),
        "play_end_frame": _value_or_none(play_end_frame),
        "play_end_time_sec": _value_or_none(play_end_event.get("play_end_time_sec")),
        "play_end_ball_x": None if play_end_row is None else _value_or_none(play_end_row.get("ball_x_smooth", play_end_row.get("ball_x"))),
        "play_end_ball_y": None if play_end_row is None else _value_or_none(play_end_row.get("ball_y_smooth", play_end_row.get("ball_y"))),
        "play_end_ball_speed": None if play_end_row is None else _value_or_none(play_end_row.get("ball_speed")),
        "end_inside_penalty_area": None if play_end_row is None else _value_or_none(play_end_row.get("inside_penalty_area")),
        "play_outcome": play_end_event.get("play_outcome", "unknown"),
        "play_end_confidence": float(play_end_event.get("play_end_confidence") or 0.0),
        "goal_entry_frame": _value_or_none(entry_frame),
        "goal_entry_x_px": _value_or_none(entry_x),
        "goal_entry_y_px": _value_or_none(entry_y),
        "goal_entry_u": _value_or_none(goal_entry_u),
        "goal_entry_v": _value_or_none(goal_entry_v),
        "goal_entry_x_m": None if goal_entry_u is None else float(GOAL_WIDTH_M * goal_entry_u),
        "goal_entry_z_m": None if goal_entry_v is None else float(GOAL_HEIGHT_M * goal_entry_v),
        "goal_zone_horizontal": zones["goal_zone_horizontal"],
        "goal_zone_vertical": zones["goal_zone_vertical"],
        "goal_entry_confidence": entry_confidence,
        "shot_confidence": float(shot_result.get("shot_confidence") or 0.0),
        "pre_shot_start_frame": max(0, int(shot_frame) - window_frames),
        "pre_shot_end_frame": int(shot_frame),
        "post_shot_start_frame": int(shot_frame),
        "post_shot_end_frame": int(play_end_frame) if play_end_frame is not None else int(shot_frame) + window_frames,
        "status": status,
        "notes": " ".join(notes).strip(),
    }
    return row_dict


def _prepare_frame_flags(
    frame_df: pd.DataFrame,
    shot_result: dict[str, Any],
    play_end_event: dict[str, Any] | None = None,
) -> pd.DataFrame:
    output = frame_df.copy()
    output.attrs.update(frame_df.attrs)
    candidate_frames = set(int(frame) for frame in shot_result.get("candidate_frames", []))
    shot_frame = shot_result.get("shot_frame")
    play_end_frame = play_end_event.get("play_end_frame") if play_end_event else None
    output["event_type"] = ""
    output["is_candidate_shot_frame"] = output["frame"].isin(candidate_frames)
    output["is_selected_shot_frame"] = False
    output["is_play_end_frame"] = False
    if shot_frame is not None:
        output.loc[output["frame"] == int(shot_frame), "is_selected_shot_frame"] = True
        output.loc[output["frame"] == int(shot_frame), "event_type"] = "shot"
    if play_end_frame is not None:
        play_end_mask = output["frame"] == int(play_end_frame)
        output.loc[play_end_mask, "is_play_end_frame"] = True
        existing = output.loc[play_end_mask, "event_type"].astype(str)
        output.loc[play_end_mask, "event_type"] = np.where(existing == "shot", "shot_and_play_end", "play_end")
    return output


def run_pipeline(
    video_path: str | Path,
    model_path: str | Path | None,
    config_path: str | Path | None,
    frame_output: str | Path,
    shot_output: str | Path,
    only_inside_box: bool = False,
    annotated_video: str | Path | None = None,
    save_review_frames_dir: str | Path | None = None,
    show_progress: bool = True,
    min_valid_detections: int = 8,
) -> dict[str, Any]:
    """Run the complete Stage 1 pipeline and write requested outputs."""
    config, config_note = load_pitch_config(config_path)
    video_id = get_video_id(video_path, config)
    notes: list[str] = []
    _append_note(notes, config_note)

    try:
        frame_df = detect_ball_positions(video_path, model_path=model_path, video_id=video_id, show_progress=show_progress)
    except VideoOpenError:
        frame_df = pd.DataFrame(columns=FRAME_LEVEL_COLUMNS)
        write_frame_level_csv(frame_df, frame_output)
        shot_row = _empty_shot_row(video_id, "error", "Video could not be opened.")
        write_shot_level(pd.DataFrame([shot_row]), shot_output)
        return {"frame_df": frame_df, "shot_df": pd.DataFrame([shot_row]), "status": "error"}

    tracking_note = frame_df.attrs.get("tracking_note")
    _append_note(notes, tracking_note)
    fps = float(frame_df.attrs.get("fps", 30.0))

    frame_df = smooth_ball_trajectory(frame_df, min_valid_detections=min_valid_detections)
    frame_df = estimate_ball_velocity(frame_df, fps=fps)

    penalty_area = config.get("penalty_area_polygon") if config else None
    goal_corners = config.get("goal_corners") if config else None
    if config and not penalty_area:
        _append_note(notes, "Penalty area polygon missing.")

    valid_detections = int(frame_df["ball_detected"].fillna(False).sum()) if "ball_detected" in frame_df.columns else 0
    if valid_detections < min_valid_detections:
        shot_result = {
            "shot_frame": None,
            "shot_time_sec": None,
            "shot_confidence": 0.0,
            "candidate_frames": [],
            "reason": "Not enough valid ball detections to estimate shot reliably.",
        }
        frame_df = _prepare_frame_flags(frame_df, shot_result)
        frame_df = add_event_inside_penalty_area_column(frame_df, penalty_area, [])
        status = "missing_pitch_config" if config is None else "low_ball_detection"
        _append_note(notes, shot_result["reason"])
        shot_row = _empty_shot_row(video_id, status, " ".join(notes).strip())
    else:
        shot_result = detect_shot_frame(frame_df, fps=fps)
        if shot_result.get("shot_frame") is None:
            frame_df = _prepare_frame_flags(frame_df, shot_result)
            frame_df = add_event_inside_penalty_area_column(frame_df, penalty_area, [])
            status = "missing_pitch_config" if config is None else "no_reliable_shot_detected"
            _append_note(notes, shot_result.get("reason"))
            shot_row = _empty_shot_row(video_id, status, " ".join(notes).strip())
        else:
            status = "missing_pitch_config" if config is None else "success"
            shot_frame = int(shot_result["shot_frame"])
            goal_entry = estimate_goal_entry_point(frame_df, shot_frame, goal_corners)
            play_end_event = estimate_play_end_event(
                frame_df,
                shot_frame,
                goal_entry,
                fps=fps,
                frame_width=int(frame_df.attrs.get("width") or 0) or None,
                frame_height=int(frame_df.attrs.get("height") or 0) or None,
            )
            frame_df = _prepare_frame_flags(frame_df, shot_result, play_end_event)
            frame_df = add_event_inside_penalty_area_column(
                frame_df,
                penalty_area,
                [shot_frame, play_end_event.get("play_end_frame")],
            )
            shot_row = build_shot_level_row(
                frame_df,
                shot_result,
                goal_corners,
                goal_entry,
                play_end_event,
                video_id,
                status,
                notes,
            )
            inside_flag = shot_row.get("inside_penalty_area")
            inside_known_false = isinstance(inside_flag, (bool, np.bool_)) and not bool(inside_flag)
            if only_inside_box and inside_known_false:
                shot_row["status"] = "filtered_outside_penalty_area"
                shot_row["notes"] = (str(shot_row.get("notes") or "") + " Shot outside penalty area and --only-inside-box was set.").strip()

    frame_df = ensure_columns(frame_df, FRAME_LEVEL_COLUMNS)
    shot_df = ensure_columns(pd.DataFrame([shot_row]), SHOT_LEVEL_COLUMNS)
    write_frame_level_csv(frame_df, frame_output)
    write_shot_level(shot_df, shot_output)

    shot_row_for_annotation = shot_df.iloc[0].to_dict()
    if annotated_video:
        annotate_video(video_path, frame_df, shot_row_for_annotation, config, annotated_video)
    if save_review_frames_dir:
        save_review_frames(video_path, frame_df, shot_row_for_annotation, config, save_review_frames_dir)

    return {"frame_df": frame_df, "shot_df": shot_df, "status": shot_row.get("status")}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect 1v1 football shots and goal-entry coordinates.")
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument("--model", default=None, help="YOLO ball detector model path.")
    parser.add_argument("--config", default=None, help="Pitch/goal configuration JSON path.")
    parser.add_argument("--frame-output", required=True, help="Frame-level CSV output path.")
    parser.add_argument("--shot-output", required=True, help="Shot-level CSV or JSON output path.")
    parser.add_argument("--only-inside-box", action="store_true", help="Mark shots outside the penalty area as filtered.")
    parser.add_argument("--annotated-video", default=None, help="Optional annotated MP4 output path.")
    parser.add_argument("--save-review-frames", default=None, help="Optional directory for review frames around the shot.")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bar.")
    parser.add_argument("--min-valid-detections", type=int, default=8, help="Minimum valid ball detections required.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = run_pipeline(
        video_path=args.video,
        model_path=args.model,
        config_path=args.config,
        frame_output=args.frame_output,
        shot_output=args.shot_output,
        only_inside_box=args.only_inside_box,
        annotated_video=args.annotated_video,
        save_review_frames_dir=args.save_review_frames,
        show_progress=not args.no_progress,
        min_valid_detections=args.min_valid_detections,
    )
    print(f"Pipeline finished with status: {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
