"""Utilities to turn PFF FC World Cup 2022 data into save-probability rows.

The public PFF FC World Cup release is distributed after request/approval, so
the importer is intentionally tolerant of small schema differences. It expects
local files, extracts shot events, normalizes shot/keeper positions around the
defended goal, and produces the tabular contract consumed by
``scripts/train_save_probability_model.py``.
"""

from __future__ import annotations

import json
import math
import bz2
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from reach_probability import GOAL_HEIGHT_M, GOAL_WIDTH_M

PFF_PITCH_LENGTH_M = 105.0
PFF_PITCH_WIDTH_M = 68.0
PFF_HALF_LENGTH_M = PFF_PITCH_LENGTH_M / 2.0
PFF_HALF_WIDTH_M = PFF_PITCH_WIDTH_M / 2.0
DEFAULT_KEEPER_BODY = {
    "keeper_body_width": 0.74,
    "keeper_body_height": 1.55,
    "keeper_hand_span": 1.78,
    "keeper_foot_span": 0.70,
    "keeper_polygon_area_uv": 0.11,
}
SAVE_OUTCOMES = {
    "save",
    "saved",
    "saved to post",
    "saved off target",
    "keeper save",
    "saved_by_keeper",
    "blocked by goalkeeper",
}
GOAL_OUTCOMES = {"goal", "scored"}
PFF_SHOT_OUTCOME_ALIASES = {
    "S": "save",
    "F": "save_off_target",
    "G": "goal",
    "B": "blocked_on_target",
    "C": "blocked_off_target",
    "L": "goalline_clearance",
    "O": "off_target",
}
PFF_HEIGHT_MIDPOINTS = {
    "G": 0.08,
    "BOTTOMTHIRD": 0.18,
    "MIDDLETHIRD": 0.50,
    "TOPTHIRD": 0.82,
    "U": 0.93,
    "C": 1.02,
}
SAVE_OUTCOMES.add("save_off_target")


def load_pff_events(events_path: Path) -> list[dict[str, Any]]:
    """Load PFF events from a JSON list, mapping, or nested ``events`` key."""
    if events_path.is_dir():
        events: list[dict[str, Any]] = []
        for path in sorted(events_path.glob("*.json")):
            events.extend(load_pff_events(path))
        return events
    with events_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    return list(_iter_event_records(payload))


def _iter_event_records(payload: Any) -> Iterable[dict[str, Any]]:
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item
        return
    if not isinstance(payload, dict):
        return
    for key in ("events", "data", "records"):
        value = payload.get(key)
        if isinstance(value, list):
            yield from _iter_event_records(value)
            return
    for value in payload.values():
        if isinstance(value, dict):
            yield value
        elif isinstance(value, list):
            yield from _iter_event_records(value)


def build_pff_save_probability_rows(
    events: Iterable[dict[str, Any]],
    *,
    include_unlabeled: bool = False,
    keeper_default_u: float = 0.5,
    keeper_default_v: float = -0.45,
) -> pd.DataFrame:
    """Build a save-probability dataframe from PFF shot events."""
    rows = []
    for event in events:
        if not is_shot_event(event):
            continue
        row = event_to_save_probability_row(
            event,
            keeper_default_u=keeper_default_u,
            keeper_default_v=keeper_default_v,
        )
        if include_unlabeled or not np.isnan(row["save_label"]):
            rows.append(row)
    return pd.DataFrame(rows)


def hydrate_rows_with_pff_tracking(
    rows: pd.DataFrame,
    tracking_dir: Path,
    *,
    strict: bool = False,
) -> pd.DataFrame:
    """Replace keeper/ball fallbacks with positions from PFF tracking files.

    The function scans ``<game_id>.jsonl`` or ``<game_id>.jsonl.bz2`` files and
    matches rows by ``frame_id``. It supports nested tracking records with
    ``players``/``objects`` lists. If the schema differs, the original rows are
    preserved and remain marked as fallback unless ``strict=True``.
    """
    if rows.empty or not tracking_dir.exists():
        if strict and not tracking_dir.exists():
            raise FileNotFoundError(f"Tracking directory not found: {tracking_dir}")
        return rows.copy()

    output = rows.copy()
    output["tracking_used"] = False
    output["ball_source"] = output.get("ball_source", "event")
    for game_id, group in output.groupby("game_id", dropna=True):
        tracking_path = find_tracking_file(tracking_dir, str(game_id))
        if tracking_path is None:
            if strict:
                raise FileNotFoundError(f"Tracking file not found for game_id={game_id}")
            continue
        frame_to_indices: dict[str, list[int]] = {}
        for idx, frame_id in group["frame_id"].items():
            frame_key = _frame_key(frame_id)
            if frame_key is None:
                continue
            frame_to_indices.setdefault(frame_key, []).append(idx)
        if not frame_to_indices:
            continue
        for frame_key, frame in _iter_requested_tracking_frames(tracking_path, set(frame_to_indices)):
            for idx in frame_to_indices.get(frame_key, []):
                shot_team = output.at[idx, "team_id"] if "team_id" in output.columns else None
                goal_sign = float(output.at[idx, "attacking_goal_sign"]) if "attacking_goal_sign" in output.columns else 1.0
                keeper, ball = extract_keeper_and_ball_from_tracking_frame(frame, shot_team, goal_sign)
                if keeper is not None:
                    output.at[idx, "keeper_center_u"] = keeper[0]
                    output.at[idx, "keeper_center_v"] = keeper[1]
                    output.at[idx, "keeper_pose_confidence"] = 0.85
                    output.at[idx, "keeper_source"] = "tracking"
                    output.at[idx, "tracking_used"] = True
                    if output.at[idx, "notes"]:
                        output.at[idx, "notes"] = str(output.at[idx, "notes"]).replace(
                            "Keeper fallback used; replace with tracking-derived keeper position.",
                            "",
                        ).strip()
                if ball is not None:
                    output.at[idx, "ball_position_u"] = ball[0]
                    output.at[idx, "ball_position_v"] = ball[1]
                    output.at[idx, "ball_source"] = "tracking"
    return output


def find_tracking_file(tracking_dir: Path, game_id: str) -> Path | None:
    """Find a likely PFF tracking file for a game id."""
    candidates = [
        tracking_dir / f"{game_id}.jsonl.bz2",
        tracking_dir / f"{game_id}.jsonl",
        tracking_dir / f"{game_id}.json.bz2",
        tracking_dir / f"{game_id}.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = sorted(tracking_dir.glob(f"*{game_id}*.jsonl*")) + sorted(tracking_dir.glob(f"*{game_id}*.json*"))
    return matches[0] if matches else None


def is_shot_event(event: dict[str, Any]) -> bool:
    """Return True when an event appears to be a shot."""
    if _nested_get(event, "possessionEvents.possessionEventType") == "SH":
        return True
    values = [
        _nested_get(event, "event_type"),
        _nested_get(event, "eventType"),
        _nested_get(event, "type"),
        _nested_get(event, "name"),
        _nested_get(event, "event.name"),
        _nested_get(event, "type.name"),
        _nested_get(event, "action_type"),
    ]
    clean = " ".join(str(value).lower() for value in values if value is not None)
    if "shot" in clean or "penalty kick" in clean:
        return True
    return bool(_nested_get(event, "shot") or _nested_get(event, "is_shot"))


def event_to_save_probability_row(
    event: dict[str, Any],
    *,
    keeper_default_u: float = 0.5,
    keeper_default_v: float = -0.45,
) -> dict[str, Any]:
    """Convert a single shot event into the model's training-row contract."""
    start = _extract_pff_shooter_point(event)
    if start is None:
        start = _extract_point(event, prefixes=("start", "location", "coordinate", "coordinates"), allow_top_level=True)
    end = _extract_shot_end_point(event)
    if end is None:
        end = _extract_pff_ball_point(event)
    goal_sign = infer_attacking_goal_sign(event, start=start, end=end)

    ball_u, ball_v = normalize_field_point_to_goal_uv(start, goal_sign) if start else (np.nan, np.nan)
    target_u, target_v = normalize_goalmouth_point_to_goal_uv(end, goal_sign) if end else (np.nan, np.nan)
    target_v, target_v_source = _resolve_goal_entry_v(event, target_v)
    keeper = extract_keeper_from_event(event, goal_sign)
    keeper_source = "event_freeze_frame"
    keeper_confidence = 0.55
    if keeper is None:
        keeper_u, keeper_v = keeper_default_u, keeper_default_v
        keeper_source = "fallback_center"
        keeper_confidence = 0.2
    else:
        keeper_u, keeper_v = keeper

    distance_to_goal_m = _distance_to_goal_line(start, goal_sign) if start else np.nan
    shot_speed = _first_float(
        event,
        ["shot_speed", "shot.speed", "speed", "ball_speed", "ball.speed", "pff_shot_speed"],
    )
    if math.isnan(shot_speed):
        shot_speed = 24.0
        speed_source = "default_24_mps"
    else:
        speed_source = "event"
    time_to_goal = _first_float(event, ["time_to_goal", "shot.time_to_goal", "ball_flight_time"])
    if math.isnan(time_to_goal):
        duration = _first_float(event, ["duration"])
        if not math.isnan(duration) and 0.15 <= duration <= 2.5:
            time_to_goal = duration
            time_source = "event_duration"
        else:
            time_to_goal = float(np.clip(distance_to_goal_m / max(shot_speed, 1.0), 0.25, 1.6)) if not math.isnan(distance_to_goal_m) else np.nan
            time_source = "distance_over_speed"
    else:
        time_source = "event"

    outcome = normalize_pff_outcome(_extract_outcome(event))
    return {
        "data_source": "pff_fc_worldcup_2022",
        "game_id": _first_value(event, ["game_id", "gameId", "match_id", "matchId"]),
        "event_id": _first_value(event, ["event_id", "eventId", "id"]),
        "period_id": _first_value(event, ["period_id", "periodId", "period", "half"]),
        "frame_id": _first_value(event, ["frame_id", "frameId", "frame", "start_frame", "startFrame", "gameEventId"]),
        "timestamp": _first_value(event, ["timestamp", "time", "game_clock", "clock", "seconds"]),
        "team_id": _first_value(event, ["team_id", "teamId", "team.id", "possession_team_id", "gameEvents.teamId"]),
        "team_name": _first_value(event, ["team_name", "teamName", "gameEvents.teamName"]),
        "player_id": _first_value(event, ["player_id", "playerId", "player.id", "possessionEvents.shooterPlayerId", "gameEvents.playerId"]),
        "player_name": _first_value(event, ["player_name", "playerName", "player.name", "possessionEvents.shooterPlayerName", "gameEvents.playerName"]),
        "outcome": outcome,
        "save_label": outcome_to_save_label(outcome),
        "model_label_scope": "save_vs_goal" if outcome in SAVE_OUTCOMES.union(GOAL_OUTCOMES) else "unlabeled_non_goal_save",
        "goal_entry_u": target_u,
        "goal_entry_v": target_v,
        "ball_position_u": ball_u,
        "ball_position_v": ball_v,
        "keeper_center_u": keeper_u,
        "keeper_center_v": keeper_v,
        "keeper_pose_confidence": keeper_confidence,
        "keeper_source": keeper_source,
        "shot_ball_speed": shot_speed,
        "time_ball_to_goal": time_to_goal,
        "reaction_time": 0.32,
        "attacking_goal_sign": goal_sign,
        "shot_x_m": start[0] if start else np.nan,
        "shot_y_m": start[1] if start else np.nan,
        "shot_end_x_m": end[0] if end else np.nan,
        "shot_end_y_m": end[1] if end else np.nan,
        "shot_end_z_m": end[2] if end and len(end) > 2 else np.nan,
        "distance_to_goal_m": distance_to_goal_m,
        "shot_speed_source": speed_source,
        "time_to_goal_source": time_source,
        "goal_entry_v_source": target_v_source,
        "pff_shot_outcome_type": _nested_get(event, "possessionEvents.shotOutcomeType"),
        "pff_shot_initial_height_type": _nested_get(event, "possessionEvents.shotInitialHeightType"),
        "pff_save_height_type": _nested_get(event, "possessionEvents.saveHeightType"),
        "ball_visibility": _first_value(_first_ball_dict(event) or {}, ["visibility"]),
        "notes": (
            "Keeper fallback used; replace with tracking-derived keeper position."
            if keeper_source == "fallback_center"
            else ""
        ),
        **DEFAULT_KEEPER_BODY,
    }


def infer_attacking_goal_sign(
    event: dict[str, Any],
    *,
    start: tuple[float, ...] | None = None,
    end: tuple[float, ...] | None = None,
) -> float:
    """Infer whether the shot attacks the +x or -x goal in PFF coordinates."""
    direction = _first_value(
        event,
        ["attacking_direction", "attack_direction", "direction", "team_direction", "play_direction"],
    )
    if direction is not None:
        clean = str(direction).lower()
        if any(token in clean for token in ("right", "positive", "+x", "home_to_away")):
            return 1.0
        if any(token in clean for token in ("left", "negative", "-x", "away_to_home")):
            return -1.0
    if end and not math.isnan(end[0]) and abs(end[0]) > PFF_HALF_LENGTH_M * 0.6:
        return 1.0 if end[0] >= 0 else -1.0
    if start and not math.isnan(start[0]) and abs(start[0]) > PFF_HALF_LENGTH_M * 0.35:
        return 1.0 if start[0] >= 0 else -1.0
    pff_sign = _infer_pff_goal_sign(event)
    if pff_sign is not None:
        return pff_sign
    if start and not math.isnan(start[0]):
        return 1.0 if start[0] >= 0 else -1.0
    return 1.0


def normalize_field_point_to_goal_uv(point: tuple[float, ...], goal_sign: float) -> tuple[float, float]:
    """Map a PFF field coordinate to our goal-relative ``u`` and pitch-depth ``v``."""
    x, y = point[:2]
    oriented_x = x * goal_sign
    oriented_y = y * goal_sign
    u = (oriented_y + GOAL_WIDTH_M / 2.0) / GOAL_WIDTH_M
    distance_from_goal_line_m = PFF_HALF_LENGTH_M - oriented_x
    v = -distance_from_goal_line_m / GOAL_HEIGHT_M
    return float(u), float(v)


def normalize_goalmouth_point_to_goal_uv(point: tuple[float, ...], goal_sign: float) -> tuple[float, float]:
    """Map shot end/goalmouth coordinates to continuous goal-frame coordinates."""
    if len(point) >= 2:
        _, y = point[:2]
    else:
        return np.nan, np.nan
    z = point[2] if len(point) > 2 and not math.isnan(point[2]) else 0.0
    oriented_y = y * goal_sign
    u = (oriented_y + GOAL_WIDTH_M / 2.0) / GOAL_WIDTH_M
    v = z / GOAL_HEIGHT_M
    return float(u), float(v)


def extract_keeper_from_event(event: dict[str, Any], goal_sign: float) -> tuple[float, float] | None:
    """Extract goalkeeper position from embedded freeze-frame/player context when present."""
    pff_keeper = _extract_pff_keeper_point(event, goal_sign)
    if pff_keeper is not None:
        return pff_keeper

    candidates = []
    for key in ("freeze_frame", "freezeFrame", "players", "player_tracking", "tracking", "objects"):
        value = _nested_get(event, key)
        if isinstance(value, list):
            candidates.extend(item for item in value if isinstance(item, dict))
    if not candidates:
        return None

    keeper_candidates = [item for item in candidates if _looks_like_keeper(item)]
    if not keeper_candidates:
        shot_team = _first_value(event, ["team_id", "teamId", "team.id"])
        opponents = [
            item
            for item in candidates
            if _first_value(item, ["team_id", "teamId", "team.id"]) not in {None, shot_team}
            and _extract_point(item, prefixes=("location", "coordinate", "coordinates", "position"), allow_top_level=True)
        ]
        keeper_candidates = sorted(
            opponents,
            key=lambda item: _distance_to_goal_line(
                _extract_point(item, prefixes=("location", "coordinate", "coordinates", "position"), allow_top_level=True),
                goal_sign,
            ),
        )[:1]
    for item in keeper_candidates:
        point = _extract_point(item, prefixes=("location", "coordinate", "coordinates", "position"), allow_top_level=True)
        if point:
            return normalize_field_point_to_goal_uv(point, goal_sign)
    return None


def extract_keeper_and_ball_from_tracking_frame(
    frame: dict[str, Any],
    shot_team_id: Any,
    goal_sign: float,
) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    """Extract keeper and ball goal-relative coordinates from one tracking frame."""
    objects = _extract_tracking_objects(frame)
    keeper = None
    keeper_candidates = [item for item in objects if _looks_like_keeper(item)]
    if not keeper_candidates:
        opponents = [
            item
            for item in objects
            if not _same_team(_first_value(item, ["team_id", "teamId", "team.id", "team"]), shot_team_id)
            and _extract_point(item, prefixes=("location", "coordinate", "coordinates", "position"), allow_top_level=True)
        ]
        keeper_candidates = sorted(
            opponents,
            key=lambda item: _distance_to_goal_line(
                _extract_point(item, prefixes=("location", "coordinate", "coordinates", "position"), allow_top_level=True),
                goal_sign,
            ),
        )[:1]
    for item in keeper_candidates:
        point = _extract_point(item, prefixes=("location", "coordinate", "coordinates", "position"), allow_top_level=True)
        if point:
            keeper = normalize_field_point_to_goal_uv(point, goal_sign)
            break

    ball_point = _extract_point(frame, prefixes=("ball", "ball_position", "ballLocation"), allow_top_level=False)
    if ball_point is None:
        for item in objects:
            if _looks_like_ball(item):
                ball_point = _extract_point(item, prefixes=("location", "coordinate", "coordinates", "position"), allow_top_level=True)
                if ball_point:
                    break
    ball = normalize_field_point_to_goal_uv(ball_point, goal_sign) if ball_point else None
    return keeper, ball


def outcome_to_save_label(outcome: str | None) -> float:
    if outcome in SAVE_OUTCOMES:
        return 1.0
    if outcome in GOAL_OUTCOMES:
        return 0.0
    return np.nan


def normalize_pff_outcome(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, dict):
        value = _first_value(value, ["name", "label", "value", "outcome"])
        if value is None:
            return None
    clean = str(value).strip().lower().replace("_", " ")
    clean = " ".join(clean.split())
    aliases = {
        "saved": "save",
        "keeper saved": "save",
        "saved by keeper": "save",
        "save off target": "save_off_target",
        "blocked by gk": "blocked by goalkeeper",
        "score": "goal",
        "scored": "goal",
    }
    if clean.upper() in PFF_SHOT_OUTCOME_ALIASES:
        return PFF_SHOT_OUTCOME_ALIASES[clean.upper()]
    return aliases.get(clean, clean)


def _extract_outcome(event: dict[str, Any]) -> Any:
    return _first_value(
        event,
        [
            "possessionEvents.shotOutcomeType",
            "outcome",
            "result",
            "shot_outcome",
            "shot.outcome.name",
            "shot.outcome",
            "shot.result",
            "type.outcome",
        ],
    )


def _infer_pff_goal_sign(event: dict[str, Any]) -> float | None:
    home_start_left = _nested_get(event, "stadiumMetadata.homeTeamStartLeft")
    shot_home = _nested_get(event, "gameEvents.homeTeam")
    period = _nested_get(event, "gameEvents.period")
    if home_start_left is None or shot_home is None or period is None:
        return None
    try:
        odd_period = int(period) % 2 == 1
    except (TypeError, ValueError):
        odd_period = True
    home_attacks_positive = bool(home_start_left) if odd_period else not bool(home_start_left)
    shot_team_attacks_positive = home_attacks_positive if bool(shot_home) else not home_attacks_positive
    return 1.0 if shot_team_attacks_positive else -1.0


def _extract_pff_shooter_point(event: dict[str, Any]) -> tuple[float, ...] | None:
    player_id = _first_value(event, ["possessionEvents.shooterPlayerId", "gameEvents.playerId"])
    shot_home = _nested_get(event, "gameEvents.homeTeam")
    if player_id is None:
        return None
    preferred = "homePlayers" if shot_home is True else "awayPlayers" if shot_home is False else None
    lists = [preferred] if preferred else []
    lists.extend(key for key in ("homePlayers", "awayPlayers") if key not in lists)
    for key in lists:
        value = _nested_get(event, key)
        if not isinstance(value, list):
            continue
        for item in value:
            if isinstance(item, dict) and _same_team(_first_value(item, ["playerId", "player_id", "id"]), player_id):
                return _extract_point(item, prefixes=("location", "coordinate", "coordinates", "position"), allow_top_level=True)
    return None


def _extract_pff_keeper_point(event: dict[str, Any], goal_sign: float) -> tuple[float, float] | None:
    keeper_id = _nested_get(event, "possessionEvents.keeperPlayerId")
    shot_home = _nested_get(event, "gameEvents.homeTeam")
    opponent_key = "awayPlayers" if shot_home is True else "homePlayers" if shot_home is False else None
    candidate_lists = [opponent_key] if opponent_key else []
    candidate_lists.extend(key for key in ("homePlayers", "awayPlayers") if key not in candidate_lists)
    candidates: list[dict[str, Any]] = []
    for key in candidate_lists:
        value = _nested_get(event, key)
        if isinstance(value, list):
            candidates.extend(item for item in value if isinstance(item, dict))
    if not candidates:
        return None
    if keeper_id is not None:
        for item in candidates:
            if _same_team(_first_value(item, ["playerId", "player_id", "id"]), keeper_id):
                point = _extract_point(item, prefixes=("location", "coordinate", "coordinates", "position"), allow_top_level=True)
                return normalize_field_point_to_goal_uv(point, goal_sign) if point else None
    if opponent_key:
        opponent_players = [item for item in candidates if item in (_nested_get(event, opponent_key) or [])]
    else:
        opponent_players = candidates
    ranked = sorted(
        opponent_players,
        key=lambda item: _distance_to_goal_line(
            _extract_point(item, prefixes=("location", "coordinate", "coordinates", "position"), allow_top_level=True),
            goal_sign,
        ),
    )
    for item in ranked:
        point = _extract_point(item, prefixes=("location", "coordinate", "coordinates", "position"), allow_top_level=True)
        if point:
            return normalize_field_point_to_goal_uv(point, goal_sign)
    return None


def _extract_pff_ball_point(event: dict[str, Any]) -> tuple[float, ...] | None:
    return _coerce_point(_nested_get(event, "ball"))


def _first_ball_dict(event: dict[str, Any]) -> dict[str, Any] | None:
    value = _nested_get(event, "ball")
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                return item
    return None


def _resolve_goal_entry_v(event: dict[str, Any], target_v: float) -> tuple[float, str]:
    height_type = _first_value(
        event,
        [
            "possessionEvents.saveHeightType",
            "possessionEvents.shotInitialHeightType",
            "possessionEvents.ballHeightType",
        ],
    )
    if height_type is not None:
        clean = str(height_type).strip().upper()
        if clean in PFF_HEIGHT_MIDPOINTS:
            return PFF_HEIGHT_MIDPOINTS[clean], f"pff_height_type:{clean}"
    if not math.isnan(target_v):
        return target_v, "ball_z"
    return np.nan, "missing"


def _extract_shot_end_point(event: dict[str, Any]) -> tuple[float, ...] | None:
    point = _extract_point(
        event,
        prefixes=("end", "end_location", "endLocation", "shot_end", "goalmouth", "shot.goalmouth"),
        allow_top_level=False,
    )
    if point:
        return point
    shot = _nested_get(event, "shot")
    if isinstance(shot, dict):
        return _extract_point(
            shot,
            prefixes=("end", "end_location", "endLocation", "goalmouth", "goalmouth_location"),
            allow_top_level=False,
        )
    return None


def _extract_point(
    event: dict[str, Any],
    prefixes: tuple[str, ...],
    *,
    allow_top_level: bool = True,
) -> tuple[float, ...] | None:
    for prefix in prefixes:
        value = _nested_get(event, prefix)
        point = _coerce_point(value)
        if point:
            return point
        x = _nested_get(event, f"{prefix}_x")
        y = _nested_get(event, f"{prefix}_y")
        z = _nested_get(event, f"{prefix}_z")
        point = _coerce_point([x, y, z])
        if point:
            return point
        x = _nested_get(event, f"{prefix}.x")
        y = _nested_get(event, f"{prefix}.y")
        z = _nested_get(event, f"{prefix}.z")
        point = _coerce_point([x, y, z])
        if point:
            return point
    if allow_top_level:
        x = _first_value(event, ["x", "x_m"])
        y = _first_value(event, ["y", "y_m"])
        z = _first_value(event, ["z", "z_m"])
        return _coerce_point([x, y, z])
    return None


def _iter_requested_tracking_frames(tracking_path: Path, requested_frame_keys: set[str]) -> Iterable[tuple[str, dict[str, Any]]]:
    opener = bz2.open if tracking_path.suffix == ".bz2" else open
    with opener(tracking_path, "rt", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                frame = json.loads(line)
            except json.JSONDecodeError:
                continue
            frame_key = _frame_key(_first_value(frame, ["frame_id", "frameId", "frame", "id", "sample_id"]))
            if frame_key in requested_frame_keys:
                yield frame_key, frame


def _extract_tracking_objects(frame: dict[str, Any]) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    for key in ("players", "objects", "detections", "tracks", "home_players", "away_players"):
        value = _nested_get(frame, key)
        if isinstance(value, list):
            objects.extend(item for item in value if isinstance(item, dict))
        elif isinstance(value, dict):
            objects.extend(item for item in value.values() if isinstance(item, dict))
    for team_key in ("home", "away"):
        value = _nested_get(frame, team_key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    copy = dict(item)
                    copy.setdefault("team", team_key)
                    objects.append(copy)
        elif isinstance(value, dict):
            players = value.get("players") or value.get("objects")
            if isinstance(players, list):
                for item in players:
                    if isinstance(item, dict):
                        copy = dict(item)
                        copy.setdefault("team", value.get("team_id") or value.get("team") or team_key)
                        objects.append(copy)
    return objects


def _looks_like_ball(item: dict[str, Any]) -> bool:
    value = _first_value(item, ["object_type", "type", "class", "name", "label", "role"])
    return value is not None and "ball" in str(value).strip().lower()


def _coerce_point(value: Any) -> tuple[float, ...] | None:
    if isinstance(value, dict):
        raw = [value.get("x"), value.get("y"), value.get("z")]
    elif isinstance(value, (list, tuple)):
        if value and isinstance(value[0], dict):
            return _coerce_point(value[0])
        raw = list(value[:3])
    else:
        return None
    coords = []
    for item in raw:
        try:
            coords.append(float(item))
        except (TypeError, ValueError):
            coords.append(np.nan)
    if len(coords) < 2 or math.isnan(coords[0]) or math.isnan(coords[1]):
        return None
    while len(coords) < 3:
        coords.append(np.nan)
    return tuple(coords)


def _looks_like_keeper(item: dict[str, Any]) -> bool:
    values = [
        _first_value(item, ["position", "position_name", "player_position", "role", "position.name"]),
        _first_value(item, ["jersey_position", "pff_position"]),
    ]
    clean = " ".join(str(value).lower() for value in values if value is not None)
    return "goalkeeper" in clean or clean in {"gk", "keeper"}


def _distance_to_goal_line(point: tuple[float, ...] | None, goal_sign: float) -> float:
    if not point:
        return np.nan
    return float(PFF_HALF_LENGTH_M - point[0] * goal_sign)


def _frame_key(value: Any) -> str | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        clean = str(value).strip()
        return clean or None
    if not math.isfinite(number):
        return None
    if number.is_integer():
        return str(int(number))
    return f"{number:.3f}"


def _same_team(value_a: Any, value_b: Any) -> bool:
    if value_a is None or value_b is None:
        return False
    return str(value_a).strip().lower() == str(value_b).strip().lower()


def _first_float(event: dict[str, Any], paths: list[str]) -> float:
    value = _first_value(event, paths)
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _first_value(event: dict[str, Any], paths: list[str]) -> Any:
    for path in paths:
        value = _nested_get(event, path)
        if value is not None:
            return value
    return None


def _nested_get(mapping: dict[str, Any], path: str) -> Any:
    if path in mapping:
        return mapping[path]
    current: Any = mapping
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current
