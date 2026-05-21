"""Create a goalkeeper reach/polygon demo from one video frame.

This is intentionally a proxy demo. If a SoccerNet/MOT-style tracking file is
available, it uses the goalkeeper bounding box. If not, a manual bbox or a
goal-based fallback is enough to draw the idea for a presentation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from goalkeeper_pose import add_goal_relative_pose_coordinates, add_keeper_pose_quality  # noqa: E402
from keeper_polygon import POLYGON_ORDER, add_keeper_polygon_features  # noqa: E402
from io_utils import load_pitch_config  # noqa: E402


GOAL_CORNER_ORDER = ["bottom_left", "bottom_right", "top_right", "top_left"]


def _resolve(path: Path | None) -> Path | None:
    if path is None:
        return None
    return path if path.is_absolute() else ROOT / path


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _first_existing(columns: list[str], candidates: list[str]) -> str | None:
    lower_map = {str(column).lower(): column for column in columns}
    for candidate in candidates:
        if candidate in lower_map:
            return lower_map[candidate]
    return None


def _read_table(path: Path) -> pd.DataFrame:
    first_line = ""
    with path.open("r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            if line.strip():
                first_line = line.strip()
                break

    has_header = any(char.isalpha() for char in first_line)
    if has_header:
        return pd.read_csv(path, sep=None, engine="python")

    df = pd.read_csv(path, header=None, sep=r"[,\s]+", engine="python")
    base_columns = ["frame", "track_id", "x", "y", "w", "h", "score", "class", "visibility"]
    df.columns = base_columns[: len(df.columns)]
    return df


def _standardize_tracking(df: pd.DataFrame) -> pd.DataFrame:
    frame_col = _first_existing(list(df.columns), ["frame", "frame_id", "frame_number", "image_id"])
    if frame_col is None:
        raise ValueError("Tracking file needs a frame column.")

    x_col = _first_existing(list(df.columns), ["x", "bb_left", "left", "bbox_x", "x1", "xmin"])
    y_col = _first_existing(list(df.columns), ["y", "bb_top", "top", "bbox_y", "y1", "ymin"])
    w_col = _first_existing(list(df.columns), ["w", "width", "bb_width", "bbox_w"])
    h_col = _first_existing(list(df.columns), ["h", "height", "bb_height", "bbox_h"])
    x2_col = _first_existing(list(df.columns), ["x2", "xmax", "right"])
    y2_col = _first_existing(list(df.columns), ["y2", "ymax", "bottom"])
    class_col = _first_existing(list(df.columns), ["class", "label", "role", "category", "object_class", "team"])
    score_col = _first_existing(list(df.columns), ["score", "conf", "confidence", "visibility"])

    if x_col is None or y_col is None:
        raise ValueError("Tracking file needs bbox x/y columns.")

    output = pd.DataFrame()
    output["frame"] = pd.to_numeric(df[frame_col], errors="coerce")
    output["x"] = pd.to_numeric(df[x_col], errors="coerce")
    output["y"] = pd.to_numeric(df[y_col], errors="coerce")
    if w_col and h_col:
        output["w"] = pd.to_numeric(df[w_col], errors="coerce")
        output["h"] = pd.to_numeric(df[h_col], errors="coerce")
    elif x2_col and y2_col:
        x2 = pd.to_numeric(df[x2_col], errors="coerce")
        y2 = pd.to_numeric(df[y2_col], errors="coerce")
        output["w"] = x2 - output["x"]
        output["h"] = y2 - output["y"]
    else:
        raise ValueError("Tracking file needs either w/h or x2/y2 bbox columns.")

    output["class"] = df[class_col].astype(str) if class_col else ""
    output["score"] = pd.to_numeric(df[score_col], errors="coerce") if score_col else np.nan
    output = output.replace([np.inf, -np.inf], np.nan).dropna(subset=["frame", "x", "y", "w", "h"])
    output = output[(output["w"] > 0) & (output["h"] > 0)]
    output["frame"] = output["frame"].astype(int)
    return output


def _load_tracking_bbox(
    tracking_file: Path,
    frame: int,
    class_filter: str,
    goalkeeper_class_id: int | None,
    goal_center: tuple[float, float] | None,
) -> tuple[dict[str, float], dict[str, Any]]:
    tracking = _standardize_tracking(_read_table(tracking_file))
    if tracking.empty:
        raise ValueError("Tracking file has no usable boxes.")

    frames = np.asarray(sorted(tracking["frame"].unique()), dtype=int)
    nearest_frame = int(frames[np.argmin(np.abs(frames - frame))])
    local = tracking[tracking["frame"] == nearest_frame].copy()

    filtered = local
    if goalkeeper_class_id is not None:
        numeric_class = pd.to_numeric(local["class"], errors="coerce")
        filtered = local[numeric_class == goalkeeper_class_id]
    elif class_filter:
        mask = local["class"].str.lower().str.contains(class_filter.lower(), na=False)
        if mask.any():
            filtered = local[mask]

    if filtered.empty:
        filtered = local

    filtered = filtered.copy()
    filtered["area"] = filtered["w"] * filtered["h"]
    if goal_center is not None:
        cx = filtered["x"] + filtered["w"] / 2.0
        cy = filtered["y"] + filtered["h"] / 2.0
        filtered["goal_distance"] = np.hypot(cx - goal_center[0], cy - goal_center[1])
        selected = filtered.sort_values(["goal_distance", "area"], ascending=[True, False]).iloc[0]
    else:
        selected = filtered.sort_values("area", ascending=False).iloc[0]

    bbox = {
        "x": float(selected["x"]),
        "y": float(selected["y"]),
        "w": float(selected["w"]),
        "h": float(selected["h"]),
    }
    meta = {
        "source": "tracking_file",
        "tracking_frame": nearest_frame,
        "tracking_class": str(selected.get("class", "")),
        "tracking_score": float(selected["score"]) if np.isfinite(selected.get("score", np.nan)) else np.nan,
    }
    return bbox, meta


def _goal_center(config: dict[str, Any] | None) -> tuple[float, float] | None:
    goal_corners = (config or {}).get("goal_corners")
    if not goal_corners:
        return None
    points = [goal_corners.get(key) for key in GOAL_CORNER_ORDER if goal_corners.get(key) is not None]
    if len(points) < 4:
        return None
    arr = np.asarray(points, dtype=float)
    return float(arr[:, 0].mean()), float(arr[:, 1].mean())


def _fallback_bbox(frame_shape: tuple[int, int, int], config: dict[str, Any] | None) -> tuple[dict[str, float], dict[str, Any]]:
    height, width = frame_shape[:2]
    goal_corners = (config or {}).get("goal_corners")
    if goal_corners and all(goal_corners.get(key) is not None for key in GOAL_CORNER_ORDER):
        arr = np.asarray([goal_corners[key] for key in GOAL_CORNER_ORDER], dtype=float)
        goal_w = float(arr[:, 0].max() - arr[:, 0].min())
        goal_h = float(arr[:, 1].max() - arr[:, 1].min())
        cx = float(arr[:, 0].mean())
        bottom = float(arr[:, 1].max())
        bbox_h = max(80.0, goal_h * 0.95)
        bbox_w = max(38.0, goal_w * 0.22)
        return (
            {"x": cx - bbox_w / 2.0, "y": bottom - bbox_h, "w": bbox_w, "h": bbox_h},
            {"source": "goal_fallback", "tracking_frame": np.nan, "tracking_class": "", "tracking_score": np.nan},
        )

    bbox_w = width * 0.08
    bbox_h = height * 0.28
    return (
        {"x": width * 0.5 - bbox_w / 2.0, "y": height * 0.55 - bbox_h / 2.0, "w": bbox_w, "h": bbox_h},
        {"source": "frame_center_fallback", "tracking_frame": np.nan, "tracking_class": "", "tracking_score": np.nan},
    )


def _parse_bbox(values: list[float] | None) -> dict[str, float] | None:
    if values is None:
        return None
    if len(values) != 4:
        raise ValueError("--keeper-bbox expects exactly four numbers: x y w h.")
    x, y, w, h = [float(value) for value in values]
    if w <= 0 or h <= 0:
        raise ValueError("--keeper-bbox width and height must be positive.")
    return {"x": x, "y": y, "w": w, "h": h}


def _bbox_to_proxy_pose(
    bbox: dict[str, float],
    video_id: str,
    frame: int,
    time_sec: float,
    confidence: float,
) -> pd.DataFrame:
    x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
    cx = x + w / 2.0
    points = {
        "head": (cx, y + 0.08 * h),
        "left_hand": (x + 0.05 * w, y + 0.45 * h),
        "right_hand": (x + 0.95 * w, y + 0.45 * h),
        "left_foot": (x + 0.28 * w, y + h),
        "right_foot": (x + 0.72 * w, y + h),
    }
    row: dict[str, Any] = {
        "video_id": video_id,
        "frame": frame,
        "time_sec": time_sec,
        "shot_frame": frame,
        "frame_relative_to_shot": 0,
        "play_end_frame": np.nan,
        "is_shot_frame": True,
        "is_play_end_frame": False,
        "keeper_detected": True,
        "keeper_confidence": confidence,
    }
    point_conf = float(np.clip(confidence * 0.85, 0.0, 1.0))
    for point, (px, py) in points.items():
        row[f"{point}_x"] = px
        row[f"{point}_y"] = py
        row[f"{point}_confidence"] = point_conf
    return pd.DataFrame([row])


def _load_frame(video_path: Path, frame_index: int) -> tuple[np.ndarray, float, int, int]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_index < 0:
        frame_index = max(0, total_frames // 2)
    if total_frames:
        frame_index = min(frame_index, total_frames - 1)

    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_index} from {video_path}")
    return frame, fps, total_frames, frame_index


def _shot_frame_from_csv(path: Path | None) -> int | None:
    if path is None or not path.exists():
        return None
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            data = data[0] if data else {}
        value = data.get("shot_frame") if isinstance(data, dict) else None
    else:
        df = pd.read_csv(path)
        value = df.iloc[0].get("shot_frame") if not df.empty else None
    try:
        return int(value) if np.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def _draw_transparent_polygon(frame: np.ndarray, points: np.ndarray, color: tuple[int, int, int], alpha: float) -> None:
    overlay = frame.copy()
    cv2.fillPoly(overlay, [points.astype(np.int32)], color)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, dst=frame)
    cv2.polylines(frame, [points.astype(np.int32)], isClosed=True, color=color, thickness=2)


def _draw_goal(frame: np.ndarray, config: dict[str, Any] | None) -> None:
    goal_corners = (config or {}).get("goal_corners")
    if not goal_corners or not all(goal_corners.get(key) is not None for key in GOAL_CORNER_ORDER):
        return
    points = np.asarray([goal_corners[key] for key in GOAL_CORNER_ORDER], dtype=np.int32)
    cv2.polylines(frame, [points], isClosed=True, color=(255, 140, 0), thickness=2)


def _draw_demo(
    frame: np.ndarray,
    bbox: dict[str, float],
    pose_df: pd.DataFrame,
    reach_radius: float,
    proxy_mode: str,
    title: str,
) -> np.ndarray:
    output = frame.copy()
    x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
    cx = int(round(x + w / 2.0))
    cy = int(round(y + h / 2.0))
    p1 = (int(round(x)), int(round(y)))
    p2 = (int(round(x + w)), int(round(y + h)))
    cv2.rectangle(output, p1, p2, (0, 255, 255), 2)

    if proxy_mode == "ellipse":
        axes = (int(round(max(8.0, w * 0.80))), int(round(max(8.0, h * 0.55))))
        cv2.ellipse(output, (cx, cy), axes, 0, 0, 360, (0, 220, 255), 2)
    else:
        cv2.circle(output, (cx, cy), int(round(reach_radius)), (0, 220, 255), 2)

    polygon_points: list[tuple[float, float]] = []
    row = pose_df.iloc[0]
    for point in POLYGON_ORDER:
        polygon_points.append((float(row[f"{point}_x"]), float(row[f"{point}_y"])))
    polygon = np.asarray(polygon_points, dtype=float)
    _draw_transparent_polygon(output, polygon, (60, 160, 255), 0.22)

    labels = {
        "left_hand": "LH",
        "head": "H",
        "right_hand": "RH",
        "right_foot": "RF",
        "left_foot": "LF",
    }
    for point, (px, py) in zip(POLYGON_ORDER, polygon_points):
        cv2.circle(output, (int(round(px)), int(round(py))), 5, (0, 80, 255), -1)
        cv2.putText(output, labels[point], (int(round(px)) + 6, int(round(py)) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 80, 255), 1)

    cv2.putText(output, title, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 3)
    cv2.putText(output, title, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
    cv2.putText(output, "proxy keeper polygon + reach circle", (15, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 3)
    cv2.putText(output, "proxy keeper polygon + reach circle", (15, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return output


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Demo a goalkeeper proxy polygon/circle from a video frame.")
    parser.add_argument("--video", type=Path, required=True, help="Local video path.")
    parser.add_argument("--config", type=Path, default=None, help="Optional pitch/goal config calibrated for this video.")
    parser.add_argument("--shot-output", type=Path, default=None, help="Optional shot-level CSV/JSON to reuse shot_frame.")
    parser.add_argument("--frame", type=int, default=None, help="Frame to annotate. Defaults to shot-output frame or video midpoint.")
    parser.add_argument("--tracking-file", type=Path, default=None, help="Optional SoccerNet/MOT-style tracking CSV/TXT.")
    parser.add_argument("--class-filter", default="goalkeeper", help="Text class filter used when tracking has class labels.")
    parser.add_argument("--goalkeeper-class-id", type=int, default=None, help="Numeric class id for goalkeeper if tracking uses ids.")
    parser.add_argument("--keeper-bbox", nargs=4, type=float, default=None, metavar=("X", "Y", "W", "H"), help="Manual goalkeeper bbox.")
    parser.add_argument("--proxy-mode", choices=["circle", "ellipse"], default="circle", help="Reach proxy to draw.")
    parser.add_argument("--radius-scale", type=float, default=0.62, help="Circle radius as scale of max bbox side.")
    parser.add_argument("--out-image", type=Path, default=Path("outputs/review_frames/keeper_polygon_demo.jpg"))
    parser.add_argument("--out-csv", type=Path, default=Path("outputs/keeper_polygon_demo.csv"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    video_path = _resolve(args.video)
    config_path = _resolve(args.config)
    shot_output = _resolve(args.shot_output)
    tracking_file = _resolve(args.tracking_file)
    out_image = _resolve(args.out_image)
    out_csv = _resolve(args.out_csv)
    assert video_path is not None
    assert out_image is not None
    assert out_csv is not None

    config, config_note = load_pitch_config(config_path)
    requested_frame = args.frame
    if requested_frame is None:
        requested_frame = _shot_frame_from_csv(shot_output)
    if requested_frame is None:
        requested_frame = -1

    frame, fps, total_frames, frame_index = _load_frame(video_path, requested_frame)
    goal_center = _goal_center(config)
    manual_bbox = _parse_bbox(args.keeper_bbox)

    if manual_bbox is not None:
        bbox = manual_bbox
        meta = {"source": "manual_bbox", "tracking_frame": np.nan, "tracking_class": "", "tracking_score": np.nan}
    elif tracking_file is not None and tracking_file.exists():
        bbox, meta = _load_tracking_bbox(
            tracking_file,
            frame_index,
            args.class_filter,
            args.goalkeeper_class_id,
            goal_center,
        )
    else:
        bbox, meta = _fallback_bbox(frame.shape, config)

    raw_confidence = meta.get("tracking_score")
    confidence = float(raw_confidence) if raw_confidence is not None and np.isfinite(raw_confidence) else 0.65
    confidence = float(np.clip(confidence, 0.05, 1.0))
    video_id = video_path.stem
    pose_df = _bbox_to_proxy_pose(bbox, video_id, frame_index, frame_index / fps if fps > 0 else np.nan, confidence)
    pose_df = add_goal_relative_pose_coordinates(pose_df, (config or {}).get("goal_corners"))
    pose_df = add_keeper_pose_quality(pose_df)
    pose_df = add_keeper_polygon_features(pose_df)

    reach_radius = max(float(bbox["w"]), float(bbox["h"])) * float(args.radius_scale)
    reach_area = float(np.pi * reach_radius * reach_radius)
    for key, value in {
        "demo_proxy_type": args.proxy_mode,
        "keeper_source": meta.get("source"),
        "tracking_frame": meta.get("tracking_frame"),
        "tracking_class": meta.get("tracking_class"),
        "bbox_x": bbox["x"],
        "bbox_y": bbox["y"],
        "bbox_w": bbox["w"],
        "bbox_h": bbox["h"],
        "reach_center_x": bbox["x"] + bbox["w"] / 2.0,
        "reach_center_y": bbox["y"] + bbox["h"] / 2.0,
        "reach_radius_px": reach_radius,
        "reach_area_px": reach_area,
        "demo_note": config_note or "Proxy demo only; not a trained keeper pose model.",
    }.items():
        pose_df[key] = value

    annotated = frame.copy()
    _draw_goal(annotated, config)
    annotated = _draw_demo(
        annotated,
        bbox,
        pose_df,
        reach_radius,
        args.proxy_mode,
        title=f"{video_id[:32]} frame {frame_index}",
    )

    _ensure_parent(out_image)
    _ensure_parent(out_csv)
    cv2.imwrite(str(out_image), annotated)
    pose_df.to_csv(out_csv, index=False)

    print(f"Demo image: {out_image}")
    print(f"Demo CSV: {out_csv}")
    print(f"Keeper source: {meta.get('source')}")
    print(f"Frame: {frame_index} / {total_frames if total_frames else 'unknown'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
