"""Model metric report helpers.

Global model metrics belong in report files, while per-observation confidence
belongs in the frame-level or shot-level datasets.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_detection_model_report(
    model_name: str,
    model_path: str | Path,
    data_yaml: str | Path,
    split: str,
    imgsz: int,
    conf: float,
    iou: float,
    metrics_box: Any,
    task: str = "ball_detection",
) -> dict[str, Any]:
    """Build a serializable report from Ultralytics detection metrics."""
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": task,
        "model_name": model_name,
        "model_path": str(model_path),
        "data_yaml": str(data_yaml),
        "validation_split": split,
        "imgsz": int(imgsz),
        "conf_threshold": float(conf),
        "iou_threshold": float(iou),
        "metrics": {
            "precision": _as_float(getattr(metrics_box, "mp", None)),
            "recall": _as_float(getattr(metrics_box, "mr", None)),
            "map50": _as_float(getattr(metrics_box, "map50", None)),
            "map50_95": _as_float(getattr(metrics_box, "map", None)),
        },
    }


def write_model_report_json(report: dict[str, Any], output_path: str | Path) -> None:
    """Write a model report as JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def write_model_report_csv(report: dict[str, Any], output_path: str | Path) -> None:
    """Write a flattened one-row model report as CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = {
        "created_at_utc": report.get("created_at_utc"),
        "task": report.get("task"),
        "model_name": report.get("model_name"),
        "model_path": report.get("model_path"),
        "data_yaml": report.get("data_yaml"),
        "validation_split": report.get("validation_split"),
        "imgsz": report.get("imgsz"),
        "conf_threshold": report.get("conf_threshold"),
        "iou_threshold": report.get("iou_threshold"),
        "precision": report.get("metrics", {}).get("precision"),
        "recall": report.get("metrics", {}).get("recall"),
        "map50": report.get("metrics", {}).get("map50"),
        "map50_95": report.get("metrics", {}).get("map50_95"),
    }
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(flat))
        writer.writeheader()
        writer.writerow(flat)

