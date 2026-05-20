"""Validate the pitch configuration file used by the Stage 1 pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_GOAL_CORNERS = {"bottom_left", "bottom_right", "top_left", "top_right"}


def _is_point(value) -> bool:
    return isinstance(value, list) and len(value) == 2 and all(isinstance(item, (int, float)) for item in value)


def validate_config(config: dict) -> list[str]:
    errors: list[str] = []
    penalty_area = config.get("penalty_area_polygon")
    if penalty_area is not None:
        if not isinstance(penalty_area, list) or len(penalty_area) < 3 or not all(_is_point(point) for point in penalty_area):
            errors.append("penalty_area_polygon must contain at least three [x, y] points.")

    goal_corners = config.get("goal_corners")
    if goal_corners is not None:
        if not isinstance(goal_corners, dict):
            errors.append("goal_corners must be an object.")
        else:
            missing = REQUIRED_GOAL_CORNERS - set(goal_corners)
            if missing:
                errors.append(f"goal_corners missing: {', '.join(sorted(missing))}.")
            for key in REQUIRED_GOAL_CORNERS & set(goal_corners):
                if not _is_point(goal_corners[key]):
                    errors.append(f"goal_corners.{key} must be [x, y].")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate pitch_config JSON.")
    parser.add_argument("config", help="Path to pitch config JSON.")
    args = parser.parse_args(argv)

    path = Path(args.config)
    with path.open("r", encoding="utf-8") as file:
        config = json.load(file)
    errors = validate_config(config)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("Config looks valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

