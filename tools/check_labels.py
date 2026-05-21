"""Diagnose YOLO ball labels."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
IMG_EXTS = {".jpg", ".jpeg", ".png"}


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def check_split(image_dir: Path, label_dir: Path, split: str) -> dict[str, int]:
    images = {path.stem: path for path in image_dir.glob("*") if path.suffix.lower() in IMG_EXTS}
    labels = {path.stem: path for path in label_dir.glob("*.txt")} if label_dir.exists() else {}
    errors: list[str] = []
    widths: list[float] = []
    heights: list[float] = []
    areas: list[float] = []
    total_boxes = 0

    for stem, label_path in sorted(labels.items()):
        if stem not in images:
            errors.append(f"[ORPHAN] label without image: {label_path.name}")
            continue
        for line_no, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
            clean = line.strip()
            if not clean:
                continue
            parts = clean.split()
            if len(parts) != 5:
                errors.append(f"[FORMAT] {label_path.name}:{line_no} expected 5 values, got {len(parts)}")
                continue
            try:
                cls = int(parts[0])
                cx, cy, width, height = map(float, parts[1:])
            except ValueError:
                errors.append(f"[FORMAT] {label_path.name}:{line_no} has non-numeric values")
                continue
            if cls != 0:
                errors.append(f"[CLASS] {label_path.name}:{line_no} class {cls}, expected 0")
            for name, value in (("cx", cx), ("cy", cy), ("w", width), ("h", height)):
                if not 0.0 <= value <= 1.0:
                    errors.append(f"[RANGE] {label_path.name}:{line_no} {name}={value:.4f} outside [0, 1]")
            widths.append(width)
            heights.append(height)
            areas.append(width * height)
            total_boxes += 1

    for stem in images:
        if stem not in labels:
            errors.append(f"[ORPHAN] image without label: {stem}")

    print(f"\nSplit: {split} | images: {len(images)} | labels: {len(labels)} | boxes: {total_boxes}")
    if errors:
        print(f"Problems found: {len(errors)}")
        for error in errors[:25]:
            print(f"  {error}")
        if len(errors) > 25:
            print(f"  ... {len(errors) - 25} more")
    else:
        print("No label-format errors found.")

    if total_boxes:
        width_arr = np.array(widths)
        height_arr = np.array(heights)
        area_arr = np.array(areas)
        print(f"BBox width: min {width_arr.min():.4f}, median {np.median(width_arr):.4f}, max {width_arr.max():.4f}")
        print(f"BBox height: min {height_arr.min():.4f}, median {np.median(height_arr):.4f}, max {height_arr.max():.4f}")
        print(f"BBox area: min {area_arr.min():.6f}, median {np.median(area_arr):.6f}, max {area_arr.max():.4f}")
        big = int((area_arr > 0.25).sum())
        tiny = int((area_arr < 0.0001).sum())
        if big:
            print(f"[WARN] {big} boxes cover more than 25% of the image.")
        if tiny:
            print(f"[INFO] {tiny} very tiny boxes. These may be distant balls.")

    return {"images": len(images), "labels": len(labels), "boxes": total_boxes, "errors": len(errors)}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check YOLO ball labels.")
    parser.add_argument("--data", type=Path, default=Path("data/ball_dataset/data.yaml"))
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    data_yaml = _resolve(args.data)
    if not data_yaml.exists():
        print(f"[ERROR] data.yaml not found: {data_yaml}")
        return 1
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    dataset_root = data_yaml.parent
    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    totals = {"images": 0, "labels": 0, "boxes": 0, "errors": 0}
    for split in splits:
        image_dir = dataset_root / cfg.get(split, f"images/{split}")
        label_dir = dataset_root / "labels" / Path(cfg.get(split, f"images/{split}")).name
        if not image_dir.exists():
            print(f"[SKIP] {split}: missing {image_dir}")
            continue
        result = check_split(image_dir, label_dir, split)
        for key in totals:
            totals[key] += result[key]

    print(f"\nTotal: images={totals['images']} boxes={totals['boxes']} errors={totals['errors']}")
    if totals["boxes"] < 500:
        print("[WARN] Fewer than 500 boxes. More labels will usually improve robustness.")
    return 1 if totals["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

