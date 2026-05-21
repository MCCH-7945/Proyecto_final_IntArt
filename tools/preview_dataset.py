"""Create a visual grid of YOLO labels for manual QA."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
IMG_EXTS = {".jpg", ".jpeg", ".png"}


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def draw_labels(image_bgr: np.ndarray, label_path: Path) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    output = image_bgr.copy()
    if not label_path.exists():
        overlay = output.copy()
        cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 120), -1)
        output = cv2.addWeighted(output, 0.85, overlay, 0.15, 0)
        cv2.putText(output, "NO LABEL", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return output

    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls, cx, cy, box_w, box_h = int(parts[0]), *map(float, parts[1:])
        if cls != 0:
            continue
        x1 = int((cx - box_w / 2.0) * width)
        y1 = int((cy - box_h / 2.0) * height)
        x2 = int((cx + box_w / 2.0) * width)
        y2 = int((cy + box_h / 2.0) * height)
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(output, (int(cx * width), int(cy * height)), 3, (0, 255, 0), -1)
        cv2.putText(output, "ball", (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
    return output


def make_grid(images: list[np.ndarray], cols: int, cell_size: tuple[int, int]) -> np.ndarray:
    cell_w, cell_h = cell_size
    rows = (len(images) + cols - 1) // cols
    grid = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)
    for index, image in enumerate(images):
        row, col = divmod(index, cols)
        resized = cv2.resize(image, (cell_w, cell_h))
        grid[row * cell_h:(row + 1) * cell_h, col * cell_w:(col + 1) * cell_w] = resized
    return grid


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preview YOLO ball labels.")
    parser.add_argument("--data", type=Path, default=Path("data/ball_dataset/data.yaml"))
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--cell-w", type=int, default=320)
    parser.add_argument("--cell-h", type=int, default=180)
    parser.add_argument("--out", type=Path, default=Path("runs/ball_detector/dataset_preview.jpg"))
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    data_yaml = _resolve(args.data)
    if not data_yaml.exists():
        print(f"[ERROR] data.yaml not found: {data_yaml}")
        return 1
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    dataset_root = data_yaml.parent
    image_dir = dataset_root / cfg.get(args.split, f"images/{args.split}")
    label_dir = dataset_root / "labels" / Path(cfg.get(args.split, f"images/{args.split}")).name
    image_paths = sorted(path for path in image_dir.glob("*") if path.suffix.lower() in IMG_EXTS)
    if not image_paths:
        print(f"[ERROR] No images found in {image_dir}")
        return 1

    random.seed(args.seed)
    sample = random.sample(image_paths, min(args.n, len(image_paths)))
    annotated: list[np.ndarray] = []
    for image_path in sample:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        annotated.append(draw_labels(image, label_dir / f"{image_path.stem}.txt"))
    if not annotated:
        print("[ERROR] No images could be read.")
        return 1

    out = _resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    grid = make_grid(annotated, args.cols, (args.cell_w, args.cell_h))
    cv2.imwrite(str(out), grid)
    print(f"Preview saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

