"""Select useful, diverse frames for annotation."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
IMG_EXTS = {".jpg", ".jpeg", ".png"}


def laplacian_variance(image_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def dhash(image_bgr: np.ndarray, size: int = 8) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (size + 1, size))
    return (small[:, 1:] > small[:, :-1]).flatten()


def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.sum(a != b))


def select_from_dir(
    src_dir: Path,
    out_dir: Path,
    blur_thresh: float,
    hash_thresh: int,
    remaining: int,
) -> int:
    images = sorted(path for path in src_dir.rglob("*") if path.suffix.lower() in IMG_EXTS)
    if not images or remaining <= 0:
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    selected: list[Path] = []
    last_hash: np.ndarray | None = None

    for image_path in tqdm(images, desc=src_dir.name, unit="img"):
        if len(selected) >= remaining:
            break
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        if laplacian_variance(image) < blur_thresh:
            continue
        image_hash = dhash(image)
        if last_hash is not None and hamming_distance(image_hash, last_hash) < hash_thresh:
            continue
        last_hash = image_hash
        selected.append(image_path)

    for image_path in selected:
        dest_name = f"{image_path.parent.name}_{image_path.name}"
        shutil.copy2(image_path, out_dir / dest_name)
    return len(selected)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select frames for ball annotation.")
    parser.add_argument("--src-dir", type=Path, default=ROOT / "data" / "raw_frames")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "frames_to_annotate")
    parser.add_argument("--blur-thresh", type=float, default=80.0)
    parser.add_argument("--hash-thresh", type=int, default=8)
    parser.add_argument("--max-total", type=int, default=1000)
    parser.add_argument("--clear-output", action="store_true", help="Delete selected-frame directory before copying.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if not args.src_dir.exists():
        print(f"[ERROR] Source directory not found: {args.src_dir}")
        print("Run first: python scripts/extract_frames.py")
        return 1

    if args.clear_output and args.out_dir.exists():
        shutil.rmtree(args.out_dir)

    subdirs = [path for path in args.src_dir.iterdir() if path.is_dir()]
    if not subdirs:
        subdirs = [args.src_dir]

    total = 0
    for subdir in sorted(subdirs):
        selected = select_from_dir(
            subdir,
            args.out_dir,
            args.blur_thresh,
            args.hash_thresh,
            args.max_total - total,
        )
        print(f"{subdir.name}: {selected} frames selected")
        total += selected
        if total >= args.max_total:
            break

    print(f"Total selected: {total}")
    print(f"Annotate frames in: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
