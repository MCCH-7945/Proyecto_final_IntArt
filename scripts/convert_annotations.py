"""Convert ball annotations from common tools into YOLO format."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "data" / "ball_dataset"
IMG_EXTS = {".jpg", ".jpeg", ".png"}
SPLITS = {"train": 0.75, "val": 0.15, "test": 0.10}


def ensure_dataset_dirs() -> None:
    for split in SPLITS:
        (DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
    data_yaml = DATASET_DIR / "data.yaml"
    if not data_yaml.exists():
        data_yaml.write_text(
            "path: data/ball_dataset\ntrain: images/train\nval: images/val\ntest: images/test\n\nnc: 1\nnames:\n  0: ball\n",
            encoding="utf-8",
        )


def split_files(paths: list[Path], seed: int = 42) -> dict[str, list[Path]]:
    random.seed(seed)
    shuffled = sorted(paths)
    random.shuffle(shuffled)
    total = len(shuffled)
    n_train = int(total * SPLITS["train"])
    n_val = int(total * SPLITS["val"])
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:],
    }


def xyxy_to_yolo(x1: float, y1: float, x2: float, y2: float, width: float, height: float) -> str:
    cx = ((x1 + x2) / 2.0) / width
    cy = ((y1 + y2) / 2.0) / height
    box_w = (x2 - x1) / width
    box_h = (y2 - y1) / height
    return f"0 {cx:.6f} {cy:.6f} {box_w:.6f} {box_h:.6f}"


def _find_image(images_dir: Path, name: str) -> Path | None:
    direct = images_dir / name
    if direct.exists():
        return direct
    matches = list(images_dir.rglob(name))
    return matches[0] if matches else None


def convert_cvat_xml(xml_path: Path, images_dir: Path) -> list[tuple[Path, list[str]]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    pairs: list[tuple[Path, list[str]]] = []
    for image_el in tqdm(root.findall("image"), desc="CVAT XML"):
        name = image_el.attrib["name"]
        width = float(image_el.attrib["width"])
        height = float(image_el.attrib["height"])
        image_path = _find_image(images_dir, name)
        if image_path is None:
            print(f"[WARN] Image not found: {name}")
            continue

        lines: list[str] = []
        for box in image_el.findall("box"):
            if box.attrib.get("label", "").lower() != "ball":
                continue
            lines.append(
                xyxy_to_yolo(
                    float(box.attrib["xtl"]),
                    float(box.attrib["ytl"]),
                    float(box.attrib["xbr"]),
                    float(box.attrib["ybr"]),
                    width,
                    height,
                )
            )
        if lines:
            pairs.append((image_path, lines))
    return pairs


def convert_labelstudio_json(json_path: Path, images_dir: Path) -> list[tuple[Path, list[str]]]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    pairs: list[tuple[Path, list[str]]] = []
    for task in tqdm(data, desc="Label Studio"):
        file_name = Path(task.get("data", {}).get("image", "")).name
        image_path = _find_image(images_dir, file_name)
        if image_path is None:
            print(f"[WARN] Image not found: {file_name}")
            continue

        lines: list[str] = []
        for annotation in task.get("annotations", []):
            for result in annotation.get("result", []):
                if result.get("type") != "rectanglelabels":
                    continue
                value = result.get("value", {})
                labels = [label.lower() for label in value.get("rectanglelabels", [])]
                if "ball" not in labels:
                    continue
                x = float(value["x"]) / 100.0
                y = float(value["y"]) / 100.0
                w = float(value["width"]) / 100.0
                h = float(value["height"]) / 100.0
                cx = x + w / 2.0
                cy = y + h / 2.0
                lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        if lines:
            pairs.append((image_path, lines))
    return pairs


def convert_roboflow_yolo(roboflow_dir: Path) -> list[tuple[Path, list[str]]]:
    pairs: list[tuple[Path, list[str]]] = []
    for split_name in ("train", "valid", "val", "test"):
        image_dir = roboflow_dir / split_name / "images"
        label_dir = roboflow_dir / split_name / "labels"
        if not image_dir.exists():
            continue
        for image_path in image_dir.iterdir():
            if image_path.suffix.lower() not in IMG_EXTS:
                continue
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue
            lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            ball_lines = [line for line in lines if line.startswith("0 ")]
            if ball_lines:
                pairs.append((image_path, ball_lines))
    return pairs


def convert_yolo_raw(images_dir: Path, labels_dir: Path) -> list[tuple[Path, list[str]]]:
    pairs: list[tuple[Path, list[str]]] = []
    for image_path in sorted(images_dir.iterdir()):
        if image_path.suffix.lower() not in IMG_EXTS:
            continue
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        ball_lines = [line for line in lines if line.startswith("0 ")]
        if ball_lines:
            pairs.append((image_path, ball_lines))
    return pairs


def write_dataset(pairs: list[tuple[Path, list[str]]], seed: int) -> None:
    ensure_dataset_dirs()
    image_paths = [image_path for image_path, _ in pairs]
    label_map = {image_path: label_lines for image_path, label_lines in pairs}
    splits = split_files(image_paths, seed=seed)

    for split_name, paths in splits.items():
        image_out_dir = DATASET_DIR / "images" / split_name
        label_out_dir = DATASET_DIR / "labels" / split_name
        image_out_dir.mkdir(parents=True, exist_ok=True)
        label_out_dir.mkdir(parents=True, exist_ok=True)
        for image_path in paths:
            shutil.copy2(image_path, image_out_dir / image_path.name)
            (label_out_dir / f"{image_path.stem}.txt").write_text(
                "\n".join(label_map[image_path]) + "\n",
                encoding="utf-8",
            )
        print(f"{split_name}: {len(paths)} images")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert annotations to YOLO format for the ball dataset.")
    parser.add_argument("--format", choices=["cvat_xml", "labelstudio", "roboflow_yolo", "yolo_raw"], required=True)
    parser.add_argument("--annotations", type=Path, help="Annotation file for CVAT XML or Label Studio JSON.")
    parser.add_argument("--images", type=Path, help="Directory with annotated images.")
    parser.add_argument("--labels", type=Path, help="Directory with raw YOLO labels.")
    parser.add_argument("--roboflow-dir", type=Path, help="Roboflow YOLO export directory.")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.format == "cvat_xml":
        if not args.annotations or not args.images:
            print("[ERROR] --annotations and --images are required for cvat_xml.")
            return 1
        pairs = convert_cvat_xml(args.annotations, args.images)
    elif args.format == "labelstudio":
        if not args.annotations or not args.images:
            print("[ERROR] --annotations and --images are required for labelstudio.")
            return 1
        pairs = convert_labelstudio_json(args.annotations, args.images)
    elif args.format == "roboflow_yolo":
        if not args.roboflow_dir:
            print("[ERROR] --roboflow-dir is required for roboflow_yolo.")
            return 1
        pairs = convert_roboflow_yolo(args.roboflow_dir)
    else:
        if not args.images or not args.labels:
            print("[ERROR] --images and --labels are required for yolo_raw.")
            return 1
        pairs = convert_yolo_raw(args.images, args.labels)

    if not pairs:
        print("[ERROR] No valid ball annotations found.")
        return 1

    print(f"Image-label pairs found: {len(pairs)}")
    write_dataset(pairs, seed=args.seed)
    print("Dataset ready: data/ball_dataset/")
    print("Next step: python scripts/train_ball_detector.py --data data/ball_dataset/data.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

