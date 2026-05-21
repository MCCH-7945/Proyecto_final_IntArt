"""Validate the trained ball detector and optionally write prediction previews."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model_reporting import build_detection_model_report, write_model_report_csv, write_model_report_json  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def run_metrics(model_path: Path, data_yaml: Path, imgsz: int, split: str, conf: float, iou: float) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    metrics = model.val(data=str(data_yaml), imgsz=imgsz, split=split, conf=conf, iou=iou, verbose=False)
    box = metrics.box
    print(f"Metrics on {split}")
    print(f"Precision: {box.mp:.4f}")
    print(f"Recall: {box.mr:.4f}")
    print(f"mAP50: {box.map50:.4f}")
    print(f"mAP50-95: {box.map:.4f}")
    return build_detection_model_report(
        model_name=model_path.name,
        model_path=model_path,
        data_yaml=data_yaml,
        split=split,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        metrics_box=box,
    )


def draw_predictions(model_path: Path, image_dir: Path, out_dir: Path, n: int, imgsz: int, conf: float) -> None:
    from ultralytics import YOLO

    images = sorted(path for path in image_dir.glob("*") if path.suffix.lower() in IMG_EXTS)[:n]
    if not images:
        print(f"[WARN] No images found in {image_dir}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(model_path))
    for image_path in images:
        result = model.predict(str(image_path), imgsz=imgsz, conf=conf, verbose=False)[0]
        cv2.imwrite(str(out_dir / image_path.name), result.plot())
    print(f"Previews saved to: {out_dir}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate ball_model.pt.")
    parser.add_argument("--model", type=Path, default=Path("models/ball_detector/ball_model.pt"))
    parser.add_argument("--data", type=Path, default=Path("data/ball_dataset/data.yaml"))
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.6)
    parser.add_argument("--preview", type=int, default=20)
    parser.add_argument("--preview-dir", type=Path, default=Path("runs/ball_detector/preview"))
    parser.add_argument("--report-output", type=Path, default=Path("outputs/model_reports/ball_detector_metrics.json"))
    parser.add_argument("--report-csv", type=Path, default=Path("outputs/model_reports/ball_detector_metrics.csv"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    model_path = _resolve(args.model)
    data_yaml = _resolve(args.data)
    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        return 1
    if not data_yaml.exists():
        print(f"[ERROR] data.yaml not found: {data_yaml}")
        return 1

    report = run_metrics(model_path, data_yaml, args.imgsz, args.split, args.conf, args.iou)
    write_model_report_json(report, _resolve(args.report_output))
    write_model_report_csv(report, _resolve(args.report_csv))
    print(f"Model report saved: {_resolve(args.report_output)}")
    if args.preview > 0:
        cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
        dataset_root = data_yaml.parent
        image_dir = dataset_root / cfg.get(args.split, f"images/{args.split}")
        draw_predictions(model_path, image_dir, _resolve(args.preview_dir), args.preview, args.imgsz, args.conf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
