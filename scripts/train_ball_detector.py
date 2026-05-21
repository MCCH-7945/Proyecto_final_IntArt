"""Train a YOLO detector specialized for soccer balls."""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model_reporting import build_detection_model_report, write_model_report_csv, write_model_report_json  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def check_dataset(data_yaml: Path) -> None:
    """Fail early when the YOLO dataset is missing or empty."""
    data_yaml = _resolve(data_yaml)
    if not data_yaml.exists():
        print(f"[ERROR] data.yaml not found: {data_yaml}")
        print("Run first: python scripts/convert_annotations.py")
        sys.exit(1)

    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    dataset_root = data_yaml.parent
    errors: list[str] = []
    for split in ("train", "val"):
        image_dir = dataset_root / cfg.get(split, f"images/{split}")
        label_dir = dataset_root / "labels" / Path(cfg.get(split, f"images/{split}")).name
        images = [path for path in image_dir.glob("*") if path.suffix.lower() in IMG_EXTS] if image_dir.exists() else []
        labels = list(label_dir.glob("*.txt")) if label_dir.exists() else []
        if not images:
            errors.append(f"No images in {image_dir}")
        if not labels:
            errors.append(f"No labels in {label_dir}")

    if errors:
        print("[ERROR] Dataset incomplete:")
        for error in errors:
            print(f"  - {error}")
        print("Expected flow:")
        print("  1. python scripts/extract_frames.py")
        print("  2. python scripts/select_frames.py")
        print("  3. Annotate externally with class ball")
        print("  4. python scripts/convert_annotations.py --format <fmt> ...")
        sys.exit(1)

    train_count = len([path for path in (dataset_root / cfg["train"]).glob("*") if path.suffix.lower() in IMG_EXTS])
    val_count = len([path for path in (dataset_root / cfg["val"]).glob("*") if path.suffix.lower() in IMG_EXTS])
    print(f"Train images: {train_count}")
    print(f"Val images: {val_count}")
    if train_count < 200:
        print(f"[WARN] Only {train_count} train images. For robustness, aim for 500+ ball boxes.")


def build_train_kwargs(args: argparse.Namespace) -> dict:
    """Build conservative Ultralytics training options."""
    train_kwargs = {
        "data": str(_resolve(args.data)),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": str(_resolve(args.project)),
        "name": args.name,
        "exist_ok": True,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "lrf": 0.01,
        "warmup_epochs": 3,
        "weight_decay": 0.0005,
        "patience": 30,
        "fliplr": 0.5,
        "flipud": 0.0,
        "degrees": 5.0,
        "translate": 0.05,
        "scale": 0.4,
        "shear": 2.0,
        "perspective": 0.0002,
        "hsv_h": 0.015,
        "hsv_s": 0.5,
        "hsv_v": 0.3,
        "mosaic": 1.0,
        "mixup": 0.05,
        "copy_paste": 0.0,
        "blur": 0.01,
        "median_blur": 0.01,
        "erasing": 0.1,
        "save": True,
        "save_period": 10,
        "seed": 42,
        "verbose": True,
    }
    try:
        from ultralytics.cfg import DEFAULT_CFG_DICT
    except Exception:
        return train_kwargs

    supported = set(DEFAULT_CFG_DICT)
    supported.update({"data", "epochs", "imgsz", "batch", "project", "name", "exist_ok"})
    filtered = {key: value for key, value in train_kwargs.items() if key in supported}
    dropped = sorted(set(train_kwargs) - set(filtered))
    if dropped:
        print(f"[INFO] Dropping unsupported Ultralytics args for this version: {', '.join(dropped)}")
    return filtered


def train(args: argparse.Namespace) -> None:
    from ultralytics import YOLO

    print("Starting ball-detector training")
    print(f"Base model: {args.base_model}")
    print(f"Data: {_resolve(args.data)}")
    print(f"imgsz: {args.imgsz}")
    print(f"batch: {args.batch}")
    model = YOLO(args.base_model)
    start = time.time()
    model.train(**build_train_kwargs(args))
    elapsed_min = (time.time() - start) / 60.0
    print(f"Training finished in {elapsed_min:.1f} minutes")


def find_best_weights(project: Path, name: str) -> Path:
    project = _resolve(project)
    preferred = project / name / "weights" / "best.pt"
    if preferred.exists():
        return preferred
    candidates = sorted(project.rglob("best.pt"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        print(f"[ERROR] best.pt not found under {project}")
        sys.exit(1)
    return candidates[-1]


def validate(model_path: Path, data_yaml: Path, imgsz: int, report_output: Path, report_csv: Path) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    conf = 0.25
    iou = 0.6
    metrics = model.val(data=str(_resolve(data_yaml)), imgsz=imgsz, conf=conf, iou=iou, verbose=True)
    box = metrics.box
    print("Final metrics")
    print(f"Precision: {box.mp:.4f}")
    print(f"Recall: {box.mr:.4f}")
    print(f"mAP50: {box.map50:.4f}")
    print(f"mAP50-95: {box.map:.4f}")
    if box.map50 < 0.50:
        print("[WARN] mAP50 < 0.50. Add more labeled examples or train longer.")
    if box.mp < 0.60:
        print("[WARN] Low precision. Review false positives: heads, boots, gloves, white marks.")
    if box.mr < 0.60:
        print("[WARN] Low recall. Add difficult ball examples: small, blurry, occluded.")
    report = build_detection_model_report(
        model_name=model_path.name,
        model_path=model_path,
        data_yaml=_resolve(data_yaml),
        split="val",
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        metrics_box=box,
    )
    write_model_report_json(report, _resolve(report_output))
    write_model_report_csv(report, _resolve(report_csv))
    print(f"Model report saved: {_resolve(report_output)}")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train YOLO ball detector.")
    parser.add_argument("--data", type=Path, default=Path("data/ball_dataset/data.yaml"))
    parser.add_argument("--base-model", type=str, default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--project", type=Path, default=Path("runs/ball_detector"))
    parser.add_argument("--name", type=str, default="train")
    parser.add_argument("--output", type=Path, default=Path("models/ball_detector/ball_model.pt"))
    parser.add_argument("--report-output", type=Path, default=Path("outputs/model_reports/ball_detector_metrics.json"))
    parser.add_argument("--report-csv", type=Path, default=Path("outputs/model_reports/ball_detector_metrics.csv"))
    parser.add_argument("--skip-val", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    check_dataset(args.data)
    train(args)
    best_pt = find_best_weights(args.project, args.name)
    output = _resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_pt, output)
    print(f"Model saved: {output}")
    if not args.skip_val:
        validate(output, args.data, args.imgsz, args.report_output, args.report_csv)
    print("Done. Use this model with --model models/ball_detector/ball_model.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
