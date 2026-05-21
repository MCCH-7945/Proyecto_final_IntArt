"""Lightweight setup checker for the project.

This script avoids importing heavy project modules so it can run before
dependencies are installed.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


REQUIRED_IMPORTS = {
    "cv2": "opencv-python",
    "ultralytics": "ultralytics",
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "shapely": "shapely",
    "matplotlib": "matplotlib",
    "tqdm": "tqdm",
    "jsonschema": "jsonschema",
    "yt_dlp": "yt-dlp",
    "yaml": "pyyaml",
    "PIL": "Pillow",
    "seaborn": "seaborn",
    "sklearn": "scikit-learn",
    "pytest": "pytest",
}


EXPECTED_PATHS = [
    "configs/pitch_config_example.json",
    "data/raw_videos",
    "data/raw_videos/raw_videos.txt",
    "data/ball_dataset/data.yaml",
    "data/ball_dataset/images/train",
    "data/ball_dataset/images/val",
    "data/ball_dataset/images/test",
    "data/ball_dataset/labels/train",
    "data/ball_dataset/labels/val",
    "data/ball_dataset/labels/test",
    "models/ball_detector",
    "outputs/frame_level",
    "outputs/shot_level",
    "outputs/annotated_videos",
    "outputs/review_frames",
    "outputs/model_reports",
    "src/shot_detector.py",
    "scripts/extract_frames.py",
    "scripts/select_frames.py",
    "scripts/convert_annotations.py",
    "scripts/train_ball_detector.py",
    "scripts/validate_model.py",
    "tools/check_labels.py",
    "tools/preview_dataset.py",
    "tools/run_inference.py",
]


def check_python_version() -> bool:
    version = sys.version_info
    print(f"Python: {version.major}.{version.minor}.{version.micro}")
    if version < (3, 10):
        print("  FAIL: Python 3.10+ is required.")
        return False
    if version >= (3, 13):
        print("  WARN: Python 3.11 or 3.12 is recommended for OpenCV/Ultralytics/Torch.")
        return False
    print("  OK")
    return True


def check_paths() -> bool:
    ok = True
    print("\nProject paths:")
    for relative in EXPECTED_PATHS:
        path = ROOT / relative
        exists = path.exists()
        print(f"  {'OK  ' if exists else 'MISS'} {relative}")
        ok = ok and exists
    return ok


def check_imports() -> bool:
    ok = True
    print("\nPython packages:")
    for module_name, package_name in REQUIRED_IMPORTS.items():
        found = importlib.util.find_spec(module_name) is not None
        print(f"  {'OK  ' if found else 'MISS'} {package_name}")
        ok = ok and found
    return ok


def check_url_list() -> bool:
    url_list = ROOT / "data" / "raw_videos" / "raw_videos.txt"
    if not url_list.exists():
        return False
    urls = [
        line.strip()
        for line in url_list.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    print(f"\nURL list: {len(urls)} URL(s) in {url_list.relative_to(ROOT)}")
    return bool(urls)


def check_model() -> bool:
    model_path = ROOT / "models" / "ball_detector" / "ball_model.pt"
    exists = model_path.exists()
    print(f"\nBall model: {'OK' if exists else 'MISSING'} {model_path.relative_to(ROOT)}")
    if not exists:
        print("  This is expected before training. Train with scripts/train_ball_detector.py.")
    return exists


def main() -> int:
    print(f"Project root: {ROOT}")
    python_ok = check_python_version()
    paths_ok = check_paths()
    imports_ok = check_imports()
    urls_ok = check_url_list()
    model_ok = check_model()

    print("\nSummary:")
    print(f"  Python version suitable: {python_ok}")
    print(f"  Required paths present : {paths_ok}")
    print(f"  Dependencies installed : {imports_ok}")
    print(f"  URL list has entries   : {urls_ok}")
    print(f"  Ball model exists      : {model_ok}")

    if not python_ok or not paths_ok or not imports_ok:
        print("\nNext setup command:")
        print("  python3.11 -m venv .venv")
        print("  source .venv/bin/activate")
        print("  python -m pip install --upgrade pip")
        print("  pip install -r requirements.txt")
        return 1

    if not model_ok:
        print("\nCode and environment look ready. The next missing artifact is the trained model.")
        return 2

    print("\nEverything needed to run the main pipeline is present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
