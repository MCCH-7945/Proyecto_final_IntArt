"""Run the trained ball detector over a video or video URL."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from video_source import download_video_url, is_url  # noqa: E402


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def run_video(model_path: Path, video_path: Path, out_path: Path, conf: float, imgsz: int, show: bool) -> None:
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    detections_per_frame: list[int] = []
    frame_idx = 0
    with tqdm(total=total if total else None, desc="Inference", unit="fr") as progress:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            result = model.predict(frame, imgsz=imgsz, conf=conf, verbose=False)[0]
            annotated = result.plot()
            n_detections = len(result.boxes)
            detections_per_frame.append(n_detections)
            color = (0, 255, 0) if n_detections else (0, 0, 255)
            cv2.putText(annotated, f"Ball detections: {n_detections}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            writer.write(annotated)
            if show:
                cv2.imshow("Ball detector", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            frame_idx += 1
            progress.update(1)

    capture.release()
    writer.release()
    if show:
        cv2.destroyAllWindows()

    detected_frames = sum(1 for count in detections_per_frame if count > 0)
    avg_detections = sum(detections_per_frame) / max(frame_idx, 1)
    print(f"Video saved: {out_path}")
    print(f"Frames: {frame_idx}")
    print(f"Frames with ball: {detected_frames} ({100 * detected_frames / max(frame_idx, 1):.1f}%)")
    print(f"Average detections/frame: {avg_detections:.2f}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ball detector inference on a video.")
    parser.add_argument("--model", type=Path, default=Path("models/ball_detector/ball_model.pt"))
    parser.add_argument("--video", required=True, help="Local video path or HTTP(S) URL.")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--url-cache-dir", type=Path, default=Path("data/raw_videos/url_cache"))
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument(
        "--cookies-from-browser",
        default=None,
        help="Load cookies from a browser for yt-dlp, e.g. chrome, safari, firefox, or 'chrome:Profile 1'.",
    )
    parser.add_argument("--cookies-file", type=Path, default=None, help="Netscape cookies.txt file for yt-dlp.")
    parser.add_argument("--sleep-interval", type=float, default=0.0, help="Seconds yt-dlp sleeps before URL download.")
    parser.add_argument("--sleep-requests", type=float, default=0.0, help="Seconds yt-dlp sleeps between internal HTTP requests.")
    parser.add_argument(
        "--max-sleep-interval",
        type=float,
        default=None,
        help="Optional upper bound for randomized yt-dlp sleep; use with --sleep-interval.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    model_path = _resolve(args.model)
    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        print("Train first: python scripts/train_ball_detector.py")
        return 1

    if is_url(args.video):
        video_path = download_video_url(
            args.video,
            _resolve(args.url_cache_dir),
            force=args.force_download,
            slug_prefix="inference",
            cookies_from_browser=args.cookies_from_browser,
            cookies_file=args.cookies_file,
            sleep_interval=args.sleep_interval,
            max_sleep_interval=args.max_sleep_interval,
            sleep_requests=args.sleep_requests,
        )
    else:
        video_path = _resolve(Path(args.video))
    if not video_path.exists():
        print(f"[ERROR] Video not found: {video_path}")
        return 1

    out_path = _resolve(args.out) if args.out else ROOT / "runs" / "ball_detector" / f"{video_path.stem}_detected.mp4"
    run_video(model_path, video_path, out_path, args.conf, args.imgsz, args.show)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
