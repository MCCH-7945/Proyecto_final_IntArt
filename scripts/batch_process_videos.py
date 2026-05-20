"""Batch runner for all videos in a directory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from shot_detector import run_pipeline  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch process football 1v1 videos.")
    parser.add_argument("--input-dir", required=True, help="Directory containing video files.")
    parser.add_argument("--model", default=None, help="YOLO ball detector model path.")
    parser.add_argument("--config", default=None, help="Pitch/goal configuration JSON path.")
    parser.add_argument("--output-dir", default="outputs", help="Base output directory.")
    parser.add_argument("--pattern", default="*.mp4", help="Glob pattern for videos.")
    parser.add_argument("--annotated", action="store_true", help="Write annotated videos.")
    parser.add_argument("--review-frames", action="store_true", help="Write review frames.")
    parser.add_argument("--only-inside-box", action="store_true", help="Filter shots outside the penalty area.")
    parser.add_argument("--no-progress", action="store_true", help="Disable per-video progress bars.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    videos = sorted(input_dir.glob(args.pattern))
    if not videos:
        print(f"No videos found in {input_dir} with pattern {args.pattern}")
        return 1

    for video in videos:
        stem = video.stem
        frame_output = output_dir / "frame_level" / f"{stem}_ball_tracking.csv"
        shot_output = output_dir / "shot_level" / f"{stem}_shot_tuple.csv"
        annotated_video = output_dir / "annotated_videos" / f"{stem}_annotated.mp4" if args.annotated else None
        review_dir = output_dir / "review_frames" / stem if args.review_frames else None
        result = run_pipeline(
            video_path=video,
            model_path=args.model,
            config_path=args.config,
            frame_output=frame_output,
            shot_output=shot_output,
            only_inside_box=args.only_inside_box,
            annotated_video=annotated_video,
            save_review_frames_dir=review_dir,
            show_progress=not args.no_progress,
        )
        print(f"{video.name}: {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

