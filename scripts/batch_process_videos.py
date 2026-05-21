"""Batch runner for all videos in a directory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NamedTuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from shot_detector import run_pipeline  # noqa: E402
from video_source import download_video_url, read_url_list  # noqa: E402


class VideoSource(NamedTuple):
    """A local video or a URL that will be downloaded before processing."""

    name: str
    path: Path | None = None
    url: str | None = None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch process football 1v1 videos.")
    parser.add_argument("--input-dir", default="data/raw_videos", help="Directory containing video files and URL .txt lists.")
    parser.add_argument("--model", default=None, help="YOLO ball detector model path.")
    parser.add_argument("--config", default=None, help="Pitch/goal configuration JSON path.")
    parser.add_argument("--output-dir", default="outputs", help="Base output directory.")
    parser.add_argument("--pattern", default="*.mp4", help="Glob pattern for videos.")
    parser.add_argument("--url-list", action="append", default=None, help="Text file with one video URL per line. Can be passed more than once.")
    parser.add_argument("--no-auto-url-lists", action="store_true", help="Do not auto-read *.txt files from --input-dir.")
    parser.add_argument("--url-cache-dir", default=None, help="Directory used to cache downloaded URLs. Defaults to INPUT_DIR/url_cache.")
    parser.add_argument("--force-download", action="store_true", help="Download URL videos again even if cached.")
    parser.add_argument("--url-start-index", type=int, default=1, help="One-based URL index to start from when reading URL lists.")
    parser.add_argument("--max-downloads", type=int, default=None, help="Limit how many URL entries are attempted in this run.")
    parser.add_argument(
        "--cookies-from-browser",
        default=None,
        help="Load cookies from a browser for yt-dlp, e.g. chrome, safari, firefox, or 'chrome:Profile 1'.",
    )
    parser.add_argument("--cookies-file", default=None, help="Netscape cookies.txt file for yt-dlp.")
    parser.add_argument("--sleep-interval", type=float, default=0.0, help="Seconds yt-dlp sleeps before each URL download.")
    parser.add_argument("--sleep-requests", type=float, default=0.0, help="Seconds yt-dlp sleeps between internal HTTP requests.")
    parser.add_argument(
        "--max-sleep-interval",
        type=float,
        default=None,
        help="Optional upper bound for randomized yt-dlp sleep; use with --sleep-interval.",
    )
    parser.add_argument("--annotated", action="store_true", help="Write annotated videos.")
    parser.add_argument("--review-frames", action="store_true", help="Write review frames.")
    parser.add_argument("--only-inside-box", action="store_true", help="Filter shots outside the penalty area.")
    parser.add_argument("--no-progress", action="store_true", help="Disable per-video progress bars.")
    return parser


def _collect_sources(
    input_dir: Path,
    pattern: str,
    url_lists: list[str] | None,
    auto_url_lists: bool,
    url_start_index: int = 1,
    max_downloads: int | None = None,
) -> list[VideoSource]:
    sources = [VideoSource(name=video.stem, path=video) for video in sorted(input_dir.glob(pattern))]

    list_paths: list[Path] = []
    if auto_url_lists:
        list_paths.extend(sorted(input_dir.glob("*.txt")))
    if url_lists:
        list_paths.extend(Path(path) for path in url_lists)

    seen_lists: set[Path] = set()
    url_seen = 0
    url_added = 0
    start_index = max(1, url_start_index)
    for list_path in list_paths:
        resolved = list_path.expanduser().resolve()
        if resolved in seen_lists:
            continue
        seen_lists.add(resolved)
        urls = read_url_list(resolved)
        for index, url in enumerate(urls, start=1):
            url_seen += 1
            if url_seen < start_index:
                continue
            if max_downloads is not None and url_added >= max(0, max_downloads):
                return sources
            sources.append(VideoSource(name=f"{list_path.stem}_{index:03d}", url=url))
            url_added += 1
    return sources


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    cache_dir = Path(args.url_cache_dir) if args.url_cache_dir else input_dir / "url_cache"
    sources = _collect_sources(
        input_dir=input_dir,
        pattern=args.pattern,
        url_lists=args.url_list,
        auto_url_lists=not args.no_auto_url_lists,
        url_start_index=args.url_start_index,
        max_downloads=args.max_downloads,
    )
    if not sources:
        print(f"No videos or URL lists found in {input_dir}")
        return 1

    for source in sources:
        stem = source.name
        if source.url:
            try:
                video = download_video_url(
                    source.url,
                    cache_dir=cache_dir,
                    force=args.force_download,
                    slug_prefix=source.name,
                    cookies_from_browser=args.cookies_from_browser,
                    cookies_file=args.cookies_file,
                    sleep_interval=args.sleep_interval,
                    max_sleep_interval=args.max_sleep_interval,
                    sleep_requests=args.sleep_requests,
                )
            except Exception as exc:
                print(f"{source.name}: download_failed ({exc})")
                continue
        else:
            video = source.path

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
        print(f"{stem}: {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
