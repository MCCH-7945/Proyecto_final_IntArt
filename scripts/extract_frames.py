"""Extract annotation frames from local videos or URL lists.

Examples:
    python scripts/extract_frames.py --fps 2 --max-frames 400
    python scripts/extract_frames.py --videos data/raw_videos/clip.mp4 --fps 5

If data/raw_videos contains .txt files, each non-empty non-comment line is
treated as a video URL and downloaded to data/raw_videos/url_cache before frame
extraction.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from video_source import download_video_url, read_url_list  # noqa: E402


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv", ".webm"}


def extract_video(
    video_path: Path,
    out_dir: Path,
    fps_target: float,
    max_frames: int,
    quality: int,
    sampling_mode: str = "uniform",
    start_sec: float | None = None,
    end_sec: float | None = None,
) -> int:
    """Extract frames from a video.

    `uniform` spreads `max_frames` across the selected time range. This is the
    safest default for long matches because it avoids sampling only kickoff.

    `fps` preserves the older behavior: sample at approximately `fps_target`
    from the beginning of the selected range until `max_frames` is reached.
    """
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        return 0

    video_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    out_dir.mkdir(parents=True, exist_ok=True)

    start_frame = max(0, int(round((start_sec or 0.0) * video_fps)))
    end_frame = total_frames
    if end_sec is not None and end_sec > 0:
        end_frame = min(total_frames, int(round(end_sec * video_fps))) if total_frames else int(round(end_sec * video_fps))
    if total_frames:
        start_frame = min(start_frame, total_frames - 1)
        end_frame = min(max(end_frame, start_frame + 1), total_frames)

    if sampling_mode == "uniform" and total_frames > 0:
        span = max(1, end_frame - start_frame)
        n_frames = min(max_frames, span)
        if n_frames <= 1:
            frame_indices = [start_frame]
        else:
            step = (span - 1) / (n_frames - 1)
            frame_indices = sorted({start_frame + int(round(index * step)) for index in range(n_frames)})
    else:
        step = max(1, round(video_fps / max(fps_target, 0.001)))
        frame_indices = list(range(start_frame, end_frame, step))[:max_frames]

    saved = 0
    with tqdm(total=len(frame_indices), desc=video_path.name, unit="fr") as progress:
        for frame_idx in frame_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = capture.read()
            if not ok:
                progress.update(1)
                continue
            out_path = out_dir / f"frame_{frame_idx:07d}.jpg"
            cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            saved += 1
            progress.update(1)

    capture.release()
    return saved


def _local_videos(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        return []
    return sorted(path for path in input_dir.iterdir() if path.suffix.lower() in VIDEO_EXTS)


def _urls_from_lists(input_dir: Path, url_lists: list[str] | None, auto_url_lists: bool) -> list[tuple[str, str]]:
    list_paths: list[Path] = []
    if auto_url_lists and input_dir.exists():
        list_paths.extend(sorted(input_dir.glob("*.txt")))
    if url_lists:
        list_paths.extend(Path(path) for path in url_lists)

    urls: list[tuple[str, str]] = []
    seen: set[Path] = set()
    for list_path in list_paths:
        resolved = list_path.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        for index, url in enumerate(read_url_list(resolved), start=1):
            urls.append((f"{list_path.stem}_{index:03d}", url))
    return urls


def collect_videos(args: argparse.Namespace) -> list[Path]:
    """Collect local videos and downloaded URL videos."""
    if args.videos:
        videos = [Path(video) for video in args.videos]
    else:
        videos = _local_videos(args.input_dir)

    if not args.no_auto_url_lists or args.url_list:
        url_items = _urls_from_lists(args.input_dir, args.url_list, not args.no_auto_url_lists)
        start_index = max(1, args.url_start_index)
        if start_index > 1:
            url_items = url_items[start_index - 1 :]
        if args.max_downloads is not None:
            url_items = url_items[: max(0, args.max_downloads)]
        for slug, url in url_items:
            try:
                downloaded = download_video_url(
                    url,
                    cache_dir=args.url_cache_dir,
                    force=args.force_download,
                    slug_prefix=slug,
                    cookies_from_browser=args.cookies_from_browser,
                    cookies_file=args.cookies_file,
                    sleep_interval=args.sleep_interval,
                    max_sleep_interval=args.max_sleep_interval,
                    sleep_requests=args.sleep_requests,
                )
                videos.append(downloaded)
            except Exception as exc:
                print(f"[WARN] URL download failed for {slug}: {exc}")
    return videos


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract frames for ball annotation.")
    parser.add_argument("--videos", nargs="*", default=None, help="Specific videos to extract.")
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data" / "raw_videos")
    parser.add_argument("--url-list", action="append", default=None, help="Text file with one video URL per line.")
    parser.add_argument("--no-auto-url-lists", action="store_true", help="Do not auto-read *.txt in input dir.")
    parser.add_argument("--url-cache-dir", type=Path, default=ROOT / "data" / "raw_videos" / "url_cache")
    parser.add_argument("--force-download", action="store_true", help="Redownload URL videos even if cached.")
    parser.add_argument("--url-start-index", type=int, default=1, help="One-based URL index to start from when reading URL lists.")
    parser.add_argument("--max-downloads", type=int, default=None, help="Limit how many URL entries are attempted in this run.")
    parser.add_argument(
        "--cookies-from-browser",
        default=None,
        help="Load cookies from a browser for yt-dlp, e.g. chrome, safari, firefox, or 'chrome:Profile 1'.",
    )
    parser.add_argument("--cookies-file", type=Path, default=None, help="Netscape cookies.txt file for yt-dlp.")
    parser.add_argument("--sleep-interval", type=float, default=0.0, help="Seconds yt-dlp sleeps before each URL download.")
    parser.add_argument("--sleep-requests", type=float, default=0.0, help="Seconds yt-dlp sleeps between internal HTTP requests.")
    parser.add_argument(
        "--max-sleep-interval",
        type=float,
        default=None,
        help="Optional upper bound for randomized yt-dlp sleep; use with --sleep-interval.",
    )
    parser.add_argument("--fps", type=float, default=2.0, help="Frames per second to extract.")
    parser.add_argument("--max-frames", type=int, default=500, help="Max frames per video.")
    parser.add_argument(
        "--sampling-mode",
        choices=["uniform", "fps"],
        default="uniform",
        help="uniform spreads frames across the whole video; fps samples from the selected range start.",
    )
    parser.add_argument("--start-sec", type=float, default=None, help="Optional start time for extraction range.")
    parser.add_argument("--end-sec", type=float, default=None, help="Optional end time for extraction range.")
    parser.add_argument("--quality", type=int, default=90, help="JPEG quality, 0-100.")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "raw_frames")
    parser.add_argument("--clear-output", action="store_true", help="Delete output frame directory before extracting.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    videos = collect_videos(args)
    if not videos:
        print("[WARN] No videos found. Add videos or URL .txt files under data/raw_videos/")
        return 0

    if args.clear_output and args.out_dir.exists():
        shutil.rmtree(args.out_dir)

    print(f"Videos found: {len(videos)}")
    print(f"Sampling mode: {args.sampling_mode}")
    print(f"Target FPS: {args.fps}")
    print(f"Max frames per video: {args.max_frames}")
    if args.start_sec is not None or args.end_sec is not None:
        print(f"Range: {args.start_sec or 0}s to {args.end_sec if args.end_sec is not None else 'end'}")
    print(f"Output: {args.out_dir}")

    total_saved = 0
    for video_path in videos:
        out_subdir = args.out_dir / video_path.stem
        saved = extract_video(
            video_path,
            out_subdir,
            args.fps,
            args.max_frames,
            args.quality,
            sampling_mode=args.sampling_mode,
            start_sec=args.start_sec,
            end_sec=args.end_sec,
        )
        print(f"{video_path.name}: {saved} frames -> {out_subdir}")
        total_saved += saved

    print(f"Total frames extracted: {total_saved}")
    print("Next step: python scripts/select_frames.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
