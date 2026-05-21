"""Resolve local video files and URL-based video sources."""

from __future__ import annotations

import hashlib
import re
import shutil
from pathlib import Path
from urllib.parse import urlparse


URL_SCHEMES = {"http", "https"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}


# Frame extraction only needs video. Prefer a single MP4 video stream so yt-dlp
# does not require ffmpeg to merge separate video/audio tracks.
DEFAULT_VIDEO_FORMAT = (
    "best[ext=mp4][vcodec!=none]/"
    "bestvideo[ext=mp4]/"
    "best[vcodec!=none]/"
    "best"
)


def is_url(value: str | Path | None) -> bool:
    """Return whether a string looks like an HTTP(S) URL."""
    if value is None:
        return False
    parsed = urlparse(str(value).strip())
    return parsed.scheme.lower() in URL_SCHEMES and bool(parsed.netloc)


def read_url_list(path: str | Path) -> list[str]:
    """Read one URL per line from a text file.

    Empty lines and comments starting with `#` are ignored.
    """
    urls: list[str] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            clean = line.strip()
            if not clean or clean.startswith("#"):
                continue
            if not is_url(clean):
                raise ValueError(f"Invalid URL in {path}: {clean}")
            urls.append(clean)
    return urls


def url_slug(url: str, prefix: str = "url") -> str:
    """Create a deterministic filesystem-safe id for a URL."""
    parsed = urlparse(url)
    host = parsed.netloc.replace(":", "_").replace(".", "_")
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
    host = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in host)
    prefix = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in prefix).strip("_")
    prefix = prefix or "url"
    return f"{prefix}_{host}_{digest}" if host else f"{prefix}_{digest}"


def _existing_download(cache_dir: Path, slug: str) -> Path | None:
    for candidate in sorted(cache_dir.glob(f"{slug}.*")):
        if candidate.suffix in {".part", ".ytdl", ".tmp"}:
            continue
        if candidate.suffix.lower() in VIDEO_EXTENSIONS and candidate.is_file():
            return candidate
    return None


def _cookies_from_browser_tuple(browser_spec: str | None) -> tuple[str, str | None, str | None, str | None] | None:
    """Parse yt-dlp's BROWSER[+KEYRING][:PROFILE][::CONTAINER] syntax."""
    if not browser_spec:
        return None
    match = re.fullmatch(
        r"(?x)"
        r"(?P<name>[^+:]+)"
        r"(?:\s*\+\s*(?P<keyring>[^:]+))?"
        r"(?:\s*:\s*(?!:)(?P<profile>.+?))?"
        r"(?:\s*::\s*(?P<container>.+))?",
        browser_spec.strip(),
    )
    if match is None:
        raise ValueError(f"Invalid --cookies-from-browser value: {browser_spec}")
    browser_name, keyring, profile, container = match.group("name", "keyring", "profile", "container")
    return browser_name.lower(), profile, keyring.upper() if keyring else None, container


def download_video_url(
    url: str,
    cache_dir: str | Path,
    force: bool = False,
    slug_prefix: str = "url",
    with_audio: bool = False,
    cookies_from_browser: str | None = None,
    cookies_file: str | Path | None = None,
    sleep_interval: float = 0.0,
    max_sleep_interval: float | None = None,
    sleep_requests: float = 0.0,
) -> Path:
    """Download a video URL to a local cache and return the local path.

    Uses `yt-dlp`, which handles direct MP4 URLs and common video platforms.
    """
    if not is_url(url):
        raise ValueError(f"Not a valid HTTP(S) URL: {url}")

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    slug = url_slug(url, prefix=slug_prefix)
    existing = _existing_download(cache_path, slug)
    if existing and not force:
        return existing

    try:
        from yt_dlp import YoutubeDL
    except Exception as exc:
        raise RuntimeError(
            "yt-dlp is required to download video URLs. Install requirements with `pip install -r requirements.txt`."
        ) from exc

    output_template = str(cache_path / f"{slug}.%(ext)s")
    options = {
        "outtmpl": output_template,
        "format": DEFAULT_VIDEO_FORMAT,
        "noplaylist": True,
        "quiet": False,
        "no_warnings": False,
        "restrictfilenames": True,
        "overwrites": force,
        "retries": 3,
        "fragment_retries": 3,
        "extractor_retries": 3,
    }
    browser_cookies = _cookies_from_browser_tuple(cookies_from_browser)
    if browser_cookies:
        options["cookiesfrombrowser"] = browser_cookies
    if cookies_file:
        options["cookiefile"] = str(cookies_file)
    if sleep_interval and sleep_interval > 0:
        options["sleep_interval"] = float(sleep_interval)
        if max_sleep_interval and max_sleep_interval > sleep_interval:
            options["max_sleep_interval"] = float(max_sleep_interval)
    if sleep_requests and sleep_requests > 0:
        options["sleep_interval_requests"] = float(sleep_requests)
    if with_audio and shutil.which("ffmpeg"):
        options.update(
            {
                "format": "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/" + DEFAULT_VIDEO_FORMAT,
                "merge_output_format": "mp4",
            }
        )
    try:
        with YoutubeDL(options) as ydl:
            ydl.download([url])
    except Exception as exc:
        message = str(exc)
        hints: list[str] = []
        if "HTTP Error 429" in message or "Too Many Requests" in message:
            hints.append("YouTube rate-limited this session. Wait 30-60 minutes and retry in smaller batches.")
        if "not a bot" in message or "Sign in to confirm" in message:
            hints.append("Use --cookies-from-browser chrome/safari/firefox or --cookies-file with an exported cookies.txt.")
        if hints:
            raise RuntimeError(f"{message} Hints: {' '.join(hints)}") from exc
        raise

    downloaded = _existing_download(cache_path, slug)
    if downloaded is None:
        candidates = [path for path in cache_path.glob(f"{slug}.*") if path.is_file() and path.suffix != ".part"]
        if candidates:
            return sorted(candidates)[0]
        raise RuntimeError(f"Video URL downloaded but no output file was found for {url}")
    return downloaded
