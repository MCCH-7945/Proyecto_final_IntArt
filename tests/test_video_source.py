from pathlib import Path

from video_source import is_url, read_url_list, url_slug


def test_is_url_detects_http_urls_only():
    assert is_url("https://example.com/video.mp4") is True
    assert is_url("http://example.com/video.mp4") is True
    assert is_url("data/raw_videos/input.mp4") is False
    assert is_url(None) is False


def test_read_url_list_ignores_comments_and_blank_lines(tmp_path: Path):
    url_file = tmp_path / "urls.txt"
    url_file.write_text(
        "\n# comment\nhttps://example.com/a.mp4\n\nhttps://example.com/b.mp4\n",
        encoding="utf-8",
    )
    assert read_url_list(url_file) == [
        "https://example.com/a.mp4",
        "https://example.com/b.mp4",
    ]


def test_url_slug_is_deterministic_and_safe():
    first = url_slug("https://example.com/video.mp4?x=1", prefix="my list")
    second = url_slug("https://example.com/video.mp4?x=1", prefix="my list")
    assert first == second
    assert " " not in first
