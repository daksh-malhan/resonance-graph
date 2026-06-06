from app.config import AppConfig
from app.youtube import _channel_video_from_entry, _is_short_or_too_short


def test_channel_entry_normalizes_watch_url() -> None:
    video = _channel_video_from_entry(
        {
            "id": "abc123",
            "title": "Long interview",
            "url": "abc123",
            "duration": 3600,
            "channel": "Example Channel",
        }
    )

    assert video is not None
    assert video.url == "https://www.youtube.com/watch?v=abc123"
    assert video.duration == 3600
    assert not _is_short_or_too_short(video, min_duration_seconds=61)


def test_channel_filter_rejects_shorts_url() -> None:
    video = _channel_video_from_entry(
        {
            "id": "short123",
            "title": "Short clip",
            "url": "/shorts/short123",
            "duration": 180,
        }
    )

    assert video is not None
    assert _is_short_or_too_short(video, min_duration_seconds=61)


def test_channel_filter_rejects_short_duration() -> None:
    video = _channel_video_from_entry(
        {
            "id": "tiny123",
            "title": "Tiny clip",
            "url": "tiny123",
            "duration": 30,
        }
    )

    assert video is not None
    assert _is_short_or_too_short(video, min_duration_seconds=61)


def test_channel_config_defaults() -> None:
    config = AppConfig()

    assert config.channel_min_video_duration_seconds == 61
    assert config.channel_max_videos == 0
