from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from app.config import AppConfig
from app.errors import AppError
from app.models import ChannelVideo, DownloadResult, EpisodeMetadata, SourceMetadata, YouTubeUrl
from app.utils import read_json, require_executable, write_json

logger = logging.getLogger(__name__)

YOUTUBE_WATCH_URL = "https://www.youtube.com/watch?v={video_id}"


def parse_youtube_metadata(info: dict[str, Any], episode_dir: Path) -> EpisodeMetadata:
    video_id = info.get("id")
    if not video_id:
        raise AppError("yt-dlp metadata did not include a video id")

    info_json_path = episode_dir / f"{video_id}.info.json"
    local_video_path = _find_local_video_path(info, episode_dir)
    return EpisodeMetadata(
        video_id=video_id,
        title=info.get("title") or video_id,
        channel=info.get("channel") or info.get("uploader"),
        source_url=info.get("webpage_url") or info.get("original_url") or "",
        duration=info.get("duration"),
        upload_date=info.get("upload_date"),
        local_video_path=local_video_path,
        info_json_path=info_json_path if info_json_path.exists() else None,
    )


def _find_local_video_path(info: dict[str, Any], episode_dir: Path) -> Path | None:
    for requested in info.get("requested_downloads") or []:
        filepath = requested.get("filepath")
        if filepath:
            path = Path(filepath)
            if path.exists():
                return path

    for key in ["filepath", "_filename"]:
        value = info.get(key)
        if value and Path(value).exists():
            return Path(value)

    candidates = [
        path
        for path in episode_dir.iterdir()
        if path.is_file()
        and path.suffix.lower()
        not in {".json", ".part", ".ytdl", ".txt", ".archive"}
        and not path.name.endswith(".info.json")
    ]
    return candidates[0] if candidates else None


def _extract_video_id_without_download(url: str, config: AppConfig) -> str:
    try:
        import yt_dlp
    except ImportError as exc:
        raise AppError("yt-dlp is not installed. Install project dependencies first.") from exc

    with yt_dlp.YoutubeDL({"quiet": True, "skip_download": True, "noplaylist": True}) as ydl:
        info = ydl.extract_info(url, download=False)
    if not info or not info.get("id"):
        raise AppError("Could not resolve a YouTube video id from the provided URL")
    return str(info["id"])


def download_youtube_video(url: str, config: AppConfig, force: bool = False) -> DownloadResult:
    try:
        YouTubeUrl(url=url)
    except ValidationError as exc:
        raise AppError("Please provide a valid YouTube video URL.") from exc

    require_executable("yt-dlp", "Install it with 'pip install yt-dlp'.")
    ffmpeg_path = require_executable(
        "ffmpeg",
        "Install FFmpeg or reinstall the project so the imageio-ffmpeg fallback is available.",
    )
    try:
        import yt_dlp
        from yt_dlp.utils import DownloadError
    except ImportError as exc:
        raise AppError("yt-dlp is not installed. Install project dependencies first.") from exc

    config.ensure_directories()
    video_id = _extract_video_id_without_download(url, config)
    episode_dir = config.youtube_download_dir / video_id
    episode_dir.mkdir(parents=True, exist_ok=True)
    info_json_path = episode_dir / f"{video_id}.info.json"
    metadata_path = episode_dir / "metadata.json"

    if not force and metadata_path.exists():
        logger.info("Using cached YouTube metadata for %s", video_id)
        cached = DownloadResult.model_validate_json(metadata_path.read_text())
        if cached.episode.local_video_path and Path(cached.episode.local_video_path).exists():
            return cached

    archive_path = episode_dir / "download-archive.txt"
    ydl_opts = {
        "format": (
            f"bestvideo[height<={config.max_youtube_resolution}]+bestaudio/"
            f"best[height<={config.max_youtube_resolution}]/best"
        ),
        "outtmpl": str(episode_dir / "%(id)s.%(ext)s"),
        "merge_output_format": "mp4",
        "writeinfojson": True,
        "ffmpeg_location": ffmpeg_path,
        "download_archive": str(archive_path),
        "noplaylist": True,
        "quiet": False,
        "no_warnings": False,
        "ignoreerrors": False,
    }
    if force and archive_path.exists():
        archive_path.unlink()

    logger.info("Downloading approved YouTube video %s", url)
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
    except DownloadError as exc:
        raise AppError(f"yt-dlp download failed: {exc}") from exc

    if info_json_path.exists():
        info = read_json(info_json_path)

    episode = parse_youtube_metadata(info, episode_dir)
    if not episode.local_video_path:
        raise AppError("Download finished but no local video file was found.")

    source = SourceMetadata(id=episode.source_url or url, url=episode.source_url or url)
    result = DownloadResult(source=source, episode=episode, episode_dir=episode_dir)
    write_json(metadata_path, result)
    return result


def discover_channel_videos(
    channel_url: str,
    config: AppConfig,
    min_duration_seconds: int | None = None,
    limit: int | None = None,
) -> list[ChannelVideo]:
    """Return long-form videos from a YouTube channel without downloading media."""
    try:
        YouTubeUrl(url=channel_url)
    except ValidationError as exc:
        raise AppError("Please provide a valid YouTube channel URL.") from exc

    require_executable("yt-dlp", "Install it with 'pip install yt-dlp'.")
    try:
        import yt_dlp
        from yt_dlp.utils import DownloadError
    except ImportError as exc:
        raise AppError("yt-dlp is not installed. Install project dependencies first.") from exc

    min_duration = (
        min_duration_seconds
        if min_duration_seconds is not None
        else config.channel_min_video_duration_seconds
    )
    max_videos = config.channel_max_videos if limit is None else limit
    discovery_url = _normalize_channel_videos_url(channel_url)
    ydl_opts = {
        "extract_flat": "in_playlist",
        "skip_download": True,
        "quiet": True,
        "ignoreerrors": True,
    }
    if max_videos and max_videos > 0:
        ydl_opts["playlistend"] = max_videos

    logger.info("Discovering long-form videos from channel %s", discovery_url)
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(discovery_url, download=False)
    except DownloadError as exc:
        raise AppError(f"yt-dlp channel discovery failed: {exc}") from exc

    entries = (info or {}).get("entries") or []
    videos: list[ChannelVideo] = []
    seen: set[str] = set()
    for entry in entries:
        if not entry:
            continue
        video = _channel_video_from_entry(entry)
        if not video or video.video_id in seen:
            continue
        seen.add(video.video_id)
        if _is_short_or_too_short(video, min_duration):
            continue
        videos.append(video)
        if max_videos and max_videos > 0 and len(videos) >= max_videos:
            break

    return videos


def _normalize_channel_videos_url(channel_url: str) -> str:
    trimmed = channel_url.rstrip("/")
    if any(part in trimmed for part in ["/videos", "/streams", "/shorts", "/playlist?"]):
        return trimmed
    if "youtube.com/" in trimmed and "/watch?" not in trimmed:
        return f"{trimmed}/videos"
    return trimmed


def _channel_video_from_entry(entry: dict[str, Any]) -> ChannelVideo | None:
    video_id = entry.get("id")
    if not video_id:
        return None

    raw_url = entry.get("webpage_url") or entry.get("url")
    url = _normalize_video_url(str(raw_url), str(video_id))
    return ChannelVideo(
        video_id=str(video_id),
        title=entry.get("title") or str(video_id),
        url=url,
        duration=_parse_duration(entry.get("duration")),
        channel=entry.get("channel") or entry.get("uploader"),
    )


def _normalize_video_url(raw_url: str, video_id: str) -> str:
    if raw_url.startswith("http://") or raw_url.startswith("https://"):
        return raw_url
    if raw_url.startswith("/shorts/"):
        return f"https://www.youtube.com{raw_url}"
    if raw_url.startswith("/watch"):
        return f"https://www.youtube.com{raw_url}"
    return YOUTUBE_WATCH_URL.format(video_id=video_id)


def _parse_duration(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_short_or_too_short(video: ChannelVideo, min_duration_seconds: int) -> bool:
    if "/shorts/" in video.url:
        return True
    if video.duration is not None and video.duration < min_duration_seconds:
        return True
    return False
