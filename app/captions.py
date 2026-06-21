from __future__ import annotations

import html
import logging
import re
from pathlib import Path
from typing import Any

import requests

from app.config import AppConfig
from app.errors import AppError
from app.models import DownloadResult, Transcript, TranscriptSegment
from app.utils import dedupe_adjacent_segments, read_json, read_model, write_json

logger = logging.getLogger(__name__)

CAPTION_SOURCE = "youtube_caption"
_TIMING_RE = re.compile(
    r"(?P<start>\d{1,2}:\d{2}:\d{2}[.,]\d{3}|\d{1,2}:\d{2}[.,]\d{3})\s+-->\s+"
    r"(?P<end>\d{1,2}:\d{2}:\d{2}[.,]\d{3}|\d{1,2}:\d{2}[.,]\d{3})"
)
_TAG_RE = re.compile(r"<[^>]+>")


def youtube_transcript_path_for(video_id: str, config: AppConfig) -> Path:
    return config.transcript_output_dir / f"{video_id}.youtube.transcript.json"


def extract_youtube_caption_transcript(
    download: DownloadResult,
    config: AppConfig,
    force: bool = False,
) -> Transcript | None:
    config.transcript_output_dir.mkdir(parents=True, exist_ok=True)
    output_path = youtube_transcript_path_for(download.episode.video_id, config)
    if output_path.exists() and not force:
        logger.info("Using cached YouTube caption transcript %s", output_path)
        return read_model(output_path, Transcript)

    if not download.episode.info_json_path:
        return None
    info_path = Path(download.episode.info_json_path)
    if not info_path.exists():
        return None

    info = read_json(info_path)
    track = select_caption_track(info)
    if not track:
        logger.info("No usable English YouTube captions found for %s", download.episode.video_id)
        return None

    try:
        vtt_text = download_caption_text(track["url"])
    except AppError:
        logger.exception("YouTube caption download failed")
        return None
    if not vtt_text.strip():
        return None

    raw_path = download.episode_dir / f"{download.episode.video_id}.youtube.{track['language']}.vtt"
    raw_path.write_text(vtt_text, encoding="utf-8")
    transcript = parse_vtt_transcript(vtt_text, download.episode.video_id)
    if not transcript.segments:
        return None

    write_json(output_path, transcript)
    logger.info(
        "Cached %s YouTube caption segments from %s captions",
        len(transcript.segments),
        track["kind"],
    )
    return transcript


def select_caption_track(info: dict[str, Any]) -> dict[str, str] | None:
    manual = _select_from_group(info.get("subtitles") or {}, "manual")
    if manual:
        return manual
    return _select_from_group(info.get("automatic_captions") or {}, "auto")


def download_caption_text(url: str) -> str:
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise AppError(f"Could not download YouTube captions: {exc}") from exc
    return response.text


def parse_vtt_transcript(vtt_text: str, video_id: str) -> Transcript:
    segments: list[TranscriptSegment] = []
    lines = vtt_text.replace("\ufeff", "").splitlines()
    index = 0
    cue_text: list[str] = []
    start_time: float | None = None
    end_time: float | None = None

    def flush() -> None:
        nonlocal cue_text, start_time, end_time
        if start_time is None or end_time is None or not cue_text:
            cue_text = []
            start_time = None
            end_time = None
            return
        text = _clean_caption_text(" ".join(cue_text))
        if text:
            segments.append(
                TranscriptSegment(
                    segment_id=f"{video_id}:ytcap:{len(segments):06d}",
                    video_id=video_id,
                    start_time=start_time,
                    end_time=end_time,
                    text=text,
                    source=CAPTION_SOURCE,
                )
            )
        cue_text = []
        start_time = None
        end_time = None

    while index < len(lines):
        line = lines[index].strip()
        if not line:
            flush()
            index += 1
            continue
        if line.startswith(("WEBVTT", "Kind:", "Language:", "NOTE", "STYLE", "REGION")):
            index += 1
            continue
        timing = _TIMING_RE.search(line)
        if timing:
            flush()
            start_time = _parse_vtt_time(timing.group("start"))
            end_time = _parse_vtt_time(timing.group("end"))
        elif start_time is not None:
            cue_text.append(line)
        index += 1

    flush()
    return Transcript(video_id=video_id, segments=dedupe_adjacent_segments(segments), source=CAPTION_SOURCE)


def _select_from_group(group: dict[str, list[dict[str, Any]]], kind: str) -> dict[str, str] | None:
    for language in _preferred_languages(group):
        tracks = group.get(language) or []
        vtt_track = next((track for track in tracks if track.get("ext") == "vtt"), None)
        fallback = next((track for track in tracks if track.get("url")), None)
        selected = vtt_track or fallback
        if selected and selected.get("url"):
            return {"url": str(selected["url"]), "language": language, "kind": kind}
    return None


def _preferred_languages(group: dict[str, list[dict[str, Any]]]) -> list[str]:
    exact = [language for language in ["en", "en-US", "en-GB"] if language in group]
    english = sorted(language for language in group if language.lower().startswith("en"))
    return list(dict.fromkeys([*exact, *english]))


def _parse_vtt_time(value: str) -> float:
    parts = value.replace(",", ".").split(":")
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    hours, minutes, seconds = parts
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _clean_caption_text(text: str) -> str:
    text = _TAG_RE.sub("", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
