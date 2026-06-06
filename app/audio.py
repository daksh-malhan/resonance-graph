from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from app.config import AppConfig
from app.errors import AppError
from app.models import EpisodeMetadata
from app.utils import require_executable

logger = logging.getLogger(__name__)


def extract_audio(episode: EpisodeMetadata, config: AppConfig, force: bool = False) -> Path:
    if not episode.local_video_path:
        raise AppError("Episode metadata does not include a downloaded video path.")

    video_path = Path(episode.local_video_path)
    if not video_path.exists():
        raise AppError(f"Downloaded video file is missing: {video_path}")

    ffmpeg = require_executable("ffmpeg", "Install FFmpeg and ensure it is on PATH.")
    config.audio_output_dir.mkdir(parents=True, exist_ok=True)
    audio_path = config.audio_output_dir / f"{episode.video_id}.wav"

    if audio_path.exists() and not force:
        logger.info("Using cached audio file %s", audio_path)
        return audio_path

    command = [
        ffmpeg,
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-af",
        "loudnorm",
        str(audio_path),
    ]
    logger.info("Extracting 16 kHz mono WAV audio to %s", audio_path)
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise AppError(f"FFmpeg audio extraction failed:\n{result.stderr.strip()}")
    return audio_path
