from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path

from app.config import AppConfig
from app.errors import AppError
from app.models import Transcript, TranscriptSegment
from app.utils import read_model, require_executable, write_json

logger = logging.getLogger(__name__)


def transcript_path_for(video_id: str, config: AppConfig) -> Path:
    return config.transcript_output_dir / f"{video_id}.transcript.json"


def transcribe_audio(
    audio_path: Path,
    video_id: str,
    config: AppConfig,
    force: bool = False,
) -> Transcript:
    config.transcript_output_dir.mkdir(parents=True, exist_ok=True)
    output_path = transcript_path_for(video_id, config)
    if output_path.exists() and not force:
        logger.info("Using cached transcript %s", output_path)
        return read_model(output_path, Transcript)

    backend = config.transcription_backend.lower().strip()
    if backend == "faster-whisper":
        transcript = _transcribe_with_faster_whisper(audio_path, video_id, config)
    elif backend in {"whisper.cpp", "whisper-cpp", "whisper_cpp"}:
        transcript = _transcribe_with_whisper_cpp(audio_path, video_id, config)
    else:
        raise AppError(
            "Unsupported transcription backend. Use 'faster-whisper' or 'whisper.cpp'."
        )

    write_json(output_path, transcript)
    return transcript


def _transcribe_with_faster_whisper(
    audio_path: Path,
    video_id: str,
    config: AppConfig,
) -> Transcript:
    hf_home = config.model_cache_dir / "huggingface"
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_home / "hub"))
    os.environ.setdefault("HF_XET_CACHE", str(hf_home / "xet"))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise AppError(
            "faster-whisper is not installed. Install with "
            "'pip install local-podcast-graphrag[transcription]' or 'pip install faster-whisper'."
        ) from exc

    if not audio_path.exists():
        raise AppError(f"Audio file does not exist: {audio_path}")

    logger.info("Transcribing locally with faster-whisper model %s", config.faster_whisper_model)
    download_root = config.model_cache_dir / "faster-whisper"
    download_root.mkdir(parents=True, exist_ok=True)
    model = WhisperModel(
        config.faster_whisper_model,
        device=config.faster_whisper_device,
        compute_type=config.faster_whisper_compute_type,
        download_root=str(download_root),
    )
    segments_iter, _info = model.transcribe(str(audio_path), vad_filter=True)
    segments = [
        TranscriptSegment(
            segment_id=f"{video_id}:seg:{index:06d}",
            video_id=video_id,
            start_time=float(segment.start),
            end_time=float(segment.end),
            text=segment.text.strip(),
        )
        for index, segment in enumerate(segments_iter)
        if segment.text and segment.text.strip()
    ]
    return Transcript(video_id=video_id, segments=segments)


def _transcribe_with_whisper_cpp(
    audio_path: Path,
    video_id: str,
    config: AppConfig,
) -> Transcript:
    binary = require_executable(
        config.whisper_cpp_binary,
        "Install whisper.cpp and ensure whisper-cli is on PATH.",
    )
    if not config.whisper_cpp_model:
        raise AppError("WHISPER_CPP_MODEL must point to a local whisper.cpp model file.")

    output_prefix = config.transcript_output_dir / f"{video_id}.whispercpp"
    command = [
        binary,
        "-m",
        str(config.whisper_cpp_model),
        "-f",
        str(audio_path),
        "-oj",
        "-of",
        str(output_prefix),
    ]
    logger.info("Transcribing locally with whisper.cpp")
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise AppError(f"whisper.cpp transcription failed:\n{result.stderr.strip()}")

    whisper_json = output_prefix.with_suffix(".json")
    payload = json.loads(whisper_json.read_text())
    raw_segments = payload.get("transcription") or []
    segments = []
    for index, item in enumerate(raw_segments):
        offsets = item.get("offsets") or {}
        start_ms = offsets.get("from", 0)
        end_ms = offsets.get("to", 0)
        text = (item.get("text") or "").strip()
        if not text:
            continue
        segments.append(
            TranscriptSegment(
                segment_id=f"{video_id}:seg:{index:06d}",
                video_id=video_id,
                start_time=float(start_ms) / 1000,
                end_time=float(end_ms) / 1000,
                text=text,
            )
        )
    return Transcript(video_id=video_id, segments=segments)
