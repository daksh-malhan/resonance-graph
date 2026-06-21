from __future__ import annotations

import json
import logging
import os
import subprocess
import importlib.util
from pathlib import Path

from app.config import AppConfig
from app.errors import AppError
from app.models import Transcript, TranscriptSegment
from app.utils import (
    dedupe_adjacent_segments,
    ranges_overlap,
    read_model,
    require_executable,
    write_json,
)

logger = logging.getLogger(__name__)
LOCAL_SOURCE = "local_whisper"
MERGED_SOURCE = "merged"


def _relabel(transcript: Transcript, source: str) -> Transcript:
    relabeled = transcript.model_copy(update={"source": source})
    relabeled.segments[:] = [
        segment.model_copy(update={"source": source}) for segment in relabeled.segments
    ]
    return relabeled


def transcript_path_for(video_id: str, config: AppConfig) -> Path:
    return config.transcript_output_dir / f"{video_id}.transcript.json"


def local_transcript_path_for(video_id: str, config: AppConfig) -> Path:
    return config.transcript_output_dir / f"{video_id}.local.transcript.json"


def whisper_cpp_json_path_for(video_id: str, config: AppConfig) -> Path:
    return config.transcript_output_dir / f"{video_id}.whispercpp.json"


def transcribe_audio(
    audio_path: Path,
    video_id: str,
    config: AppConfig,
    force: bool = False,
) -> Transcript:
    config.transcript_output_dir.mkdir(parents=True, exist_ok=True)
    output_path = local_transcript_path_for(video_id, config)
    if output_path.exists() and not force:
        logger.info("Using cached local transcript %s", output_path)
        return read_model(output_path, Transcript)
    primary_path = transcript_path_for(video_id, config)
    if primary_path.exists() and not force:
        primary = read_model(primary_path, Transcript)
        if primary.source in {LOCAL_SOURCE, "local_whisper"}:
            logger.info("Migrating existing primary local transcript to %s", output_path)
            primary = _relabel(primary, LOCAL_SOURCE)
            write_json(output_path, primary)
            return primary

    backend = _local_backend(config)
    whisper_cpp_json = whisper_cpp_json_path_for(video_id, config)
    if backend in {"whisper.cpp", "whisper-cpp", "whisper_cpp"} and whisper_cpp_json.exists() and not force:
        logger.info("Importing cached whisper.cpp transcript %s", whisper_cpp_json)
        transcript = _relabel(_transcript_from_whisper_cpp_json(whisper_cpp_json, video_id), LOCAL_SOURCE)
        write_json(output_path, transcript)
        return transcript

    if backend == "faster-whisper":
        transcript = _transcribe_with_faster_whisper(audio_path, video_id, config)
    elif backend in {"whisper.cpp", "whisper-cpp", "whisper_cpp"}:
        try:
            transcript = _transcribe_with_whisper_cpp(audio_path, video_id, config)
        except AppError:
            if config.transcription_backend.lower().strip() in {
                "whisper.cpp",
                "whisper-cpp",
                "whisper_cpp",
            }:
                raise
            logger.exception("whisper.cpp failed; falling back to faster-whisper")
            transcript = _transcribe_with_faster_whisper(audio_path, video_id, config)
    else:
        raise AppError(
            "Unsupported transcription backend. Use 'faster-whisper' or 'whisper.cpp'."
        )

    transcript = _relabel(transcript, LOCAL_SOURCE)
    write_json(output_path, transcript)
    return transcript


def write_primary_transcript(transcript: Transcript, config: AppConfig) -> Transcript:
    output_path = transcript_path_for(transcript.video_id, config)
    write_json(output_path, transcript)
    return transcript


def preserve_primary_local_transcript(video_id: str, config: AppConfig) -> Transcript | None:
    local_path = local_transcript_path_for(video_id, config)
    if local_path.exists():
        return read_model(local_path, Transcript)

    primary_path = transcript_path_for(video_id, config)
    if not primary_path.exists():
        return None

    primary = read_model(primary_path, Transcript)
    if primary.source not in {LOCAL_SOURCE, "local_whisper"}:
        return None

    primary = _relabel(primary, LOCAL_SOURCE)
    write_json(local_path, primary)
    return primary


def local_transcription_backend_status(config: AppConfig) -> str:
    backend = _local_backend(config)
    if backend == "faster-whisper":
        if importlib.util.find_spec("faster_whisper") is None:
            raise AppError("faster-whisper is not installed.")
        return "faster-whisper ready"

    if backend in {"whisper.cpp", "whisper-cpp", "whisper_cpp"}:
        try:
            require_executable(config.whisper_cpp_binary, "Install whisper.cpp.")
            if not config.whisper_cpp_model:
                raise AppError("WHISPER_CPP_MODEL is not configured.")
            if not config.whisper_cpp_model.exists():
                raise AppError(f"WHISPER_CPP_MODEL does not exist: {config.whisper_cpp_model}")
        except AppError as exc:
            if importlib.util.find_spec("faster_whisper") is not None:
                return f"whisper.cpp/Metal unavailable ({exc}); faster-whisper fallback ready"
            raise
        return "whisper.cpp/Metal preferred backend ready"

    raise AppError("Unsupported transcription backend.")


def merge_transcripts(
    caption_transcript: Transcript | None,
    local_transcript: Transcript | None,
    video_id: str,
) -> Transcript:
    if caption_transcript is None and local_transcript is None:
        return Transcript(video_id=video_id, segments=[], source=MERGED_SOURCE)
    if caption_transcript is None:
        assert local_transcript is not None
        return local_transcript.model_copy(update={"source": LOCAL_SOURCE})
    if local_transcript is None:
        return caption_transcript.model_copy(update={"source": "youtube_caption"})

    merged: list[TranscriptSegment] = []
    caption_segments = caption_transcript.segments
    for local in local_transcript.segments:
        if _usable_local_segment(local):
            merged.append(_merged_segment(local, len(merged), video_id, local.text))
            continue

        caption = _best_caption_overlap(local, caption_segments)
        if caption:
            merged.append(_merged_segment(caption, len(merged), video_id, caption.text))

    local_ranges = [(segment.start_time, segment.end_time) for segment in local_transcript.segments]
    for caption in caption_segments:
        if not any(ranges_overlap(caption.start_time, caption.end_time, start, end) for start, end in local_ranges):
            merged.append(_merged_segment(caption, len(merged), video_id, caption.text))

    merged = sorted(merged, key=lambda segment: (segment.start_time, segment.end_time))
    merged = [
        segment.model_copy(update={"segment_id": f"{video_id}:merged:{index:06d}"})
        for index, segment in enumerate(dedupe_adjacent_segments(merged))
    ]
    return Transcript(video_id=video_id, segments=merged, source=MERGED_SOURCE)


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
            "'pip install resonance-graph[transcription]' or 'pip install faster-whisper'."
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
    if not Path(config.whisper_cpp_model).exists():
        raise AppError(f"WHISPER_CPP_MODEL does not exist: {config.whisper_cpp_model}")

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

    whisper_json = whisper_cpp_json_path_for(video_id, config)
    if not whisper_json.exists():
        raise AppError(
            "whisper.cpp finished, but its JSON output was not found at "
            f"{whisper_json}. Check the whisper.cpp version and output flags."
        )
    return _transcript_from_whisper_cpp_json(whisper_json, video_id)


def _transcript_from_whisper_cpp_json(path: Path, video_id: str) -> Transcript:
    payload = json.loads(path.read_text())
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


def _local_backend(config: AppConfig) -> str:
    backend = (config.local_transcription_backend or config.transcription_backend).lower().strip()
    if backend == "whisper_cpp_metal":
        return "whisper.cpp"
    return backend


def _usable_local_segment(segment: TranscriptSegment) -> bool:
    text = segment.text.strip()
    if len(text) < 2:
        return False
    if len(set(text.lower().split())) <= 1 and len(text.split()) > 5:
        return False
    return True


def _best_caption_overlap(
    local: TranscriptSegment,
    captions: list[TranscriptSegment],
) -> TranscriptSegment | None:
    candidates = [
        (min(local.end_time, caption.end_time) - max(local.start_time, caption.start_time), caption)
        for caption in captions
        if ranges_overlap(local.start_time, local.end_time, caption.start_time, caption.end_time)
    ]
    candidates = [(overlap, caption) for overlap, caption in candidates if overlap > 0]
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _merged_segment(
    source: TranscriptSegment,
    index: int,
    video_id: str,
    text: str,
) -> TranscriptSegment:
    return TranscriptSegment(
        segment_id=f"{video_id}:merged:{index:06d}",
        video_id=video_id,
        start_time=source.start_time,
        end_time=source.end_time,
        text=text.strip(),
        source=MERGED_SOURCE,
    )
