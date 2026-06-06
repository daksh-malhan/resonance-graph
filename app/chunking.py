from __future__ import annotations

import logging
from pathlib import Path

from app.config import AppConfig
from app.models import Transcript, TranscriptChunk, TranscriptSegment
from app.utils import read_json, write_json

logger = logging.getLogger(__name__)


def chunks_path_for(video_id: str, config: AppConfig) -> Path:
    return config.chunk_output_dir / f"{video_id}.chunks.json"


def chunk_transcript(
    transcript: Transcript,
    config: AppConfig,
    force: bool = False,
) -> list[TranscriptChunk]:
    config.chunk_output_dir.mkdir(parents=True, exist_ok=True)
    output_path = chunks_path_for(transcript.video_id, config)
    if output_path.exists() and not force:
        logger.info("Using cached chunks %s", output_path)
        payload = read_json(output_path)
        return [TranscriptChunk.model_validate(item) for item in payload]

    chunks = build_chunks(
        transcript.segments,
        video_id=transcript.video_id,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    write_json(output_path, chunks)
    return chunks


def build_chunks(
    segments: list[TranscriptSegment],
    video_id: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[TranscriptChunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be zero or greater")

    chunks: list[TranscriptChunk] = []
    current: list[TranscriptSegment] = []
    current_len = 0

    for segment in segments:
        text_len = len(segment.text)
        if current and current_len + 1 + text_len > chunk_size:
            chunks.append(_make_chunk(video_id, len(chunks), current))
            current = _overlap_tail(current, chunk_overlap)
            current_len = _segments_text_len(current)

        current.append(segment)
        current_len += text_len + (1 if current_len else 0)

    if current:
        chunks.append(_make_chunk(video_id, len(chunks), current))

    return chunks


def _make_chunk(video_id: str, ordinal: int, segments: list[TranscriptSegment]) -> TranscriptChunk:
    text = " ".join(segment.text.strip() for segment in segments if segment.text.strip()).strip()
    return TranscriptChunk(
        chunk_id=f"{video_id}:chunk:{ordinal:06d}",
        video_id=video_id,
        ordinal=ordinal,
        text=text,
        start_time=min(segment.start_time for segment in segments),
        end_time=max(segment.end_time for segment in segments),
        segment_ids=[segment.segment_id for segment in segments],
    )


def _overlap_tail(segments: list[TranscriptSegment], chunk_overlap: int) -> list[TranscriptSegment]:
    if chunk_overlap == 0:
        return []

    selected: list[TranscriptSegment] = []
    total = 0
    for segment in reversed(segments):
        selected.append(segment)
        total += len(segment.text) + (1 if total else 0)
        if total >= chunk_overlap:
            break
    return list(reversed(selected))


def _segments_text_len(segments: list[TranscriptSegment]) -> int:
    if not segments:
        return 0
    return sum(len(segment.text) for segment in segments) + len(segments) - 1
