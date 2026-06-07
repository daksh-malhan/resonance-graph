from __future__ import annotations

import logging
from collections.abc import Callable

from app.audio import extract_audio
from app.captions import extract_youtube_caption_transcript
from app.chunking import chunk_transcript
from app.config import AppConfig
from app.models import DownloadResult, Transcript, TranscriptChunk
from app.neo4j_store import Neo4jStore
from app.ollama import OllamaClient, embed_chunks
from app.transcription import merge_transcripts, transcribe_audio, write_primary_transcript
from app.youtube import download_youtube_video

logger = logging.getLogger(__name__)
StageCallback = Callable[[str, str], None]


def ingest_url_pipeline(
    url: str,
    config: AppConfig,
    force: bool = False,
    stage_callback: StageCallback | None = None,
) -> tuple[DownloadResult, list[TranscriptChunk]]:
    config.ensure_directories()
    _stage(stage_callback, "downloading", "Downloading video and metadata")
    download = download_youtube_video(url, config, force=force)

    _stage(stage_callback, "extracting_audio", "Extracting normalized audio")
    audio_path = extract_audio(download.episode, config, force=force)

    caption_transcript: Transcript | None = None
    if config.transcript_fast_path.lower().strip() == "youtube_captions":
        _stage(stage_callback, "fetching_captions", "Checking for YouTube captions")
        caption_transcript = extract_youtube_caption_transcript(download, config, force=force)

    final_download = download
    final_chunks: list[TranscriptChunk] = []
    if caption_transcript and caption_transcript.segments:
        _stage(stage_callback, "caption_ingesting", "Ingesting caption transcript")
        write_primary_transcript(caption_transcript, config)
        final_download, final_chunks = _ingest_transcript(
            download,
            caption_transcript,
            config,
            force=force,
            transcript_source="youtube_caption",
            transcript_status="caption_ready",
            stage_callback=stage_callback,
        )

    if caption_transcript and not config.background_local_transcription:
        _stage(stage_callback, "complete", "Caption transcript ingested")
        return final_download, final_chunks

    _stage(stage_callback, "local_transcribing", "Running local Whisper transcription")
    try:
        local_transcript = transcribe_audio(
            audio_path,
            download.episode.video_id,
            config,
            force=force,
        )
    except Exception:
        if caption_transcript:
            logger.exception("Local transcription failed; keeping caption transcript")
            _stage(
                stage_callback,
                "complete",
                "Caption transcript kept because local transcription failed",
            )
            return final_download, final_chunks
        raise

    if caption_transcript:
        _stage(stage_callback, "merging_transcripts", "Merging captions with local transcript")
        transcript = merge_transcripts(caption_transcript, local_transcript, download.episode.video_id)
        transcript_source = "merged"
        transcript_status = "merged_ready"
    else:
        transcript = local_transcript
        transcript_source = "local_whisper"
        transcript_status = "local_ready"

    write_primary_transcript(transcript, config)
    final_download, final_chunks = _ingest_transcript(
        download,
        transcript,
        config,
        force=force,
        transcript_source=transcript_source,
        transcript_status=transcript_status,
        stage_callback=stage_callback,
    )
    _stage(stage_callback, "complete", "Video ingested")
    return final_download, final_chunks


def _ingest_transcript(
    download: DownloadResult,
    transcript: Transcript,
    config: AppConfig,
    force: bool,
    transcript_source: str,
    transcript_status: str,
    stage_callback: StageCallback | None = None,
) -> tuple[DownloadResult, list[TranscriptChunk]]:
    _stage(stage_callback, "chunking", "Building overlapping transcript chunks")
    chunks = chunk_transcript(transcript, config, force=force)

    _stage(stage_callback, "embedding_chunks", "Embedding transcript chunks")
    ollama = OllamaClient(config)
    ollama.ensure_models()
    embedded_chunks = embed_chunks(chunks, ollama, config, force=force)

    dimension = len(embedded_chunks[0].embedding or []) if embedded_chunks else ollama.embedding_dimension()
    updated_download = download.model_copy(
        update={
            "episode": download.episode.model_copy(
                update={
                    "transcript_source": transcript_source,
                    "transcript_status": transcript_status,
                }
            )
        }
    )

    _stage(stage_callback, "writing_graph", "Writing transcript graph to Neo4j")
    store = Neo4jStore(config)
    try:
        store.setup_schema(dimension)
        store.ingest_episode(updated_download, transcript, embedded_chunks)
    finally:
        store.close()

    return updated_download, embedded_chunks


def _stage(callback: StageCallback | None, stage: str, message: str) -> None:
    logger.info("%s: %s", stage, message)
    if callback:
        callback(stage, message)
