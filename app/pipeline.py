from __future__ import annotations

from app.audio import extract_audio
from app.chunking import chunk_transcript
from app.config import AppConfig
from app.models import DownloadResult, TranscriptChunk
from app.neo4j_store import Neo4jStore
from app.ollama import OllamaClient, embed_chunks
from app.transcription import transcribe_audio
from app.youtube import download_youtube_video


def ingest_url_pipeline(
    url: str,
    config: AppConfig,
    force: bool = False,
) -> tuple[DownloadResult, list[TranscriptChunk]]:
    config.ensure_directories()
    download = download_youtube_video(url, config, force=force)
    audio_path = extract_audio(download.episode, config, force=force)
    transcript = transcribe_audio(audio_path, download.episode.video_id, config, force=force)
    chunks = chunk_transcript(transcript, config, force=force)

    ollama = OllamaClient(config)
    ollama.ensure_models()
    embedded_chunks = embed_chunks(chunks, ollama, config, force=force)

    dimension = len(embedded_chunks[0].embedding or []) if embedded_chunks else ollama.embedding_dimension()
    store = Neo4jStore(config)
    try:
        store.setup_schema(dimension)
        store.ingest_episode(download, transcript, embedded_chunks)
    finally:
        store.close()

    return download, embedded_chunks
