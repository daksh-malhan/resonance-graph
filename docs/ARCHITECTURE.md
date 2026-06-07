# Architecture

Resonance Graph is built as a modular local pipeline. The MVP focuses on transcript-first GraphRAG and keeps each stage replaceable.

## Pipeline

1. `youtube.py`
   - Validates approved YouTube input at the tool boundary.
   - Downloads videos with `yt-dlp`.
   - Stores source metadata and info JSON.
   - Discovers long-form channel videos while excluding Shorts.

2. `audio.py`
   - Extracts normalized 16 kHz mono WAV audio.
   - Uses system FFmpeg or the `imageio-ffmpeg` fallback.
   - Reuses cached audio unless force mode is enabled.

3. `transcription.py`
   - Transcribes audio locally.
   - Prefers `whisper.cpp` with Metal on Mac when configured.
   - Uses `faster-whisper` as the fallback local backend.
   - Preserves timestamped segment boundaries.

4. `captions.py`
   - Extracts legally available YouTube captions from public metadata.
   - Prefers manual English captions, then English auto-captions.
   - Parses VTT captions into timestamped transcript segments.

5. `chunking.py`
   - Groups transcript segments into retrieval chunks.
   - Preserves chunk start/end timestamps and source segment IDs.
   - Writes chunk JSON for debugging and cache reuse.

6. `ollama.py`
   - Talks to the local Ollama HTTP API.
   - Creates embeddings for chunks and questions.
   - Calls the configured chat model for answer generation.

7. `neo4j_store.py`
   - Creates constraints, indexes, and the vector index.
   - Upserts `Source`, `Episode`, `TranscriptSegment`, and `Chunk` nodes.
   - Runs vector retrieval and graph overview queries.

8. `retrieval.py`
   - Embeds questions.
   - Retrieves relevant chunks from Neo4j, optionally scoped to one episode.
   - Formats retrieved context.
   - Generates transcript-grounded answers.

9. `web.py` and `app/static/`
   - Provide the local website and JSON endpoints.
   - Reuse the same pipeline modules as the CLI.

10. `background_jobs.py` and `worker.py`
   - Store detached local transcription job state as JSON under `data/jobs`.
   - Run local Whisper, transcript merge, re-chunking, re-embedding, and Neo4j updates after caption-ready ingest returns.
   - Preserve resumability through cached media, audio, captions, local transcripts, chunks, and embeddings.

## Graph Model

```mermaid
erDiagram
  Source ||--o{ Episode : HAS_EPISODE
  Episode ||--o{ Chunk : HAS_CHUNK
  Episode ||--o{ TranscriptSegment : HAS_SEGMENT
  Chunk }o--o{ TranscriptSegment : CONTAINS_SEGMENT
```

## Idempotency

The pipeline is designed to resume:

- Downloads are protected by a `yt-dlp` archive.
- Audio, transcripts, chunks, and embeddings are cached on disk.
- Detached local transcription jobs are cached under `data/jobs`.
- Neo4j writes use `MERGE`.
- Constraints prevent duplicate graph entities.

## Extension Points

Near-term extensions should add modules rather than overloading existing ones:

- `frames.py` for frame extraction.
- `ocr.py` for frame text.
- `vision.py` for local vision captions.
- `entities.py` for entity/topic/claim extraction.
- `api.py` for a future FastAPI backend.
- `reranking.py` for retrieval reranking.

The graph can be extended with `Frame`, `Entity`, `Topic`, and `Claim` nodes while preserving the current transcript-first core.
