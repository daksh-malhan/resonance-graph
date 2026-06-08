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

6. `channel_pipeline.py`
   - Runs channel ingestion through bounded stage lanes instead of whole-video parallelism.
   - Lets downloads, audio extraction, caption lookup, embedding/graph writes, and local fallback transcription overlap safely.
   - Uses separate worker counts from config so heavy stages remain constrained.

7. `roles.py`
   - Extracts generic role candidates from YouTube metadata, titles, descriptions, and transcript intros.
   - Stores channel/uploader/creator as actual metadata roles, not inferred hosts.
   - Adds `possible_host` and `possible_guest` only when reusable evidence patterns support them.

8. `ollama.py`
   - Talks to the local Ollama HTTP API.
   - Creates embeddings for chunks and questions.
   - Calls the configured chat model for answer generation.

9. `neo4j_store.py`
   - Creates constraints, indexes, and the vector index.
   - Upserts `Source`, `Episode`, `TranscriptSegment`, `Chunk`, `RoleCandidate`, and `Person` nodes.
   - Runs vector retrieval and graph overview queries.

10. `retrieval.py`
   - Embeds questions.
   - Retrieves relevant chunks from Neo4j, optionally scoped to one episode.
   - Formats retrieved context.
   - Generates transcript-grounded answers.

11. `web.py` and `app/static/`
   - Provide the local website and JSON endpoints.
   - Reuse the same pipeline modules as the CLI.

12. `background_jobs.py` and `worker.py`
   - Store detached local transcription job state as JSON under `data/jobs`.
   - Run local Whisper, transcript merge, re-chunking, re-embedding, and Neo4j updates after caption-ready ingest returns.
   - Preserve resumability through cached media, audio, captions, local transcripts, chunks, and embeddings.

## Channel Pipelining

Channel ingestion uses stage pipelining rather than launching full video ingests in parallel. A video can leave the download lane and enter audio extraction while another video starts downloading. The default lane sizes are:

- Download lane: `PIPELINE_DOWNLOAD_WORKERS=2`
- Audio lane: `PIPELINE_AUDIO_WORKERS=2`
- Caption lane: `PIPELINE_CAPTION_WORKERS=3`
- Ingest lane: `PIPELINE_INGEST_WORKERS=1`
- Local transcription fallback lane: `PIPELINE_LOCAL_WORKERS=1`

This keeps I/O stages moving while protecting Ollama, Neo4j, and local Whisper from uncontrolled contention.

## Graph Model

```mermaid
erDiagram
  Source ||--o{ Episode : HAS_EPISODE
  Episode ||--o{ Chunk : HAS_CHUNK
  Episode ||--o{ TranscriptSegment : HAS_SEGMENT
  Chunk }o--o{ TranscriptSegment : CONTAINS_SEGMENT
  Episode ||--o{ RoleCandidate : HAS_ROLE_CANDIDATE
  RoleCandidate }o--|| Person : REFERS_TO
```

Role candidates are evidence records, not final truth. A channel/uploader node can answer who published or uploaded a video. Host identity should be answered from `host` or `possible_host` candidates plus their evidence source and confidence.

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
- LLM-backed role extraction that emits validated JSON into the existing `RoleCandidate` model.
- `api.py` for a future FastAPI backend.
- `reranking.py` for retrieval reranking.

The graph can be extended with `Frame`, `Entity`, `Topic`, and `Claim` nodes while preserving the current transcript-first core.
