# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
pip install -e ".[transcription,dev]"   # full local dev install
docker compose up -d                    # Neo4j (bolt://localhost:7687, neo4j/local-graphrag-password)
ollama pull nomic-embed-text && ollama pull llama3.1:8b

pytest                                  # full suite (offline, no Neo4j/Ollama needed)
pytest tests/test_retrieval.py -q       # single file
pytest tests/test_retrieval.py::test_corpus_overview_question_detection   # single test
ruff check app                          # lint (line-length 100, target py311)
python -m compileall app                # CI's compile gate

resonance <cmd>                         # CLI (Typer). aliases: podcast-graphrag
resonance-web                           # local web app on 127.0.0.1:8766
```

The test suite is fully offline — it stubs Neo4j/Ollama with fakes. `resonance status` checks live deps (yt-dlp, FFmpeg, transcription backend, Ollama, Neo4j); run `resonance setup-db` once before ingesting.

## Architecture

Local-first GraphRAG: approved YouTube media → transcript graph in Neo4j → vector-search RAG with timestamp citations. Everything runs locally (Ollama for embeddings + chat, Neo4j for storage). No external LLM API.

**Layering** — `app/cli.py` (Typer) and `app/web.py` (stdlib `http.server`, no framework) are the two entrypoints; both are thin and delegate to `app/pipeline.py` (single-video) and `app/channel_pipeline.py` (multi-video). The pipeline orchestrates stage modules: `youtube` (yt-dlp) → `audio` (FFmpeg) → `captions`/`transcription` → `chunking` → `ollama` (embed) → `neo4j_store` (persist). Retrieval flows `retrieval` → `neo4j_store.vector_search` → `reranking` → `prompts` → `ollama.chat`.

**Captions-first ingestion is the core design.** `ingest_url_pipeline` fetches metadata + YouTube captions *before* downloading media, so a video becomes searchable fast from captions, then a higher-quality local Whisper transcript is produced and **merged** in the background (`merge_transcripts` prefers local segment text, fills gaps with captions). `transcript_source`/`transcript_status` on `EpisodeMetadata` track which pass a chunk came from (`youtube_caption` → `merged`/`local_whisper`). When editing the pipeline, preserve this fast-path-then-upgrade ordering.

**Two separate job systems** — don't conflate them. `app/background_jobs.py` writes JSON job files to `data/jobs/` and spawns a *detached* `python -m app.worker` subprocess for local transcription that outlives the request. `app/web.py`'s `JobStore` is a separate in-memory, thread-based tracker for foreground web ingestion progress. The web `/api/background-jobs/*` endpoints read the file-based system; `/api/jobs/*` reads the in-memory one.

**Channel pipeline is staged lanes, not whole-video parallelism.** `run_channel_ingest_pipeline` uses one bounded `ThreadPoolExecutor` per stage (download/audio/caption/ingest/local), wired via `future.add_done_callback`, so I/O stages overlap while embedding and graph writes stay constrained. Worker counts come from `pipeline_*_workers` config.

**Transcription backends** — `local_transcription_backend` selects `whisper.cpp` (Metal, preferred on Mac, shells out to `whisper-cli`) or `faster-whisper` (Python). whisper.cpp failures fall back to faster-whisper unless `transcription_backend` explicitly pins whisper.cpp. All produced transcripts are relabeled to a single canonical `source` via `_relabel`.

**Config** — `app/config.py` `AppConfig` (pydantic-settings) is the single config object threaded through every module (it's the most-connected node in the codebase). Loads from env / `.env`, or a YAML/TOML file via `APP_CONFIG_FILE`. All path and numeric fields are validated. `config.ensure_directories()` creates the `data/` tree.

**Neo4j graph** — labels `Source → Episode → {Chunk, TranscriptSegment}`, plus `RoleCandidate → Person`. `Chunk.embedding` backs a vector index (`vector_index_name`, cosine). Role candidates (`app/roles.py`) are evidence-backed and generic: channel/uploader/creator are stored as their real metadata roles; host/guest are only added as `possible_*` when a title/description/intro pattern supports them. Re-ingesting an episode clears and rebuilds its chunks/segments/role-candidates.

**Models** — `app/models.py` pydantic models are the contract between every stage; transcript data moves as `Transcript`/`TranscriptSegment` → `TranscriptChunk` → `RetrievedChunk`. Persistence and HTTP serialization use `model_dump(mode="json")`.

## Conventions

- Errors meant for users raise `AppError` (`app/errors.py`); CLI/web catch it and show the message without a traceback. Use it for actionable failures (missing executable, unreachable service); let unexpected errors propagate.
- Shared helpers live in `app/utils.py` (`ranges_overlap`, `dedupe_adjacent_segments`, `format_timestamp`, `require_executable`, JSON I/O) — reuse rather than re-implementing per module.
- All stages are cache-aware and idempotent: they short-circuit on existing output unless `force=True`. Keep new stages consistent with this.
- `require_executable` resolves binaries from PATH, the venv, and the bundled `imageio-ffmpeg` — don't hardcode binary paths.
