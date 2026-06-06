# Contributing

Thanks for improving Resonance Graph. The project is an MVP, but contributions should keep the codebase reliable, local-first, and easy to extend.

## Priorities

- Keep the transcript ingestion and RAG path working end to end.
- Prefer small, focused changes over broad rewrites.
- Preserve local processing by default.
- Keep YouTube ingestion inside the legal boundary described in the README.
- Add tests for behavior changes when practical.

## Local Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[transcription,dev]"
cp .env.example .env
docker compose up -d
resonance setup-db
```

## Checks

Run before opening a pull request:

```bash
pytest
python -m compileall app
```

For external service readiness:

```bash
resonance status
```

## Pull Request Guidelines

- Explain the user-facing behavior change.
- Mention any new configuration values.
- Include test coverage or explain why the change is not covered.
- Do not commit `.env`, downloaded media, transcripts, embeddings, model files, or local Neo4j data.
- Do not add code intended to bypass DRM, paywalls, private videos, login-only videos, or platform protections.

## Architecture Notes

The current system is intentionally modular:

- `youtube.py` owns discovery/download.
- `audio.py` owns audio extraction.
- `transcription.py` owns local transcription backends.
- `chunking.py` owns timestamp-preserving chunk construction.
- `ollama.py` owns local embeddings and chat calls.
- `neo4j_store.py` owns graph schema, ingestion, and retrieval queries.
- `retrieval.py` owns the RAG answer flow.
- `web.py` and `static/` own the local UI.

Future features should fit into these boundaries instead of making the pipeline monolithic.
