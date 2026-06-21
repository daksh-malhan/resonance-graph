# Graph Report - .  (2026-06-22)

## Corpus Check
- 7 files · ~25,427 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 542 nodes · 1979 edges · 21 communities (16 shown, 5 thin omitted)
- Extraction: 80% EXTRACTED · 20% INFERRED · 0% AMBIGUOUS · INFERRED: 400 edges (avg confidence: 0.51)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Chunking & Channel Items|Chunking & Channel Items]]
- [[_COMMUNITY_Audio Extraction & Errors|Audio Extraction & Errors]]
- [[_COMMUNITY_Retrieval & Corpus Overview|Retrieval & Corpus Overview]]
- [[_COMMUNITY_Transcription & Merge|Transcription & Merge]]
- [[_COMMUNITY_Benchmark Harness|Benchmark Harness]]
- [[_COMMUNITY_Ingestion Pipeline|Ingestion Pipeline]]
- [[_COMMUNITY_CLI Commands|CLI Commands]]
- [[_COMMUNITY_Background Jobs|Background Jobs]]
- [[_COMMUNITY_Web Frontend|Web Frontend]]
- [[_COMMUNITY_YouTube Download & Discovery|YouTube Download & Discovery]]
- [[_COMMUNITY_Docs & Architecture|Docs & Architecture]]
- [[_COMMUNITY_YouTube Captions|YouTube Captions]]
- [[_COMMUNITY_Prompt Building|Prompt Building]]
- [[_COMMUNITY_Neo4j Store|Neo4j Store]]
- [[_COMMUNITY_Ollama Client|Ollama Client]]
- [[_COMMUNITY_Reranking|Reranking]]
- [[_COMMUNITY_Retrieval Tests|Retrieval Tests]]
- [[_COMMUNITY_Package Init|Package Init]]
- [[_COMMUNITY_Bug Report Template|Bug Report Template]]
- [[_COMMUNITY_Code of Conduct|Code of Conduct]]
- [[_COMMUNITY_Package Metadata|Package Metadata]]

## God Nodes (most connected - your core abstractions)
1. `AppConfig` - 147 edges
2. `Neo4jStore` - 64 edges
3. `AppError` - 62 edges
4. `TranscriptChunk` - 50 edges
5. `Transcript` - 46 edges
6. `DownloadResult` - 43 edges
7. `OllamaClient` - 41 edges
8. `RetrievedChunk` - 38 edges
9. `$()` - 32 edges
10. `TranscriptSegment` - 30 edges

## Surprising Connections (you probably didn't know these)
- `Modular Module Boundaries` --semantically_similar_to--> `Modular Local Pipeline Architecture`  [INFERRED] [semantically similar]
  CONTRIBUTING.md → docs/ARCHITECTURE.md
- `Local Web App UI` --semantically_similar_to--> `resonance CLI`  [INFERRED] [semantically similar]
  app/static/index.html → README.md
- `Security Policy` --semantically_similar_to--> `Benchmark Privacy (no transcript upload)`  [INFERRED] [semantically similar]
  SECURITY.md → docs/BENCHMARKING.md
- `DownloadResult` --uses--> `ChannelPipelineEvent`  [INFERRED]
  scripts/benchmark_ingestion_methods.py → app/channel_pipeline.py
- `TranscriptChunk` --uses--> `ChannelPipelineEvent`  [INFERRED]
  scripts/benchmark_ingestion_methods.py → app/channel_pipeline.py

## Import Cycles
- None detected.

## Communities (21 total, 5 thin omitted)

### Community 0 - "Chunking & Channel Items"
Cohesion: 0.08
Nodes (79): ChannelPipelineItem, ChannelPipelineResult, AppConfig, ChannelVideo, build_chunks(), chunk_transcript(), chunks_path_for(), _make_chunk() (+71 more)

### Community 1 - "Audio Extraction & Errors"
Cohesion: 0.10
Nodes (40): extract_audio(), AppConfig, EpisodeMetadata, Path, AppError, User-facing application error with a concise remediation message., Path, TranscriptChunk (+32 more)

### Community 2 - "Retrieval & Corpus Overview"
Cohesion: 0.12
Nodes (26): RagAnswer, RetrievedChunk, answer_question(), build_corpus_overview_answer(), _contains_keyword(), is_corpus_overview_question(), _normalize_question(), AppConfig (+18 more)

### Community 3 - "Transcription & Merge"
Cohesion: 0.18
Nodes (34): _best_caption_overlap(), _local_backend(), local_transcript_path_for(), local_transcription_backend_status(), merge_transcripts(), _merged_segment(), preserve_primary_local_transcript(), AppConfig (+26 more)

### Community 4 - "Benchmark Harness"
Cohesion: 0.14
Nodes (33): aggregate_metrics(), BenchmarkCase, BenchmarkCaseResult, BenchmarkReport, BenchmarkSuite, BenchmarkTimeRange, _display_bool(), _display_optional() (+25 more)

### Community 5 - "Ingestion Pipeline"
Cohesion: 0.23
Nodes (32): ChannelPipelineEvent, Run channel ingestion through bounded stage lanes.      This is pipeline paralle, run_channel_ingest_pipeline(), download_video_stage(), extract_audio_stage(), fetch_caption_stage(), fetch_metadata_stage(), finalize_local_transcript_pipeline() (+24 more)

### Community 6 - "CLI Commands"
Cohesion: 0.15
Nodes (33): ask(), background_jobs(), benchmark(), _check_neo4j(), _config(), _fail(), ingest_channel(), ingest_url() (+25 more)

### Community 7 - "Background Jobs"
Cohesion: 0.14
Nodes (24): enqueue_local_finalization(), job_file_path(), list_jobs(), mark_job_failed(), mark_job_running(), mark_job_succeeded(), Any, AppConfig (+16 more)

### Community 8 - "Web Frontend"
Cohesion: 0.17
Nodes (29): $(), activeJobs, api(), askQuestion(), backgroundJobToProgress(), channelPayload(), clearFiles(), drawGraph() (+21 more)

### Community 9 - "YouTube Download & Discovery"
Cohesion: 0.13
Nodes (27): _channel_video_from_entry(), discover_channel_videos(), _extract_video_id_without_download(), fetch_youtube_metadata(), _find_local_video_path(), _is_short_or_too_short(), load_download_result(), _normalize_channel_videos_url() (+19 more)

### Community 10 - "Docs & Architecture"
Cohesion: 0.09
Nodes (30): Detached Local Transcription Jobs, Stage-Pipelined Channel Ingestion, Extension Points (frames/ocr/vision/entities), Modular Local Pipeline Architecture, Local Lexical/Metadata Reranking, Gold Eval Suite File Format, Local Benchmark Harness, Retrieval/Answer/Citation/Latency Metrics (+22 more)

### Community 11 - "YouTube Captions"
Cohesion: 0.18
Nodes (18): download_caption_text(), extract_youtube_caption_transcript(), _parse_vtt_time(), parse_vtt_transcript(), _preferred_languages(), Any, AppConfig, DownloadResult (+10 more)

### Community 12 - "Prompt Building"
Cohesion: 0.34
Nodes (15): build_answer_prompt(), _escape(), _format_channel_citation(), _format_citation(), _format_creator_citation(), _format_episode_context(), format_retrieval_context(), _format_role_candidates() (+7 more)

### Community 14 - "Ollama Client"
Cohesion: 0.26
Nodes (6): Path, embed_chunks(), _embedding_cache_path(), OllamaClient, AppConfig, Exception

### Community 15 - "Reranking"
Cohesion: 0.44
Nodes (9): _meaningful_terms(), _metadata_text(), _normalized_vector_scores(), _phrase_score(), RetrievedChunk, Rerank vector candidates with local lexical and metadata signals., rerank_chunks(), _rerank_score() (+1 more)

### Community 16 - "Retrieval Tests"
Cohesion: 0.25
Nodes (3): CorpusStore, NoChatOllama, test_corpus_overview_question_uses_episode_titles_without_llm()

## Knowledge Gaps
- **16 isolated node(s):** `HttpUrl`, `output`, `graphData`, `activeJobs`, `stagePercents` (+11 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **5 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `AppConfig` connect `Background Jobs` to `Chunking & Channel Items`, `Audio Extraction & Errors`, `Retrieval & Corpus Overview`, `Transcription & Merge`, `Benchmark Harness`, `Ingestion Pipeline`, `CLI Commands`, `YouTube Download & Discovery`, `YouTube Captions`, `Neo4j Store`, `Ollama Client`, `Retrieval Tests`?**
  _High betweenness centrality (0.320) - this node is a cross-community bridge._
- **Why does `Neo4jStore` connect `Neo4j Store` to `Chunking & Channel Items`, `Audio Extraction & Errors`, `Retrieval & Corpus Overview`, `CLI Commands`, `Background Jobs`, `Ollama Client`?**
  _High betweenness centrality (0.088) - this node is a cross-community bridge._
- **Why does `AppError` connect `Audio Extraction & Errors` to `Chunking & Channel Items`, `Ingestion Pipeline`, `CLI Commands`, `Background Jobs`, `Neo4j Store`, `Ollama Client`?**
  _High betweenness centrality (0.037) - this node is a cross-community bridge._
- **Are the 103 inferred relationships involving `AppConfig` (e.g. with `AppConfig` and `EpisodeMetadata`) actually correct?**
  _`AppConfig` has 103 INFERRED edges - model-reasoned connections that need verification._
- **Are the 25 inferred relationships involving `Neo4jStore` (e.g. with `AppConfig` and `Path`) actually correct?**
  _`Neo4jStore` has 25 INFERRED edges - model-reasoned connections that need verification._
- **Are the 37 inferred relationships involving `AppError` (e.g. with `AppConfig` and `EpisodeMetadata`) actually correct?**
  _`AppError` has 37 INFERRED edges - model-reasoned connections that need verification._
- **Are the 35 inferred relationships involving `TranscriptChunk` (e.g. with `ChannelPipelineEvent` and `ChannelPipelineItem`) actually correct?**
  _`TranscriptChunk` has 35 INFERRED edges - model-reasoned connections that need verification._