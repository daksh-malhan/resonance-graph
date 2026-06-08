# Benchmarking

Resonance Graph includes a local benchmark harness for testing retrieval quality, answer behavior, citation behavior, and latency.

Benchmarks are only meaningful when you provide a gold eval suite: questions plus expected evidence. The harness intentionally does not upload or publish local transcript text.

## Quick Run

Create a benchmark file from the template:

```bash
cp evals/example.yaml evals/my-safe-benchmark.yaml
```

Edit it with questions for videos you are legally allowed to process.

Run retrieval plus answer benchmarking:

```bash
resonance benchmark evals/my-safe-benchmark.yaml --output-dir benchmark-results/latest
```

Run retrieval-only benchmarking:

```bash
resonance benchmark evals/my-safe-benchmark.yaml --retrieval-only
```

Tune retrieval settings:

```bash
resonance benchmark evals/my-safe-benchmark.yaml --top-k 8 --neighbors 1
```

## Ingestion Method Timing

Normal ingestion uses the captions-first pipelined method. The old sequential download-first method is kept only as a timing baseline in the helper script:

```bash
python scripts/benchmark_ingestion_methods.py "https://www.youtube.com/@CHANNEL/videos" --limit 3 --max-resolution 144 --isolate-method-dirs
```

The helper defaults to `--order pipeline-first` so the normal method gets measured before the old baseline can trigger YouTube throttling.

## Eval File Format

```yaml
name: My Benchmark
description: Public-safe benchmark for approved videos.
cases:
  - id: clear-short-id
    question: "What does the guest say about sleep timing?"
    expected_video_id: "VIDEO_ID"
    expected_terms:
      - "sleep"
      - "timing"
    relevant_time_ranges:
      - video_id: "VIDEO_ID"
        start_time: 300
        end_time: 420
```

Fields:

- `id`: Stable case identifier.
- `question`: Question sent to the RAG system.
- `expected_video_id`: Optional expected episode video ID.
- `relevant_chunk_ids`: Optional exact relevant chunk IDs.
- `relevant_time_ranges`: Optional timestamp ranges that should be retrieved.
- `expected_terms`: Terms expected in retrieved context and, when answers are enabled, in the answer.
- `expect_insufficient_evidence`: Set to `true` for negative-control questions.

Use `relevant_time_ranges` when you want citation-sensitive retrieval testing without hard-coding chunk IDs.

## Metrics

The benchmark writes:

- `benchmark.json`
- `benchmark.md`

Default output path:

```text
benchmark-results/latest/
```

Metrics:

- `retrieval_hit_rate`: Fraction of cases where retrieved context hit the expected chunk/time range. If no explicit relevance is provided, full expected-term coverage in retrieved context counts as a hit.
- `mean_reciprocal_rank`: Average reciprocal rank of the first relevant chunk.
- `context_term_coverage`: Average fraction of `expected_terms` found in retrieved context.
- `answer_term_coverage`: Average fraction of `expected_terms` found in generated answers.
- `citation_rate`: Fraction of generated answers containing timestamp-like citations.
- `grounded_token_overlap`: Crude lexical overlap between answer tokens and retrieved context tokens. This is not a substitute for human review, but it is useful for regression tracking.
- `insufficient_evidence_match_rate`: Fraction of negative-control answers that correctly say the transcript does not provide enough evidence.
- `retrieval_latency_ms_avg` and `retrieval_latency_ms_p95`: Retrieval latency.
- `answer_latency_ms_avg` and `answer_latency_ms_p95`: End-to-end answer latency.

## Privacy

`benchmark-results/` is ignored by git by default because reports can reveal private questions, retrieved chunk IDs, timestamps, and optionally generated answers.

Do not publish benchmark suites or reports that reveal private video content unless you have permission.

By default, raw generated answers are not written to the JSON report. To include them locally:

```bash
resonance benchmark evals/my-safe-benchmark.yaml --include-answer-text
```

Avoid `--include-answer-text` for private data.

## Publishing Public Metrics

For public-safe benchmarks:

```bash
resonance benchmark evals/my-public-benchmark.yaml --output-dir docs/benchmarks/public-baseline
git add evals/my-public-benchmark.yaml docs/benchmarks/public-baseline
git commit -m "Add public benchmark baseline"
git push
```

Only publish metrics for videos and questions you are allowed to disclose.

## How To Use Results

If retrieval is weak:

- Increase `--top-k`.
- Try `--neighbors 1`.
- Reduce chunk size or overlap if chunks are too broad.
- Add better expected timestamp ranges to identify where retrieval misses.

If answers are weak but retrieval is strong:

- Tune the answer prompt.
- Increase neighbor context.
- Use a stronger local chat model.
- Add negative-control cases to catch hallucination.

If citation rate is weak:

- Strengthen the citation instruction in the prompt.
- Ensure retrieved chunks include clear start/end timestamps.
