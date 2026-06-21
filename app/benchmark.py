from __future__ import annotations

import json
import re
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from app.config import AppConfig
from app.models import RetrievedChunk
from app.neo4j_store import Neo4jStore
from app.ollama import OllamaClient
from app.retrieval import answer_question, retrieve_context
from app.utils import format_timestamp, ranges_overlap


TIMESTAMP_RE = re.compile(r"\b(?:(?:\d{1,2}:)?\d{1,2}:\d{2})\b")
TOKEN_RE = re.compile(r"[a-z0-9']+")


class BenchmarkTimeRange(BaseModel):
    video_id: str | None = None
    start_time: float
    end_time: float


class BenchmarkCase(BaseModel):
    id: str
    question: str
    expected_video_id: str | None = None
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    relevant_time_ranges: list[BenchmarkTimeRange] = Field(default_factory=list)
    expected_terms: list[str] = Field(default_factory=list)
    expect_insufficient_evidence: bool = False


class BenchmarkSuite(BaseModel):
    name: str
    description: str | None = None
    cases: list[BenchmarkCase]


class BenchmarkCaseResult(BaseModel):
    id: str
    retrieval_hit: bool
    first_relevant_rank: int | None
    reciprocal_rank: float
    context_term_coverage: float
    answer_term_coverage: float | None = None
    answer_has_citation: bool | None = None
    insufficient_evidence_match: bool | None = None
    grounded_token_overlap: float | None = None
    retrieval_latency_ms: int
    answer_latency_ms: int | None = None
    retrieved_count: int
    top_score: float | None = None
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    retrieved_timestamps: list[str] = Field(default_factory=list)
    answer: str | None = None


class BenchmarkReport(BaseModel):
    suite_name: str
    generated_at: str
    top_k: int
    neighbor_window: int
    include_answers: bool
    case_count: int
    metrics: dict[str, float]
    cases: list[BenchmarkCaseResult]


def load_benchmark_suite(path: Path) -> BenchmarkSuite:
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(raw)
    else:
        payload = yaml.safe_load(raw)
    return BenchmarkSuite.model_validate(payload)


def run_benchmark(
    suite: BenchmarkSuite,
    store: Neo4jStore,
    ollama: OllamaClient,
    config: AppConfig,
    top_k: int | None = None,
    neighbor_window: int = 0,
    include_answers: bool = True,
    include_answer_text: bool = False,
) -> BenchmarkReport:
    effective_top_k = top_k or config.retrieval_top_k
    results: list[BenchmarkCaseResult] = []

    for case in suite.cases:
        retrieval_started = time.perf_counter()
        contexts = retrieve_context(
            case.question,
            store,
            ollama,
            config,
            top_k=effective_top_k,
            neighbor_window=neighbor_window,
        )
        retrieval_latency_ms = round((time.perf_counter() - retrieval_started) * 1000)

        answer = None
        answer_latency_ms = None
        if include_answers:
            answer_started = time.perf_counter()
            rag_answer = answer_question(
                case.question,
                store,
                ollama,
                config,
                top_k=effective_top_k,
                neighbor_window=neighbor_window,
            )
            answer_latency_ms = round((time.perf_counter() - answer_started) * 1000)
            answer = rag_answer.answer

        results.append(
            score_case(
                case,
                contexts,
                retrieval_latency_ms=retrieval_latency_ms,
                answer_latency_ms=answer_latency_ms,
                answer=answer,
                include_answer_text=include_answer_text,
            )
        )

    return BenchmarkReport(
        suite_name=suite.name,
        generated_at=datetime.now(UTC).isoformat(),
        top_k=effective_top_k,
        neighbor_window=neighbor_window,
        include_answers=include_answers,
        case_count=len(results),
        metrics=aggregate_metrics(results),
        cases=results,
    )


def score_case(
    case: BenchmarkCase,
    contexts: list[RetrievedChunk],
    retrieval_latency_ms: int,
    answer_latency_ms: int | None = None,
    answer: str | None = None,
    include_answer_text: bool = False,
) -> BenchmarkCaseResult:
    first_rank = first_relevant_rank(case, contexts)
    context_text = "\n".join(chunk.text for chunk in contexts)
    context_term_coverage = term_coverage(case.expected_terms, context_text)
    retrieval_hit = first_rank is not None
    if not has_explicit_relevance(case) and case.expected_terms:
        retrieval_hit = context_term_coverage >= 1.0

    answer_term_coverage = None
    answer_has_citation = None
    insufficient_evidence_match = None
    grounded_token_overlap = None
    if answer is not None:
        answer_term_coverage = term_coverage(case.expected_terms, answer)
        answer_has_citation = bool(TIMESTAMP_RE.search(answer))
        insufficient_evidence_match = (
            _looks_like_insufficient_evidence(answer)
            if case.expect_insufficient_evidence
            else None
        )
        grounded_token_overlap = token_overlap(answer, context_text)

    return BenchmarkCaseResult(
        id=case.id,
        retrieval_hit=retrieval_hit,
        first_relevant_rank=first_rank,
        reciprocal_rank=(1.0 / first_rank) if first_rank else 0.0,
        context_term_coverage=context_term_coverage,
        answer_term_coverage=answer_term_coverage,
        answer_has_citation=answer_has_citation,
        insufficient_evidence_match=insufficient_evidence_match,
        grounded_token_overlap=grounded_token_overlap,
        retrieval_latency_ms=retrieval_latency_ms,
        answer_latency_ms=answer_latency_ms,
        retrieved_count=len(contexts),
        top_score=contexts[0].score if contexts else None,
        retrieved_chunk_ids=[chunk.chunk_id for chunk in contexts],
        retrieved_timestamps=[
            f"{chunk.video_id}:{format_timestamp(chunk.start_time)}-{format_timestamp(chunk.end_time)}"
            for chunk in contexts
        ],
        answer=answer if include_answer_text else None,
    )


def aggregate_metrics(results: list[BenchmarkCaseResult]) -> dict[str, float]:
    if not results:
        return {}

    retrieval_latencies = [result.retrieval_latency_ms for result in results]
    answer_latencies = [
        result.answer_latency_ms for result in results if result.answer_latency_ms is not None
    ]
    citation_values = [
        1.0 if result.answer_has_citation else 0.0
        for result in results
        if result.answer_has_citation is not None
    ]
    grounded_values = [
        result.grounded_token_overlap
        for result in results
        if result.grounded_token_overlap is not None
    ]
    answer_term_values = [
        result.answer_term_coverage
        for result in results
        if result.answer_term_coverage is not None
    ]
    insufficient_values = [
        1.0 if result.insufficient_evidence_match else 0.0
        for result in results
        if result.insufficient_evidence_match is not None
    ]

    metrics = {
        "retrieval_hit_rate": _mean(1.0 if result.retrieval_hit else 0.0 for result in results),
        "mean_reciprocal_rank": _mean(result.reciprocal_rank for result in results),
        "context_term_coverage": _mean(result.context_term_coverage for result in results),
        "retrieval_latency_ms_avg": _mean(retrieval_latencies),
        "retrieval_latency_ms_p95": _percentile(retrieval_latencies, 0.95),
    }
    if answer_latencies:
        metrics["answer_latency_ms_avg"] = _mean(answer_latencies)
        metrics["answer_latency_ms_p95"] = _percentile(answer_latencies, 0.95)
    if citation_values:
        metrics["citation_rate"] = _mean(citation_values)
    if grounded_values:
        metrics["grounded_token_overlap"] = _mean(grounded_values)
    if answer_term_values:
        metrics["answer_term_coverage"] = _mean(answer_term_values)
    if insufficient_values:
        metrics["insufficient_evidence_match_rate"] = _mean(insufficient_values)
    return {key: round(value, 4) for key, value in metrics.items()}


def first_relevant_rank(case: BenchmarkCase, contexts: list[RetrievedChunk]) -> int | None:
    if not has_explicit_relevance(case):
        return None
    for index, chunk in enumerate(contexts, start=1):
        if is_relevant(case, chunk):
            return index
    return None


def is_relevant(case: BenchmarkCase, chunk: RetrievedChunk) -> bool:
    if case.expected_video_id and chunk.video_id != case.expected_video_id:
        return False
    if chunk.chunk_id in case.relevant_chunk_ids:
        return True
    for relevant_range in case.relevant_time_ranges:
        if relevant_range.video_id and chunk.video_id != relevant_range.video_id:
            continue
        if ranges_overlap(
            chunk.start_time,
            chunk.end_time,
            relevant_range.start_time,
            relevant_range.end_time,
        ):
            return True
    return False


def has_explicit_relevance(case: BenchmarkCase) -> bool:
    return bool(case.relevant_chunk_ids or case.relevant_time_ranges)


def term_coverage(terms: list[str], text: str) -> float:
    if not terms:
        return 1.0
    normalized = text.lower()
    hits = sum(1 for term in terms if term.lower() in normalized)
    return hits / len(terms)


def token_overlap(answer: str, context: str) -> float:
    answer_tokens = {
        token for token in TOKEN_RE.findall(answer.lower()) if len(token) >= 4 and token not in _STOPWORDS
    }
    if not answer_tokens:
        return 1.0
    context_tokens = set(TOKEN_RE.findall(context.lower()))
    return len(answer_tokens & context_tokens) / len(answer_tokens)


def write_report(report: BenchmarkReport, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "benchmark.json"
    markdown_path = output_dir / "benchmark.md"
    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    markdown_path.write_text(format_markdown_report(report), encoding="utf-8")
    return json_path, markdown_path


def format_markdown_report(report: BenchmarkReport) -> str:
    lines = [
        f"# Benchmark: {report.suite_name}",
        "",
        f"- Generated: `{report.generated_at}`",
        f"- Cases: `{report.case_count}`",
        f"- Top K: `{report.top_k}`",
        f"- Neighbor window: `{report.neighbor_window}`",
        f"- Answers included: `{report.include_answers}`",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key, value in report.metrics.items():
        lines.append(f"| `{key}` | `{value}` |")

    lines.extend(
        [
            "",
            "## Cases",
            "",
            "| ID | Hit | Rank | MRR | Context Terms | Answer Terms | Citations | Retrieval ms | Answer ms |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for result in report.cases:
        lines.append(
            "| "
            f"`{result.id}` | "
            f"`{result.retrieval_hit}` | "
            f"`{result.first_relevant_rank or ''}` | "
            f"`{round(result.reciprocal_rank, 4)}` | "
            f"`{round(result.context_term_coverage, 4)}` | "
            f"`{_display_optional(result.answer_term_coverage)}` | "
            f"`{_display_bool(result.answer_has_citation)}` | "
            f"`{result.retrieval_latency_ms}` | "
            f"`{result.answer_latency_ms or ''}` |"
        )
    lines.append("")
    return "\n".join(lines)


def _looks_like_insufficient_evidence(answer: str) -> bool:
    normalized = answer.lower()
    markers = [
        "does not provide enough evidence",
        "not enough evidence",
        "does not contain",
        "cannot answer",
        "not supported",
    ]
    return any(marker in normalized for marker in markers)


def _mean(values: Any) -> float:
    materialized = list(values)
    if not materialized:
        return 0.0
    return float(statistics.fmean(materialized))


def _percentile(values: list[int | float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * percentile))
    return float(ordered[index])


def _display_optional(value: float | None) -> str:
    return "" if value is None else str(round(value, 4))


def _display_bool(value: bool | None) -> str:
    return "" if value is None else str(value)


_STOPWORDS = {
    "about",
    "after",
    "also",
    "answer",
    "because",
    "been",
    "before",
    "being",
    "context",
    "from",
    "have",
    "into",
    "only",
    "that",
    "their",
    "there",
    "these",
    "this",
    "through",
    "transcript",
    "using",
    "video",
    "what",
    "when",
    "where",
    "which",
    "with",
    "would",
}
