from app.benchmark import (
    BenchmarkCase,
    BenchmarkTimeRange,
    aggregate_metrics,
    score_case,
    term_coverage,
    token_overlap,
)
from app.models import RetrievedChunk


def _chunk(
    chunk_id: str = "video-1-chunk-3",
    start_time: float = 30,
    end_time: float = 60,
    text: str = "The guest explains sleep timing and morning sunlight.",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        video_id="video-1",
        episode_title="Episode",
        source_url="https://www.youtube.com/watch?v=video-1",
        text=text,
        start_time=start_time,
        end_time=end_time,
        score=0.91,
    )


def test_score_case_hits_relevant_time_range() -> None:
    case = BenchmarkCase(
        id="sleep",
        question="What helps sleep timing?",
        expected_video_id="video-1",
        expected_terms=["sleep", "sunlight"],
        relevant_time_ranges=[BenchmarkTimeRange(video_id="video-1", start_time=40, end_time=45)],
    )

    result = score_case(
        case,
        [_chunk()],
        retrieval_latency_ms=25,
        answer_latency_ms=100,
        answer="The transcript says morning sunlight helps sleep timing at 0:30-1:00.",
    )

    assert result.retrieval_hit is True
    assert result.first_relevant_rank == 1
    assert result.reciprocal_rank == 1.0
    assert result.context_term_coverage == 1.0
    assert result.answer_term_coverage == 1.0
    assert result.answer_has_citation is True


def test_score_case_uses_term_coverage_when_no_explicit_relevance() -> None:
    case = BenchmarkCase(
        id="terms-only",
        question="What does it say about sunlight?",
        expected_terms=["sunlight"],
    )

    result = score_case(case, [_chunk()], retrieval_latency_ms=10)

    assert result.retrieval_hit is True
    assert result.first_relevant_rank is None
    assert result.context_term_coverage == 1.0


def test_aggregate_metrics() -> None:
    case = BenchmarkCase(
        id="sleep",
        question="What helps sleep timing?",
        expected_terms=["sleep"],
    )
    results = [
        score_case(case, [_chunk()], retrieval_latency_ms=10, answer="Sleep is discussed at 0:30."),
        score_case(case, [], retrieval_latency_ms=30, answer="Not enough evidence."),
    ]

    metrics = aggregate_metrics(results)

    assert metrics["retrieval_hit_rate"] == 0.5
    assert metrics["retrieval_latency_ms_avg"] == 20.0
    assert metrics["citation_rate"] == 0.5


def test_text_metrics() -> None:
    assert term_coverage(["sleep", "light"], "Sleep and morning light") == 1.0
    assert token_overlap("Morning light supports sleep.", "sleep timing with morning light") > 0.5
