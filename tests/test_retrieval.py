from app.config import AppConfig
from app.models import RetrievedChunk
from app.retrieval import (
    answer_question,
    build_corpus_overview_answer,
    is_corpus_overview_question,
    retrieve_context,
)


class FakeOllama:
    def embed_text(self, text: str) -> list[float]:
        assert text == "What is discussed?"
        return [0.1, 0.2, 0.3]


class FakeStore:
    def __init__(self) -> None:
        self.video_id = None

    def vector_search(
        self,
        question_embedding: list[float],
        top_k: int,
        neighbor_window: int = 0,
        video_id: str | None = None,
    ) -> list[RetrievedChunk]:
        assert question_embedding == [0.1, 0.2, 0.3]
        assert top_k == 4
        assert neighbor_window == 1
        self.video_id = video_id
        return []


def test_retrieve_context_passes_video_filter() -> None:
    store = FakeStore()

    retrieve_context(
        "What is discussed?",
        store,  # type: ignore[arg-type]
        FakeOllama(),  # type: ignore[arg-type]
        AppConfig(retrieval_top_k=4),
        video_id="video-1",
    )

    assert store.video_id == "video-1"


class CorpusStore:
    def vector_search(self, *args, **kwargs):
        raise AssertionError("corpus overview should not run vector search")

    def list_episodes(self) -> list[dict]:
        return [
            {
                "title": "Howard Marks: AI, Debt vs Equity & The Next 40 Years Of Investing",
                "video_id": "one",
                "chunk_count": 12,
                "transcript_status": "merged_ready",
            },
            {
                "title": "The World Bank President On Why Jobs Fix Everything",
                "video_id": "two",
                "chunk_count": 8,
                "transcript_status": "caption_ready",
            },
        ]

    def inspect_episode(self, video_id: str) -> dict | None:
        return None


class NoChatOllama:
    def embed_text(self, text: str) -> list[float]:
        raise AssertionError("corpus overview should not embed the question")

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        raise AssertionError("corpus overview should not call the LLM")


def test_corpus_overview_question_uses_episode_titles_without_llm() -> None:
    result = answer_question(
        "What is the data about?",
        CorpusStore(),  # type: ignore[arg-type]
        NoChatOllama(),  # type: ignore[arg-type]
        AppConfig(),
    )

    assert "2 ingested episode" in result.answer
    assert "investing and markets" in result.answer
    assert "jobs and development" in result.answer
    assert "Howard Marks" in result.answer
    assert result.contexts == []


def test_corpus_overview_question_detection() -> None:
    assert is_corpus_overview_question("What is the data about ?")
    assert is_corpus_overview_question("summarize this dataset")
    assert not is_corpus_overview_question("What does Howard Marks say about debt?")


def test_corpus_overview_does_not_match_ai_inside_brain() -> None:
    answer = build_corpus_overview_answer(
        [
            {
                "title": "Essentials: Psychedelics & Neurostimulation for Brain Rewiring",
                "video_id": "brain-video",
                "chunk_count": 59,
                "transcript_status": "merged_ready",
            },
            {
                "title": "Peptides: The Science, Uses & Safety | Dr. Abud Bakri",
                "video_id": "peptides-video",
                "chunk_count": 935,
                "transcript_status": "caption_ready",
            },
        ]
    )

    assert "neuroscience and brain health" in answer
    assert "medicine and health" in answer
    assert "investing and markets" not in answer


class HostQuestionStore:
    def __init__(self) -> None:
        self.vector_search_called = False

    def vector_search(
        self,
        question_embedding: list[float],
        top_k: int,
        neighbor_window: int = 0,
        video_id: str | None = None,
    ) -> list[RetrievedChunk]:
        self.vector_search_called = True
        return [
            RetrievedChunk(
                chunk_id="one:chunk:000000",
                video_id="one",
                episode_title="Founder Interview x Example Host | Open Podcast",
                episode_channel="Open Podcast",
                episode_uploader="Example Host",
                source_url="https://www.youtube.com/watch?v=one",
                text="Welcome to the podcast.",
                start_time=0,
                end_time=10,
                score=0.9,
            )
        ]


class HostQuestionOllama:
    def embed_text(self, text: str) -> list[float]:
        assert text == "who owns the channel?"
        return [0.4, 0.5, 0.6]

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        assert "YouTube uploader: Example Host" in user_prompt
        assert "YouTube channel: Open Podcast" in user_prompt
        return "The channel metadata names Open Podcast."


def test_host_question_uses_regular_rag_context_not_hardcoded_template() -> None:
    store = HostQuestionStore()

    result = answer_question(
        "who owns the channel?",
        store,  # type: ignore[arg-type]
        HostQuestionOllama(),  # type: ignore[arg-type]
        AppConfig(),
    )

    assert store.vector_search_called is True
    assert "The channel metadata names Open Podcast." in result.answer
    assert result.contexts
