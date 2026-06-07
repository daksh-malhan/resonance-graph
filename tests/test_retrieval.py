from app.config import AppConfig
from app.models import RetrievedChunk
from app.retrieval import (
    answer_question,
    is_corpus_overview_question,
    is_metadata_identity_question,
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


class MetadataStore:
    def vector_search(self, *args, **kwargs):
        raise AssertionError("metadata identity questions should not run vector search")

    def list_episodes(self) -> list[dict]:
        return [
            {
                "title": "Martin Escobari: Trauma, Chaos & Three Industries Worth $100B | Nikhil Kamath | People by WTF",
                "video_id": "one",
                "channel": "Nikhil Kamath",
                "chunk_count": 114,
            },
            {
                "title": "The World Bank President On Why Jobs Fix Everything | Ajay Banga x Nikhil Kamath | People by WTF",
                "video_id": "two",
                "channel": "Nikhil Kamath",
                "chunk_count": 109,
            },
        ]

    def inspect_episode(self, video_id: str) -> dict | None:
        return {
            "title": "Martin Escobari: Trauma, Chaos & Three Industries Worth $100B | Nikhil Kamath | People by WTF",
            "video_id": video_id,
            "channel": "Nikhil Kamath",
            "chunk_count": 114,
        }


def test_metadata_identity_question_uses_channel_without_llm() -> None:
    result = answer_question(
        "who is the host?",
        MetadataStore(),  # type: ignore[arg-type]
        NoChatOllama(),  # type: ignore[arg-type]
        AppConfig(),
    )

    assert "Nikhil Kamath" in result.answer
    assert "channel owner or publisher" in result.answer
    assert "host/show hint" in result.answer
    assert "Episode channel/owner: Nikhil Kamath" in result.answer
    assert result.contexts == []


def test_scoped_metadata_identity_question_uses_inspect_episode() -> None:
    result = answer_question(
        "who owns the channel?",
        MetadataStore(),  # type: ignore[arg-type]
        NoChatOllama(),  # type: ignore[arg-type]
        AppConfig(),
        video_id="one",
    )

    assert "Nikhil Kamath" in result.answer
    assert "Martin Escobari" in result.answer


def test_metadata_identity_question_detection() -> None:
    assert is_metadata_identity_question("who hosts this podcast?")
    assert is_metadata_identity_question("who owns the channel?")
    assert not is_metadata_identity_question("what does the guest say about debt?")
