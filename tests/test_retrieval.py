from app.config import AppConfig
from app.models import RetrievedChunk
from app.retrieval import retrieve_context


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
