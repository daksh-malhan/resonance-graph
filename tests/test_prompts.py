from app.models import RetrievedChunk
from app.prompts import SYSTEM_PROMPT, build_answer_prompt, format_retrieval_context


def test_retrieval_context_includes_citations() -> None:
    chunk = RetrievedChunk(
        chunk_id="vid:chunk:000000",
        video_id="vid",
        episode_title="Example Episode",
        source_url="https://www.youtube.com/watch?v=vid",
        text="The host explains local embeddings.",
        start_time=65,
        end_time=95,
        score=0.8123,
    )

    context = format_retrieval_context([chunk])

    assert "Example Episode" in context
    assert "01:05-01:35" in context
    assert "The host explains local embeddings." in context


def test_prompt_requires_supported_answer() -> None:
    prompt = build_answer_prompt("What is discussed?", [])

    assert "Use only the provided context" in SYSTEM_PROMPT
    assert "does not provide enough evidence" in SYSTEM_PROMPT
    assert "Question:" in prompt
    assert "No transcript context was retrieved." in prompt
