from app.models import RetrievedChunk, RoleCandidate
from app.prompts import SYSTEM_PROMPT, build_answer_prompt, format_retrieval_context


def test_retrieval_context_includes_citations() -> None:
    chunk = RetrievedChunk(
        chunk_id="vid:chunk:000000",
        video_id="vid",
        episode_title="Example Episode",
        episode_channel="Example Channel",
        episode_uploader="Example Uploader",
        episode_creator="Example Creator",
        episode_role_candidates=[
            RoleCandidate(
                name="Example Host",
                role="possible_host",
                confidence=0.72,
                evidence_source="title_pattern",
                evidence_text="Example Guest x Example Host | Example Show",
            )
        ],
        source_url="https://www.youtube.com/watch?v=vid",
        text="The host explains local embeddings.",
        start_time=65,
        end_time=95,
        score=0.8123,
    )

    context = format_retrieval_context([chunk])

    assert "Example Episode" in context
    assert "Example Channel" in context
    assert "<episode_context>" in context
    assert "Episode title: Example Episode" in context
    assert "YouTube channel: Example Channel" in context
    assert "YouTube uploader: Example Uploader" in context
    assert "YouTube creator: Example Creator" in context
    assert "Role candidate: Example Host as possible_host from title_pattern" in context
    assert "01:05-01:35" in context
    assert "<transcript_source>local_whisper</transcript_source>" in context
    assert "The host explains local embeddings." in context


def test_prompt_requires_supported_answer() -> None:
    prompt = build_answer_prompt("What is discussed?", [])

    assert "Use only the provided context" in SYSTEM_PROMPT
    assert "does not provide enough evidence" in SYSTEM_PROMPT
    assert "Question:" in prompt
    assert "No transcript context was retrieved." in prompt


def test_prompt_allows_title_context_without_treating_it_as_transcript_evidence() -> None:
    prompt = build_answer_prompt(
        "Who is in this podcast?",
        [
            RetrievedChunk(
                chunk_id="vid:chunk:000000",
                video_id="vid",
                episode_title="Dr. Jane Smith on Memory and Learning",
                episode_channel="Open Learning Channel",
                episode_uploader="Open Learning Uploader",
                episode_role_candidates=[
                    RoleCandidate(
                        name="Dr. Jane Smith",
                        role="possible_guest",
                        confidence=0.55,
                        evidence_source="title_pattern",
                        evidence_text="Dr. Jane Smith on Memory and Learning",
                    )
                ],
                source_url="https://www.youtube.com/watch?v=vid",
                text="Today we discuss how memory consolidation works.",
                start_time=10,
                end_time=40,
                score=0.9,
                transcript_source="merged",
            )
        ],
    )

    assert "Dr. Jane Smith on Memory and Learning" in prompt
    assert "Open Learning Channel" in prompt
    assert "Open Learning Uploader" in prompt
    assert "Episode title: Dr. Jane Smith on Memory and Learning" in prompt
    assert "YouTube channel/uploader/creator metadata" in prompt
    assert "YouTube uploader: Open Learning Uploader" in prompt
    assert "Role candidate: Dr. Jane Smith as possible_guest from title_pattern" in prompt
