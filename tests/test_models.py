from app.models import EpisodeMetadata, RetrievedChunk, TranscriptChunk, TranscriptSegment


def test_pydantic_models_accept_mvp_fields() -> None:
    episode = EpisodeMetadata(
        video_id="abc123",
        title="Example",
        source_url="https://www.youtube.com/watch?v=abc123",
    )
    segment = TranscriptSegment(
        segment_id="abc123:seg:000001",
        video_id="abc123",
        start_time=1.0,
        end_time=2.0,
        text="Hello world",
    )
    chunk = TranscriptChunk(
        chunk_id="abc123:chunk:000001",
        video_id="abc123",
        ordinal=1,
        text=segment.text,
        start_time=segment.start_time,
        end_time=segment.end_time,
        segment_ids=[segment.segment_id],
        embedding=[0.1, 0.2],
    )
    retrieved = RetrievedChunk(
        chunk_id=chunk.chunk_id,
        video_id=episode.video_id,
        episode_title=episode.title,
        source_url=episode.source_url,
        text=chunk.text,
        start_time=chunk.start_time,
        end_time=chunk.end_time,
        score=0.9,
    )

    assert chunk.segment_ids == [segment.segment_id]
    assert retrieved.episode_title == "Example"
