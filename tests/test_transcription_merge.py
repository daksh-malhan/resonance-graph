from app.models import Transcript, TranscriptSegment
from app.transcription import merge_transcripts


def _segment(source: str, index: int, start: float, end: float, text: str) -> TranscriptSegment:
    return TranscriptSegment(
        segment_id=f"vid:{source}:{index}",
        video_id="vid",
        start_time=start,
        end_time=end,
        text=text,
        source=source,
    )


def test_merge_prefers_usable_local_text() -> None:
    captions = Transcript(
        video_id="vid",
        source="youtube_caption",
        segments=[_segment("youtube_caption", 0, 0, 5, "caption text")],
    )
    local = Transcript(
        video_id="vid",
        source="local_whisper",
        segments=[_segment("local_whisper", 0, 0, 5, "better local text")],
    )

    merged = merge_transcripts(captions, local, "vid")

    assert merged.source == "merged"
    assert merged.segments[0].text == "better local text"
    assert merged.segments[0].source == "merged"


def test_merge_preserves_caption_when_local_has_gap() -> None:
    captions = Transcript(
        video_id="vid",
        source="youtube_caption",
        segments=[
            _segment("youtube_caption", 0, 0, 5, "caption first"),
            _segment("youtube_caption", 1, 10, 15, "caption gap"),
        ],
    )
    local = Transcript(
        video_id="vid",
        source="local_whisper",
        segments=[_segment("local_whisper", 0, 0, 5, "local first")],
    )

    merged = merge_transcripts(captions, local, "vid")

    assert [segment.text for segment in merged.segments] == ["local first", "caption gap"]
