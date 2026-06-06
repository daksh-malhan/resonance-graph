from app.chunking import build_chunks
from app.models import TranscriptSegment


def _segment(index: int, text: str) -> TranscriptSegment:
    return TranscriptSegment(
        segment_id=f"vid:seg:{index:06d}",
        video_id="vid",
        start_time=float(index * 10),
        end_time=float(index * 10 + 5),
        text=text,
    )


def test_chunking_preserves_timestamps_and_segment_ids() -> None:
    segments = [
        _segment(0, "alpha beta gamma"),
        _segment(1, "delta epsilon zeta"),
        _segment(2, "eta theta iota"),
    ]

    chunks = build_chunks(segments, video_id="vid", chunk_size=40, chunk_overlap=10)

    assert len(chunks) == 2
    assert chunks[0].start_time == 0
    assert chunks[0].end_time == 15
    assert chunks[0].segment_ids == ["vid:seg:000000", "vid:seg:000001"]
    assert chunks[1].segment_ids[0] == "vid:seg:000001"
    assert chunks[1].end_time == 25


def test_chunking_single_long_segment_still_creates_chunk() -> None:
    chunks = build_chunks(
        [_segment(0, "x" * 120)],
        video_id="vid",
        chunk_size=30,
        chunk_overlap=5,
    )

    assert len(chunks) == 1
    assert chunks[0].text == "x" * 120
