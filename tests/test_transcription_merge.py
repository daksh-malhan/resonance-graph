import json
from pathlib import Path

from app.config import AppConfig
from app.models import Transcript, TranscriptSegment
from app.transcription import (
    local_transcript_path_for,
    merge_transcripts,
    preserve_primary_local_transcript,
    transcribe_audio,
    transcript_path_for,
    whisper_cpp_json_path_for,
)
from app.utils import write_json


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


def test_preserve_primary_local_transcript_before_caption_overwrite(tmp_path: Path) -> None:
    config = AppConfig(
        youtube_download_dir=tmp_path / "youtube",
        audio_output_dir=tmp_path / "audio",
        transcript_output_dir=tmp_path / "transcripts",
        chunk_output_dir=tmp_path / "chunks",
        embedding_cache_dir=tmp_path / "embeddings",
        model_cache_dir=tmp_path / "models",
    )
    local = Transcript(
        video_id="vid",
        source="local_whisper",
        segments=[_segment("local_whisper", 0, 0, 5, "local text")],
    )
    write_json(transcript_path_for("vid", config), local)

    preserved = preserve_primary_local_transcript("vid", config)

    assert preserved is not None
    assert local_transcript_path_for("vid", config).exists()
    assert preserved.source == "local_whisper"
    assert preserved.segments[0].text == "local text"


def test_whisper_cpp_json_path_preserves_whispercpp_suffix(tmp_path: Path) -> None:
    config = AppConfig(transcript_output_dir=tmp_path / "transcripts")

    assert whisper_cpp_json_path_for("vid", config).name == "vid.whispercpp.json"


def test_transcribe_audio_imports_cached_whisper_cpp_json(tmp_path: Path) -> None:
    config = AppConfig(
        youtube_download_dir=tmp_path / "youtube",
        audio_output_dir=tmp_path / "audio",
        transcript_output_dir=tmp_path / "transcripts",
        chunk_output_dir=tmp_path / "chunks",
        embedding_cache_dir=tmp_path / "embeddings",
        model_cache_dir=tmp_path / "models",
        local_transcription_backend="whisper_cpp_metal",
    )
    whisper_payload = {
        "transcription": [
            {
                "offsets": {"from": 1000, "to": 2500},
                "text": "cached whisper cpp text",
            }
        ]
    }
    whisper_path = whisper_cpp_json_path_for("vid", config)
    whisper_path.parent.mkdir(parents=True, exist_ok=True)
    whisper_path.write_text(json.dumps(whisper_payload))

    transcript = transcribe_audio(tmp_path / "audio.wav", "vid", config)

    assert transcript.source == "local_whisper"
    assert transcript.segments[0].text == "cached whisper cpp text"
    assert transcript.segments[0].start_time == 1.0
    assert local_transcript_path_for("vid", config).exists()
