from pathlib import Path

from app.config import AppConfig
from app.models import DownloadResult, EpisodeMetadata, SourceMetadata, Transcript, TranscriptChunk
from app.models import TranscriptSegment
from app.pipeline import ingest_url_pipeline


def _transcript(source: str) -> Transcript:
    return Transcript(
        video_id="vid",
        source=source,
        segments=[
            TranscriptSegment(
                segment_id=f"vid:{source}:000000",
                video_id="vid",
                start_time=0,
                end_time=5,
                text=f"{source} text",
                source=source,
            )
        ],
    )


def test_caption_fast_path_ingests_caption_then_merged(monkeypatch, tmp_path: Path) -> None:
    config = AppConfig(
        youtube_download_dir=tmp_path / "youtube",
        audio_output_dir=tmp_path / "audio",
        transcript_output_dir=tmp_path / "transcripts",
        chunk_output_dir=tmp_path / "chunks",
        embedding_cache_dir=tmp_path / "embeddings",
        model_cache_dir=tmp_path / "models",
    )
    download = DownloadResult(
        source=SourceMetadata(id="https://youtu.be/vid", url="https://youtu.be/vid"),
        episode=EpisodeMetadata(video_id="vid", title="Video", source_url="https://youtu.be/vid"),
        episode_dir=tmp_path / "youtube" / "vid",
    )
    calls: list[str] = []

    monkeypatch.setattr("app.pipeline.download_youtube_video", lambda *args, **kwargs: download)
    monkeypatch.setattr("app.pipeline.extract_audio", lambda *args, **kwargs: tmp_path / "audio.wav")
    monkeypatch.setattr(
        "app.pipeline.extract_youtube_caption_transcript",
        lambda *args, **kwargs: _transcript("youtube_caption"),
    )
    monkeypatch.setattr("app.pipeline.transcribe_audio", lambda *args, **kwargs: _transcript("local_whisper"))
    monkeypatch.setattr(
        "app.pipeline.merge_transcripts",
        lambda *args, **kwargs: _transcript("merged"),
    )

    def fake_ingest(*args, **kwargs):
        calls.append(kwargs["transcript_status"])
        return (
            download.model_copy(
                update={
                    "episode": download.episode.model_copy(
                        update={
                            "transcript_source": kwargs["transcript_source"],
                            "transcript_status": kwargs["transcript_status"],
                        }
                    )
                }
            ),
            [
                TranscriptChunk(
                    chunk_id="vid:chunk:000000",
                    video_id="vid",
                    ordinal=0,
                    text="text",
                    start_time=0,
                    end_time=5,
                    transcript_source=kwargs["transcript_source"],
                )
            ],
        )

    monkeypatch.setattr("app.pipeline._ingest_transcript", fake_ingest)

    final_download, chunks = ingest_url_pipeline("https://youtu.be/vid", config)

    assert calls == ["caption_ready", "merged_ready"]
    assert final_download.episode.transcript_status == "merged_ready"
    assert chunks[0].transcript_source == "merged"


def test_caption_fast_path_can_queue_background_local_merge(monkeypatch, tmp_path: Path) -> None:
    config = AppConfig(
        youtube_download_dir=tmp_path / "youtube",
        audio_output_dir=tmp_path / "audio",
        transcript_output_dir=tmp_path / "transcripts",
        chunk_output_dir=tmp_path / "chunks",
        embedding_cache_dir=tmp_path / "embeddings",
        job_output_dir=tmp_path / "jobs",
        model_cache_dir=tmp_path / "models",
    )
    download = DownloadResult(
        source=SourceMetadata(id="https://youtu.be/vid", url="https://youtu.be/vid"),
        episode=EpisodeMetadata(video_id="vid", title="Video", source_url="https://youtu.be/vid"),
        episode_dir=tmp_path / "youtube" / "vid",
    )

    monkeypatch.setattr("app.pipeline.download_youtube_video", lambda *args, **kwargs: download)
    monkeypatch.setattr("app.pipeline.extract_audio", lambda *args, **kwargs: tmp_path / "audio.wav")
    monkeypatch.setattr(
        "app.pipeline.extract_youtube_caption_transcript",
        lambda *args, **kwargs: _transcript("youtube_caption"),
    )
    monkeypatch.setattr("app.pipeline.transcribe_audio", lambda *args, **kwargs: _transcript("local_whisper"))
    monkeypatch.setattr("app.pipeline.enqueue_local_finalization", lambda *args, **kwargs: "job-1")

    def fake_ingest(*args, **kwargs):
        return (
            download.model_copy(
                update={
                    "episode": download.episode.model_copy(
                        update={
                            "transcript_source": kwargs["transcript_source"],
                            "transcript_status": kwargs["transcript_status"],
                        }
                    )
                }
            ),
            [
                TranscriptChunk(
                    chunk_id="vid:chunk:000000",
                    video_id="vid",
                    ordinal=0,
                    text="text",
                    start_time=0,
                    end_time=5,
                    transcript_source=kwargs["transcript_source"],
                )
            ],
        )

    monkeypatch.setattr("app.pipeline._ingest_transcript", fake_ingest)

    final_download, chunks = ingest_url_pipeline(
        "https://youtu.be/vid",
        config,
        background_local=True,
    )

    assert final_download.episode.transcript_status == "caption_ready"
    assert final_download.episode.local_transcription_job_id == "job-1"
    assert chunks[0].transcript_source == "youtube_caption"
