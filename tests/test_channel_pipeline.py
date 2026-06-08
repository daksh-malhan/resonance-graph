from pathlib import Path

from app.channel_pipeline import run_channel_ingest_pipeline
from app.config import AppConfig
from app.models import (
    ChannelVideo,
    DownloadResult,
    EpisodeMetadata,
    SourceMetadata,
    Transcript,
    TranscriptChunk,
    TranscriptSegment,
)


def _config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        youtube_download_dir=tmp_path / "youtube",
        audio_output_dir=tmp_path / "audio",
        transcript_output_dir=tmp_path / "transcripts",
        chunk_output_dir=tmp_path / "chunks",
        embedding_cache_dir=tmp_path / "embeddings",
        job_output_dir=tmp_path / "jobs",
        model_cache_dir=tmp_path / "models",
    )


def _video(video_id: str) -> ChannelVideo:
    return ChannelVideo(
        video_id=video_id,
        title=f"Video {video_id}",
        url=f"https://www.youtube.com/watch?v={video_id}",
        duration=600,
    )


def _download(video_id: str) -> DownloadResult:
    return DownloadResult(
        source=SourceMetadata(
            id=f"https://www.youtube.com/watch?v={video_id}",
            url=f"https://www.youtube.com/watch?v={video_id}",
        ),
        episode=EpisodeMetadata(
            video_id=video_id,
            title=f"Video {video_id}",
            source_url=f"https://www.youtube.com/watch?v={video_id}",
        ),
        episode_dir=Path("data/youtube") / video_id,
    )


def _transcript(video_id: str, source: str) -> Transcript:
    return Transcript(
        video_id=video_id,
        source=source,
        segments=[
            TranscriptSegment(
                segment_id=f"{video_id}:{source}:000000",
                video_id=video_id,
                start_time=0,
                end_time=5,
                text=f"{source} text",
                source=source,
            )
        ],
    )


def _chunks(video_id: str, source: str) -> list[TranscriptChunk]:
    return [
        TranscriptChunk(
            chunk_id=f"{video_id}:chunk:000000",
            video_id=video_id,
            ordinal=0,
            text=f"{source} chunk",
            start_time=0,
            end_time=5,
            transcript_source=source,
        )
    ]


def test_channel_pipeline_caption_fast_path_queues_local(monkeypatch, tmp_path: Path) -> None:
    config = _config(tmp_path)
    events: list[tuple[str, str]] = []

    def fake_metadata(url, config, force=False, stage_callback=None):
        video_id = url.rsplit("=", 1)[1]
        stage_callback("fetching_metadata", "fake metadata")
        return _download(video_id)

    def fake_audio(download, config, force=False, stage_callback=None):
        stage_callback("extracting_audio", "fake audio")

    def fake_caption(download, config, force=False, stage_callback=None):
        stage_callback("fetching_captions", "fake captions")
        return _transcript(download.episode.video_id, "youtube_caption")

    def fake_ingest(download, caption_transcript, config, force=False, stage_callback=None):
        stage_callback("caption_ingesting", "fake ingest")
        updated = download.model_copy(
            update={
                "episode": download.episode.model_copy(
                    update={
                        "transcript_source": "youtube_caption",
                        "transcript_status": "caption_ready",
                    }
                )
            }
        )
        return updated, _chunks(download.episode.video_id, "youtube_caption")

    def fake_queue(download, config, force=False, stage_callback=None):
        stage_callback("local_transcription_queued", "fake local queue")
        return download.model_copy(
            update={
                "episode": download.episode.model_copy(
                    update={"local_transcription_job_id": f"job-{download.episode.video_id}"}
                )
            }
        )

    monkeypatch.setattr("app.channel_pipeline.fetch_metadata_stage", fake_metadata)
    monkeypatch.setattr(
        "app.channel_pipeline.download_video_stage",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("caption path should not download media")),
    )
    monkeypatch.setattr("app.channel_pipeline.extract_audio_stage", fake_audio)
    monkeypatch.setattr("app.channel_pipeline.fetch_caption_stage", fake_caption)
    monkeypatch.setattr("app.channel_pipeline.ingest_caption_stage", fake_ingest)
    monkeypatch.setattr("app.channel_pipeline.queue_local_transcription_stage", fake_queue)

    result = run_channel_ingest_pipeline(
        [_video("one"), _video("two")],
        config,
        item_callback=lambda event: events.append((event.video.video_id, event.stage)),
    )

    assert result.succeeded == 2
    assert result.failed == 0
    assert {item["local_transcription_job_id"] for item in result.items} == {"job-one", "job-two"}
    assert ("one", "caption_ready") in events
    assert ("two", "local_transcription_queued") in events


def test_channel_pipeline_no_caption_uses_local_lane(monkeypatch, tmp_path: Path) -> None:
    config = _config(tmp_path)
    local_calls: list[str] = []
    download_calls: list[str] = []

    monkeypatch.setattr("app.channel_pipeline.fetch_metadata_stage", lambda url, *args, **kwargs: _download(url.rsplit("=", 1)[1]))

    def fake_download(url, *args, **kwargs):
        video_id = url.rsplit("=", 1)[1]
        download_calls.append(video_id)
        return _download(video_id)

    monkeypatch.setattr("app.channel_pipeline.download_video_stage", fake_download)
    monkeypatch.setattr("app.channel_pipeline.extract_audio_stage", lambda *args, **kwargs: None)
    monkeypatch.setattr("app.channel_pipeline.fetch_caption_stage", lambda *args, **kwargs: None)

    def fake_local(video_id, config, force=False, stage_callback=None, download=None, caption_transcript=None):
        local_calls.append(video_id)
        updated = download.model_copy(
            update={
                "episode": download.episode.model_copy(
                    update={
                        "transcript_source": "local_whisper",
                        "transcript_status": "local_ready",
                    }
                )
            }
        )
        return updated, _chunks(video_id, "local_whisper")

    monkeypatch.setattr("app.channel_pipeline.finalize_local_transcript_pipeline", fake_local)

    result = run_channel_ingest_pipeline([_video("one")], config)

    assert result.succeeded == 1
    assert download_calls == ["one"]
    assert local_calls == ["one"]
    assert result.items[0]["transcript_status"] == "local_ready"
