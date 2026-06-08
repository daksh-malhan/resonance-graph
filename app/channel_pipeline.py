from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

from app.config import AppConfig
from app.models import ChannelVideo, DownloadResult, Transcript, TranscriptChunk
from app.pipeline import (
    download_video_stage,
    extract_audio_stage,
    fetch_caption_stage,
    finalize_local_transcript_pipeline,
    ingest_caption_stage,
    queue_local_transcription_stage,
)


@dataclass
class ChannelPipelineEvent:
    index: int
    total: int
    video: ChannelVideo
    stage: str
    message: str
    state: str = "running"


@dataclass
class ChannelPipelineItem:
    index: int
    total: int
    video: ChannelVideo
    download: DownloadResult | None = None
    caption_transcript: Transcript | None = None
    final_download: DownloadResult | None = None
    chunks: list[TranscriptChunk] = field(default_factory=list)


@dataclass
class ChannelPipelineResult:
    succeeded: int = 0
    failed: int = 0
    skipped: int = 0
    total: int = 0
    items: list[dict] = field(default_factory=list)


PipelineCallback = Callable[[ChannelPipelineEvent], None]


def run_channel_ingest_pipeline(
    videos: list[ChannelVideo],
    config: AppConfig,
    force: bool = False,
    stop_on_error: bool = False,
    background_local: bool = True,
    item_callback: PipelineCallback | None = None,
) -> ChannelPipelineResult:
    """Run channel ingestion through bounded stage lanes.

    This is pipeline parallelism, not unbounded whole-video parallelism. Each stage has
    its own worker count so I/O-heavy stages can overlap while embedding, graph writes,
    and local transcription remain constrained.
    """
    config.ensure_directories()
    result = ChannelPipelineResult(total=len(videos))
    if not videos:
        return result

    lock = threading.Lock()
    done = threading.Condition(lock)
    remaining = len(videos)
    stop_event = threading.Event()
    items = [
        ChannelPipelineItem(index=index, total=len(videos), video=video)
        for index, video in enumerate(videos, start=1)
    ]

    def emit(item: ChannelPipelineItem, stage: str, message: str, state: str = "running") -> None:
        if item_callback:
            item_callback(
                ChannelPipelineEvent(
                    index=item.index,
                    total=item.total,
                    video=item.video,
                    stage=stage,
                    message=message,
                    state=state,
                )
            )

    def finish_success(item: ChannelPipelineItem) -> None:
        nonlocal remaining
        download = item.final_download or item.download
        title = download.episode.title if download else item.video.title
        video_id = download.episode.video_id if download else item.video.video_id
        status = download.episode.transcript_status if download else None
        job_id = download.episode.local_transcription_job_id if download else None
        emit(item, "complete", status or "Video ingested", state="succeeded")
        with done:
            result.succeeded += 1
            result.items.append(
                {
                    "video_id": video_id,
                    "title": title,
                    "chunks": len(item.chunks),
                    "transcript_status": status,
                    "local_transcription_job_id": job_id,
                    "ok": True,
                }
            )
            remaining -= 1
            done.notify_all()

    def finish_failure(item: ChannelPipelineItem, error: BaseException) -> None:
        nonlocal remaining
        emit(item, "failed", str(error), state="failed")
        with done:
            result.failed += 1
            result.items.append(
                {
                    "video_id": item.video.video_id,
                    "title": item.video.title,
                    "ok": False,
                    "error": str(error),
                }
            )
            if stop_on_error:
                stop_event.set()
            remaining -= 1
            done.notify_all()

    def finish_skipped(item: ChannelPipelineItem) -> None:
        nonlocal remaining
        message = "Skipped because an earlier video failed"
        emit(item, "failed", message, state="failed")
        with done:
            result.skipped += 1
            result.failed += 1
            result.items.append(
                {
                    "video_id": item.video.video_id,
                    "title": item.video.title,
                    "ok": False,
                    "error": message,
                }
            )
            remaining -= 1
            done.notify_all()

    def stage_callback_for(item: ChannelPipelineItem) -> Callable[[str, str], None]:
        return lambda stage, message: emit(item, stage, message, state="running")

    def submit_or_skip(
        executor: ThreadPoolExecutor,
        item: ChannelPipelineItem,
        fn: Callable[[ChannelPipelineItem], object],
        callback: Callable[[ChannelPipelineItem, Future], None],
    ) -> None:
        if stop_event.is_set():
            finish_skipped(item)
            return
        future = executor.submit(fn, item)
        future.add_done_callback(lambda completed: callback(item, completed))

    def run_download(item: ChannelPipelineItem) -> DownloadResult:
        return download_video_stage(
            item.video.url,
            config,
            force=force,
            stage_callback=stage_callback_for(item),
        )

    def run_audio(item: ChannelPipelineItem) -> None:
        assert item.download is not None
        extract_audio_stage(
            item.download,
            config,
            force=force,
            stage_callback=stage_callback_for(item),
        )

    def run_caption(item: ChannelPipelineItem) -> Transcript | None:
        assert item.download is not None
        return fetch_caption_stage(
            item.download,
            config,
            force=force,
            stage_callback=stage_callback_for(item),
        )

    def run_caption_ingest(item: ChannelPipelineItem) -> tuple[DownloadResult, list[TranscriptChunk]]:
        assert item.download is not None
        assert item.caption_transcript is not None
        download, chunks = ingest_caption_stage(
            item.download,
            item.caption_transcript,
            config,
            force=force,
            stage_callback=stage_callback_for(item),
        )
        emit(item, "caption_ready", "Caption transcript is searchable", state="running")
        if background_local and config.background_local_transcription:
            download = queue_local_transcription_stage(
                download,
                config,
                force=force,
                stage_callback=stage_callback_for(item),
            )
        return download, chunks

    def run_local(item: ChannelPipelineItem) -> tuple[DownloadResult, list[TranscriptChunk]]:
        assert item.download is not None
        return finalize_local_transcript_pipeline(
            item.download.episode.video_id,
            config,
            force=force,
            stage_callback=stage_callback_for(item),
            download=item.download,
            caption_transcript=item.caption_transcript,
        )

    with (
        ThreadPoolExecutor(max_workers=config.pipeline_download_workers, thread_name_prefix="download") as download_pool,
        ThreadPoolExecutor(max_workers=config.pipeline_audio_workers, thread_name_prefix="audio") as audio_pool,
        ThreadPoolExecutor(max_workers=config.pipeline_caption_workers, thread_name_prefix="captions") as caption_pool,
        ThreadPoolExecutor(max_workers=config.pipeline_ingest_workers, thread_name_prefix="ingest") as ingest_pool,
        ThreadPoolExecutor(max_workers=config.pipeline_local_workers, thread_name_prefix="local") as local_pool,
    ):

        def after_download(item: ChannelPipelineItem, future: Future) -> None:
            try:
                item.download = future.result()
            except Exception as exc:
                finish_failure(item, exc)
                return
            submit_or_skip(audio_pool, item, run_audio, after_audio)

        def after_audio(item: ChannelPipelineItem, future: Future) -> None:
            try:
                future.result()
            except Exception as exc:
                finish_failure(item, exc)
                return
            submit_or_skip(caption_pool, item, run_caption, after_caption)

        def after_caption(item: ChannelPipelineItem, future: Future) -> None:
            try:
                item.caption_transcript = future.result()
            except Exception as exc:
                finish_failure(item, exc)
                return
            if item.caption_transcript and item.caption_transcript.segments:
                submit_or_skip(ingest_pool, item, run_caption_ingest, after_ingest)
            else:
                submit_or_skip(local_pool, item, run_local, after_local)

        def after_ingest(item: ChannelPipelineItem, future: Future) -> None:
            try:
                item.final_download, item.chunks = future.result()
            except Exception as exc:
                finish_failure(item, exc)
                return
            finish_success(item)

        def after_local(item: ChannelPipelineItem, future: Future) -> None:
            try:
                item.final_download, item.chunks = future.result()
            except Exception as exc:
                finish_failure(item, exc)
                return
            finish_success(item)

        for item in items:
            emit(item, "queued", "Waiting", state="queued")
            submit_or_skip(download_pool, item, run_download, after_download)

        with done:
            while remaining > 0:
                done.wait()

    return result
