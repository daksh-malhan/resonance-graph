from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from app.channel_pipeline import ChannelPipelineEvent, run_channel_ingest_pipeline
from app.config import AppConfig, load_config
from app.errors import AppError
from app.models import ChannelVideo, DownloadResult, TranscriptChunk
from app.pipeline import (
    download_video_stage,
    extract_audio_stage,
    fetch_caption_stage,
    ingest_caption_stage,
)
from app.youtube import discover_channel_videos


@dataclass
class VideoRun:
    video_id: str
    title: str
    url: str
    ok: bool
    elapsed_seconds: float
    chunks: int = 0
    transcript_status: str | None = None
    error: str | None = None
    stages: list[str] = field(default_factory=list)


@dataclass
class MethodRun:
    method: str
    elapsed_seconds: float
    succeeded: int
    failed: int
    skipped: int
    chunks: int
    videos: list[VideoRun]


def _stage_collector(stages: list[str]) -> Callable[[str, str], None]:
    def collect(stage: str, message: str) -> None:
        if not stages or stages[-1] != stage:
            stages.append(stage)
        print(f"    {stage}: {message}", flush=True)

    return collect


def _run_sequential_caption_fast_path(
    videos: list[ChannelVideo],
    config: AppConfig,
    force: bool,
) -> MethodRun:
    start = time.perf_counter()
    results: list[VideoRun] = []

    for index, video in enumerate(videos, start=1):
        print(f"[old {index}/{len(videos)}] {video.title}", flush=True)
        stages: list[str] = []
        video_start = time.perf_counter()
        try:
            download = download_video_stage(
                video.url,
                config,
                force=force,
                stage_callback=_stage_collector(stages),
            )
            extract_audio_stage(
                download,
                config,
                force=force,
                stage_callback=_stage_collector(stages),
            )
            caption = fetch_caption_stage(
                download,
                config,
                force=force,
                stage_callback=_stage_collector(stages),
            )
            if not caption or not caption.segments:
                results.append(
                    VideoRun(
                        video_id=video.video_id,
                        title=video.title,
                        url=video.url,
                        ok=False,
                        elapsed_seconds=time.perf_counter() - video_start,
                        error="No YouTube captions found; local Whisper skipped for timing benchmark.",
                        stages=stages,
                    )
                )
                continue

            final_download, chunks = ingest_caption_stage(
                download,
                caption,
                config,
                force=force,
                stage_callback=_stage_collector(stages),
            )
            results.append(
                _successful_video_run(
                    video,
                    final_download,
                    chunks,
                    time.perf_counter() - video_start,
                    stages,
                )
            )
        except Exception as exc:
            results.append(
                VideoRun(
                    video_id=video.video_id,
                    title=video.title,
                    url=video.url,
                    ok=False,
                    elapsed_seconds=time.perf_counter() - video_start,
                    error=str(exc),
                    stages=stages,
                )
            )

    elapsed = time.perf_counter() - start
    return _method_run("old_sequential_caption_fast_path", elapsed, results)


def _run_pipelined_caption_fast_path(
    videos: list[ChannelVideo],
    config: AppConfig,
    force: bool,
) -> MethodRun:
    start = time.perf_counter()
    stage_map: dict[str, list[str]] = {video.video_id: [] for video in videos}

    def on_event(event: ChannelPipelineEvent) -> None:
        stages = stage_map.setdefault(event.video.video_id, [])
        if not stages or stages[-1] != event.stage:
            stages.append(event.stage)
        print(
            f"[pipe {event.index}/{event.total}] {event.video.title} | {event.stage}: {event.message}",
            flush=True,
        )

    result = run_channel_ingest_pipeline(
        videos,
        config,
        force=force,
        stop_on_error=False,
        background_local=False,
        item_callback=on_event,
    )
    elapsed = time.perf_counter() - start

    runs: list[VideoRun] = []
    by_id = {item.get("video_id"): item for item in result.items}
    for video in videos:
        item = by_id.get(video.video_id) or {}
        runs.append(
            VideoRun(
                video_id=video.video_id,
                title=video.title,
                url=video.url,
                ok=bool(item.get("ok")),
                elapsed_seconds=0.0,
                chunks=int(item.get("chunks") or 0),
                transcript_status=item.get("transcript_status"),
                error=item.get("error"),
                stages=stage_map.get(video.video_id, []),
            )
        )

    return MethodRun(
        method="pipelined_channel_caption_fast_path",
        elapsed_seconds=elapsed,
        succeeded=result.succeeded,
        failed=result.failed,
        skipped=result.skipped,
        chunks=sum(run.chunks for run in runs),
        videos=runs,
    )


def _successful_video_run(
    video: ChannelVideo,
    download: DownloadResult,
    chunks: list[TranscriptChunk],
    elapsed_seconds: float,
    stages: list[str],
) -> VideoRun:
    return VideoRun(
        video_id=download.episode.video_id,
        title=download.episode.title or video.title,
        url=video.url,
        ok=True,
        elapsed_seconds=elapsed_seconds,
        chunks=len(chunks),
        transcript_status=download.episode.transcript_status,
        stages=stages,
    )


def _method_run(method: str, elapsed: float, results: list[VideoRun]) -> MethodRun:
    return MethodRun(
        method=method,
        elapsed_seconds=elapsed,
        succeeded=sum(1 for result in results if result.ok),
        failed=sum(1 for result in results if not result.ok),
        skipped=0,
        chunks=sum(result.chunks for result in results),
        videos=results,
    )


def _write_report(
    output_dir: Path,
    channel_url: str,
    limit: int,
    discovery_seconds: float,
    videos: list[ChannelVideo],
    old: MethodRun,
    pipelined: MethodRun,
    force: bool,
    max_resolution: int | None,
    isolated_method_dirs: bool,
    run_order: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    speedup = old.elapsed_seconds / pipelined.elapsed_seconds if pipelined.elapsed_seconds else None
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "channel_url": channel_url,
        "limit": limit,
        "force": force,
        "max_youtube_resolution": max_resolution,
        "isolated_method_dirs": isolated_method_dirs,
        "run_order": run_order,
        "background_local_transcription": False,
        "local_whisper": "skipped for timing benchmark",
        "discovery_seconds": discovery_seconds,
        "selected_videos": [video.model_dump(mode="json") for video in videos],
        "old": asdict(old),
        "pipelined": asdict(pipelined),
        "speedup": speedup,
    }
    (output_dir / "ingestion-methods.json").write_text(json.dumps(payload, indent=2))

    lines = [
        "# Ingestion Method Benchmark",
        "",
        f"- Channel: `{channel_url}`",
        f"- Limit: `{limit}`",
        f"- Force redownload/rebuild: `{force}`",
        f"- Max YouTube resolution: `{max_resolution or 'configured default'}`",
        f"- Isolated method file/cache dirs: `{isolated_method_dirs}`",
        f"- Run order: `{run_order}`",
        "- Local Whisper: skipped for timing so the test compares searchable caption ingestion.",
        f"- Discovery: `{discovery_seconds:.2f}s`",
        "",
        "| Method | Time | Succeeded | Failed | Chunks |",
        "| --- | ---: | ---: | ---: | ---: |",
        (
            f"| Old sequential caption fast path | {old.elapsed_seconds:.2f}s | "
            f"{old.succeeded} | {old.failed} | {old.chunks} |"
        ),
        (
            f"| Pipelined channel caption fast path | {pipelined.elapsed_seconds:.2f}s | "
            f"{pipelined.succeeded} | {pipelined.failed} | {pipelined.chunks} |"
        ),
    ]
    if speedup is not None:
        lines.append(f"| Speedup | {speedup:.2f}x |  |  |  |")

    lines.extend(["", "## Selected Videos", ""])
    for index, video in enumerate(videos, start=1):
        lines.append(f"{index}. `{video.video_id}` - {video.title}")

    lines.extend(["", "## Notes", ""])
    lines.append(
        "This benchmark writes/upserts the selected episodes into the configured Neo4j database. "
        "It does not reset unrelated data."
    )
    lines.append(
        "If `force` is false, cached downloads, audio, captions, chunks, and embeddings can make later runs faster."
    )
    (output_dir / "ingestion-methods.md").write_text("\n".join(lines) + "\n")


def _method_config(config: AppConfig, root: Path) -> AppConfig:
    return config.model_copy(
        update={
            "youtube_download_dir": root / "youtube",
            "audio_output_dir": root / "audio",
            "transcript_output_dir": root / "transcripts",
            "chunk_output_dir": root / "chunks",
            "embedding_cache_dir": root / "embeddings",
            "job_output_dir": root / "jobs",
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare pipelined channel ingest with the old sequential baseline.")
    parser.add_argument("channel_url")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--force", action="store_true", help="Force redownload/rebuild for both method runs.")
    parser.add_argument(
        "--max-resolution",
        type=int,
        default=None,
        help="Override MAX_YOUTUBE_RESOLUTION for both methods during the benchmark.",
    )
    parser.add_argument(
        "--isolate-method-dirs",
        action="store_true",
        help="Use separate download/audio/transcript/chunk/embedding dirs for old and pipelined runs.",
    )
    parser.add_argument(
        "--order",
        choices=["old-first", "pipeline-first"],
        default="pipeline-first",
        help="Run order for the two methods.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark-results/ingestion-methods-latest"),
    )
    args = parser.parse_args()

    if args.limit <= 0:
        raise AppError("--limit must be greater than zero")

    updates = {"background_local_transcription": False}
    if args.max_resolution is not None:
        if args.max_resolution <= 0:
            raise AppError("--max-resolution must be greater than zero")
        updates["max_youtube_resolution"] = args.max_resolution
    config = load_config().model_copy(update=updates)
    config.ensure_directories()

    discovery_start = time.perf_counter()
    videos = discover_channel_videos(args.channel_url, config, limit=args.limit)
    discovery_seconds = time.perf_counter() - discovery_start
    if not videos:
        raise AppError("No long-form videos discovered for benchmark.")

    print(f"Discovered {len(videos)} long-form video(s) in {discovery_seconds:.2f}s", flush=True)
    old_config = (
        _method_config(config, args.output_dir / "old-data")
        if args.isolate_method_dirs
        else config
    )
    pipeline_config = (
        _method_config(config, args.output_dir / "pipeline-data")
        if args.isolate_method_dirs
        else config
    )

    if args.order == "pipeline-first":
        pipelined = _run_pipelined_caption_fast_path(videos, pipeline_config, force=args.force)
        old = _run_sequential_caption_fast_path(videos, old_config, force=args.force)
    else:
        old = _run_sequential_caption_fast_path(videos, old_config, force=args.force)
        pipelined = _run_pipelined_caption_fast_path(videos, pipeline_config, force=args.force)
    _write_report(
        args.output_dir,
        args.channel_url,
        args.limit,
        discovery_seconds,
        videos,
        old,
        pipelined,
        args.force,
        args.max_resolution,
        args.isolate_method_dirs,
        args.order,
    )

    speedup = old.elapsed_seconds / pipelined.elapsed_seconds if pipelined.elapsed_seconds else 0
    print("", flush=True)
    print(f"Old sequential: {old.elapsed_seconds:.2f}s, ok={old.succeeded}, failed={old.failed}", flush=True)
    print(f"Pipelined:      {pipelined.elapsed_seconds:.2f}s, ok={pipelined.succeeded}, failed={pipelined.failed}", flush=True)
    print(f"Speedup:        {speedup:.2f}x", flush=True)
    print(f"Report:         {args.output_dir / 'ingestion-methods.md'}", flush=True)


if __name__ == "__main__":
    main()
