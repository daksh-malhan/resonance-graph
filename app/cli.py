from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer

from app.benchmark import load_benchmark_suite, run_benchmark, write_report
from app.background_jobs import list_jobs, read_job
from app.channel_pipeline import ChannelPipelineEvent, run_channel_ingest_pipeline
from app.config import AppConfig, load_config
from app.errors import AppError
from app.neo4j_store import Neo4jStore
from app.ollama import OllamaClient
from app.pipeline import ingest_url_pipeline
from app.prompts import format_retrieval_context
from app.retrieval import answer_question, retrieve_context
from app.transcription import local_transcription_backend_status
from app.utils import configure_logging, format_timestamp, require_executable
from app.youtube import discover_channel_videos, refresh_cached_youtube_metadata

LEGAL_BOUNDARY = (
    "Only ingest videos that you own, have permission to process, are Creative Commons/"
    "public-domain, or are otherwise legally allowed to download. This tool does not "
    "support bypassing DRM, paywalls, private videos, login-only videos, or platform protections."
)

cli_app = typer.Typer(
    no_args_is_help=True,
    help=f"Resonance Graph: local podcast/video GraphRAG using Neo4j and Ollama.\n\n{LEGAL_BOUNDARY}",
)


def _config(verbose: bool = False) -> AppConfig:
    configure_logging(verbose)
    config = load_config()
    config.ensure_directories()
    return config


def _fail(error: Exception) -> None:
    if isinstance(error, AppError):
        typer.secho(str(error), fg=typer.colors.RED, err=True)
    else:
        logging.exception("Unexpected error")
        typer.secho(f"Unexpected error: {error}", fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)


@cli_app.command("setup-db")
def setup_db(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show detailed logs.")] = False,
) -> None:
    """Create Neo4j constraints, indexes, and the Chunk vector index."""
    try:
        config = _config(verbose)
        ollama = OllamaClient(config)
        ollama.ensure_models()
        dimension = ollama.embedding_dimension()
        store = Neo4jStore(config)
        try:
            store.setup_schema(dimension)
        finally:
            store.close()
        typer.echo(f"Neo4j schema is ready. Chunk embedding dimension: {dimension}")
    except Exception as exc:
        _fail(exc)


@cli_app.command("ingest-url")
def ingest_url(
    url: Annotated[str, typer.Argument(help=f"Approved YouTube video URL. {LEGAL_BOUNDARY}")],
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Redo download, audio, transcript, chunks, and embeddings."),
    ] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show detailed logs.")] = False,
) -> None:
    """Ingest an approved YouTube video with the captions-first pipeline."""
    try:
        config = _config(verbose)
        download, chunks = ingest_url_pipeline(
            url,
            config,
            force=force,
            background_local=True,
        )
        typer.echo(f"Ingested: {download.episode.title}")
        typer.echo(f"Video ID: {download.episode.video_id}")
        typer.echo(f"Chunks: {len(chunks)}")
        typer.echo(f"Transcript status: {download.episode.transcript_status or 'unknown'}")
        if download.episode.local_transcription_job_id:
            typer.echo(
                "Local transcription merge is running in the background. "
                f"Job ID: {download.episode.local_transcription_job_id}"
            )
        typer.echo(f"Source: {download.episode.source_url}")
    except Exception as exc:
        _fail(exc)


@cli_app.command("ingest-channel")
def ingest_channel(
    channel_url: Annotated[
        str,
        typer.Argument(help=f"Approved YouTube channel URL. {LEGAL_BOUNDARY}"),
    ],
    limit: Annotated[
        int | None,
        typer.Option("--limit", help="Maximum number of long-form videos to ingest."),
    ] = None,
    min_duration: Annotated[
        int | None,
        typer.Option(
            "--min-duration",
            help="Minimum duration in seconds. Defaults to CHANNEL_MIN_VIDEO_DURATION_SECONDS.",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="List matching long-form videos without ingesting."),
    ] = False,
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Redo cached download, audio, transcript, chunks, and embeddings."),
    ] = False,
    stop_on_error: Annotated[
        bool,
        typer.Option("--stop-on-error", help="Stop the channel ingest after the first failed video."),
    ] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show detailed logs.")] = False,
) -> None:
    """Discover and ingest long-form channel videos with the pipelined method."""
    try:
        config = _config(verbose)
        if limit is not None and limit < 0:
            raise AppError("--limit must be zero or greater.")
        if min_duration is not None and min_duration <= 0:
            raise AppError("--min-duration must be greater than zero.")

        videos = discover_channel_videos(
            channel_url,
            config,
            min_duration_seconds=min_duration,
            limit=limit,
        )
        if not videos:
            typer.echo("No long-form videos found for this channel.")
            return

        effective_limit = limit if limit is not None else config.channel_max_videos
        if effective_limit and effective_limit > 0:
            typer.echo(
                f"Selected {len(videos)} long-form video(s) due to limit {effective_limit}."
            )
        else:
            typer.echo(f"Found {len(videos)} long-form video(s).")
        if dry_run:
            for index, video in enumerate(videos, start=1):
                duration = (
                    format_timestamp(video.duration)
                    if video.duration is not None
                    else "unknown duration"
                )
                typer.echo(f"{index}. {video.video_id} | {duration} | {video.title} | {video.url}")
            return

        seen_stages: set[tuple[str, str]] = set()

        def on_item(event: ChannelPipelineEvent) -> None:
            key = (event.video.video_id, event.stage)
            if key in seen_stages and event.stage not in {"complete", "failed"}:
                return
            seen_stages.add(key)
            typer.echo(f"[{event.index}/{event.total}] {event.video.title} | {event.stage}: {event.message}")

        result = run_channel_ingest_pipeline(
            videos,
            config,
            force=force,
            stop_on_error=stop_on_error,
            background_local=True,
            item_callback=on_item,
        )

        typer.echo(
            "\nChannel ingest complete. "
            f"Succeeded: {result.succeeded}. Failed: {result.failed}. Skipped: {result.skipped}."
        )
        if result.failed:
            raise typer.Exit(code=1)
    except typer.Exit:
        raise
    except Exception as exc:
        _fail(exc)


@cli_app.command("ask")
def ask(
    question: Annotated[str, typer.Argument(help="Question to answer from ingested transcripts.")],
    top_k: Annotated[int | None, typer.Option("--top-k", help="Number of chunks to retrieve.")] = None,
    neighbors: Annotated[
        int,
        typer.Option("--neighbors", help="Include N adjacent chunks around each retrieved chunk."),
    ] = 0,
    show_context: Annotated[
        bool,
        typer.Option("--show-context", help="Print retrieved context after the answer."),
    ] = False,
    video_id: Annotated[
        str | None,
        typer.Option("--video-id", help="Restrict retrieval to one ingested YouTube video id."),
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show detailed logs.")] = False,
) -> None:
    """Ask a question using Neo4j vector search and a local Ollama chat model."""
    try:
        config = _config(verbose)
        ollama = OllamaClient(config)
        ollama.ensure_models()
        store = Neo4jStore(config)
        try:
            result = answer_question(
                question,
                store,
                ollama,
                config,
                top_k,
                neighbors,
                video_id=video_id,
            )
        finally:
            store.close()
        typer.echo(result.answer)
        if show_context:
            typer.echo("\nRetrieved context:\n")
            typer.echo(format_retrieval_context(result.contexts))
    except Exception as exc:
        _fail(exc)


@cli_app.command("retrieve")
def retrieve(
    question: Annotated[str, typer.Argument(help="Question to retrieve context for.")],
    top_k: Annotated[int | None, typer.Option("--top-k", help="Number of chunks to retrieve.")] = None,
    neighbors: Annotated[
        int,
        typer.Option("--neighbors", help="Include N adjacent chunks around each retrieved chunk."),
    ] = 0,
    video_id: Annotated[
        str | None,
        typer.Option("--video-id", help="Restrict retrieval to one ingested YouTube video id."),
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show detailed logs.")] = False,
) -> None:
    """Print retrieved transcript chunks without generating an answer."""
    try:
        config = _config(verbose)
        ollama = OllamaClient(config)
        store = Neo4jStore(config)
        try:
            chunks = retrieve_context(
                question,
                store,
                ollama,
                config,
                top_k,
                neighbors,
                video_id=video_id,
            )
        finally:
            store.close()
        typer.echo(format_retrieval_context(chunks))
    except Exception as exc:
        _fail(exc)


@cli_app.command("benchmark")
def benchmark(
    suite_path: Annotated[
        Path,
        typer.Argument(help="YAML or JSON benchmark suite with questions and expected evidence."),
    ],
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", help="Directory for benchmark.json and benchmark.md."),
    ] = Path("benchmark-results/latest"),
    top_k: Annotated[int | None, typer.Option("--top-k", help="Number of chunks to retrieve.")] = None,
    neighbors: Annotated[
        int,
        typer.Option("--neighbors", help="Include N adjacent chunks around each retrieved chunk."),
    ] = 0,
    include_answers: Annotated[
        bool,
        typer.Option(
            "--answers/--retrieval-only",
            help="Generate answers and score answer-level signals. Retrieval-only is faster.",
        ),
    ] = True,
    include_answer_text: Annotated[
        bool,
        typer.Option(
            "--include-answer-text",
            help="Write raw generated answers into benchmark.json. Avoid for private data.",
        ),
    ] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show detailed logs.")] = False,
) -> None:
    """Run an accuracy and latency benchmark over ingested transcript data."""
    try:
        config = _config(verbose)
        suite = load_benchmark_suite(suite_path)
        ollama = OllamaClient(config)
        ollama.ensure_models()
        store = Neo4jStore(config)
        try:
            report = run_benchmark(
                suite,
                store,
                ollama,
                config,
                top_k=top_k,
                neighbor_window=neighbors,
                include_answers=include_answers,
                include_answer_text=include_answer_text,
            )
        finally:
            store.close()
        json_path, markdown_path = write_report(report, output_dir)
        typer.echo(f"Benchmark complete: {suite.name}")
        for key, value in report.metrics.items():
            typer.echo(f"{key}: {value}")
        typer.echo(f"JSON: {json_path}")
        typer.echo(f"Markdown: {markdown_path}")
    except Exception as exc:
        _fail(exc)


@cli_app.command("list-episodes")
def list_episodes(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show detailed logs.")] = False,
) -> None:
    """List ingested episodes."""
    try:
        config = _config(verbose)
        store = Neo4jStore(config)
        try:
            episodes = store.list_episodes()
        finally:
            store.close()
        if not episodes:
            typer.echo("No episodes ingested yet.")
            return
        for episode in episodes:
            duration = (
                format_timestamp(float(episode["duration"]))
                if episode.get("duration") is not None
                else "unknown"
            )
            typer.echo(
                f"{episode['video_id']} | {episode['title']} | "
                f"{episode.get('channel') or 'unknown channel'} | "
                f"{duration} | chunks: {episode['chunk_count']} | "
                f"{episode.get('transcript_status') or 'unknown'}"
            )
    except Exception as exc:
        _fail(exc)


@cli_app.command("inspect-episode")
def inspect_episode(
    video_id: Annotated[str, typer.Argument(help="YouTube video id to inspect.")],
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show detailed logs.")] = False,
) -> None:
    """Show stored metadata and counts for one episode."""
    try:
        config = _config(verbose)
        store = Neo4jStore(config)
        try:
            episode = store.inspect_episode(video_id)
        finally:
            store.close()
        if not episode:
            typer.echo(f"No episode found for video id: {video_id}")
            return
        for key, value in episode.items():
            typer.echo(f"{key}: {value}")
    except Exception as exc:
        _fail(exc)


@cli_app.command("refresh-metadata")
def refresh_metadata(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show detailed logs.")] = False,
) -> None:
    """Refresh cached YouTube metadata fields in Neo4j without re-ingesting transcripts."""
    try:
        config = _config(verbose)
        downloads = refresh_cached_youtube_metadata(config)
        if not downloads:
            typer.echo("No cached YouTube info JSON files found.")
            return
        store = Neo4jStore(config)
        try:
            for download in downloads:
                store.upsert_episode_metadata(download)
        finally:
            store.close()
        typer.echo(f"Refreshed metadata for {len(downloads)} cached video(s).")
    except Exception as exc:
        _fail(exc)


@cli_app.command("background-jobs")
def background_jobs(
    job_id: Annotated[
        str | None,
        typer.Option("--job-id", help="Show one background local transcription job."),
    ] = None,
    limit: Annotated[int, typer.Option("--limit", help="Maximum recent jobs to show.")] = 10,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show detailed logs.")] = False,
) -> None:
    """Show detached local transcription and transcript-merge jobs."""
    try:
        config = _config(verbose)
        if job_id:
            job = read_job(job_id, config)
            if not job:
                typer.echo(f"No background job found for id: {job_id}")
                return
            for key, value in job.items():
                typer.echo(f"{key}: {value}")
            return

        jobs = list_jobs(config, limit=limit)
        if not jobs:
            typer.echo("No background jobs yet.")
            return
        for job in jobs:
            typer.echo(
                f"{job['id']} | {job.get('video_id')} | {job.get('state')} | "
                f"{job.get('stage')} | {job.get('message')}"
            )
    except Exception as exc:
        _fail(exc)


@cli_app.command("reset-db")
def reset_db(
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Confirm deletion without prompting.")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show detailed logs.")] = False,
) -> None:
    """Delete all Source, Episode, Chunk, and TranscriptSegment graph data."""
    try:
        config = _config(verbose)
        if not yes:
            if not typer.confirm("Delete all graph data from Neo4j?"):
                typer.echo("Reset aborted.")
                return
        store = Neo4jStore(config)
        try:
            store.reset_database()
        finally:
            store.close()
        typer.echo("Neo4j graph data deleted. Constraints and indexes were kept.")
    except Exception as exc:
        _fail(exc)


@cli_app.command("status")
def status(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show detailed logs.")] = False,
) -> None:
    """Check local dependencies and service availability."""
    config = _config(verbose)
    checks = [
        ("yt-dlp executable", lambda: require_executable("yt-dlp", "Install yt-dlp.")),
        ("FFmpeg executable", lambda: require_executable("ffmpeg", "Install FFmpeg.")),
        ("transcription backend", lambda: local_transcription_backend_status(config)),
        ("Ollama service/models", lambda: OllamaClient(config).ensure_models()),
        ("Neo4j service", lambda: _check_neo4j(config)),
    ]
    failed = False
    for label, check in checks:
        try:
            message = check()
            suffix = f": {message}" if message else ""
            typer.secho(f"OK   {label}{suffix}", fg=typer.colors.GREEN)
        except Exception as exc:
            failed = True
            typer.secho(f"FAIL {label}: {exc}", fg=typer.colors.RED)
    if failed:
        raise typer.Exit(code=1)


def _check_neo4j(config: AppConfig) -> None:
    store = Neo4jStore(config)
    try:
        store.healthcheck()
    finally:
        store.close()


if __name__ == "__main__":
    cli_app()
