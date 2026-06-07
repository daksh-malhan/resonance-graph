from __future__ import annotations

import argparse
import logging

from app.background_jobs import (
    mark_job_failed,
    mark_job_running,
    mark_job_succeeded,
)
from app.config import AppConfig, load_config
from app.pipeline import finalize_local_transcript_pipeline
from app.utils import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Resonance Graph background worker")
    subparsers = parser.add_subparsers(dest="command", required=True)

    finalize = subparsers.add_parser(
        "finalize-local",
        help="Run local transcription, merge transcripts, and update Neo4j for one video.",
    )
    finalize.add_argument("video_id")
    finalize.add_argument("--job-id", required=True)
    finalize.add_argument("--force", action="store_true")
    finalize.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    configure_logging(args.verbose)
    config = load_config()
    config.ensure_directories()

    if args.command == "finalize-local":
        _run_finalize_local(args.job_id, args.video_id, config, force=args.force)


def _run_finalize_local(job_id: str, video_id: str, config: AppConfig, force: bool) -> None:
    def on_stage(stage: str, message: str) -> None:
        mark_job_running(job_id, config, stage, message)

    try:
        mark_job_running(job_id, config, "starting", "Starting local transcription finalization")
        download, chunks = finalize_local_transcript_pipeline(
            video_id,
            config,
            force=force,
            stage_callback=on_stage,
            strict_local_failure=True,
        )
        mark_job_succeeded(
            job_id,
            config,
            {
                "video_id": download.episode.video_id,
                "title": download.episode.title,
                "chunks": len(chunks),
                "transcript_source": download.episode.transcript_source,
                "transcript_status": download.episode.transcript_status,
            },
        )
    except Exception as exc:
        logging.exception("Background local finalization failed")
        mark_job_failed(job_id, config, str(exc))
        raise


if __name__ == "__main__":
    main()
