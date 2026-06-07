from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from app.config import AppConfig
from app.errors import AppError


def enqueue_local_finalization(video_id: str, config: AppConfig, force: bool = False) -> str:
    config.ensure_directories()
    job_id = uuid.uuid4().hex
    job_path = job_file_path(job_id, config)
    log_path = config.job_output_dir / f"{job_id}.log"
    write_job(
        job_id,
        config,
        {
            "id": job_id,
            "kind": "local-finalization",
            "video_id": video_id,
            "state": "queued",
            "stage": "queued",
            "message": "Queued local transcription merge",
            "started_at": None,
            "finished_at": None,
            "result": None,
            "error": None,
            "log_path": str(log_path),
        },
    )

    command = [
        sys.executable,
        "-m",
        "app.worker",
        "finalize-local",
        video_id,
        "--job-id",
        job_id,
    ]
    if force:
        command.append("--force")

    with log_path.open("a") as log_file:
        process = subprocess.Popen(
            command,
            cwd=Path.cwd(),
            env=os.environ.copy(),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    update_job(job_id, config, pid=process.pid, job_path=str(job_path))
    return job_id


def job_file_path(job_id: str, config: AppConfig) -> Path:
    return config.job_output_dir / f"{job_id}.json"


def read_job(job_id: str, config: AppConfig) -> dict[str, Any] | None:
    path = job_file_path(job_id, config)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def list_jobs(config: AppConfig, limit: int = 25) -> list[dict[str, Any]]:
    config.job_output_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        json.loads(path.read_text())
        for path in config.job_output_dir.glob("*.json")
        if path.is_file()
    ]
    jobs.sort(key=lambda job: job.get("started_at") or 0, reverse=True)
    return jobs[: max(1, limit)]


def write_job(job_id: str, config: AppConfig, payload: dict[str, Any]) -> None:
    config.job_output_dir.mkdir(parents=True, exist_ok=True)
    path = job_file_path(job_id, config)
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    tmp_path.replace(path)


def update_job(job_id: str, config: AppConfig, **values: Any) -> None:
    current = read_job(job_id, config)
    if current is None:
        raise AppError(f"Background job not found: {job_id}")
    current.update(values)
    write_job(job_id, config, current)


def mark_job_running(job_id: str, config: AppConfig, stage: str, message: str) -> None:
    current = read_job(job_id, config) or {}
    update_job(
        job_id,
        config,
        state="running",
        stage=stage,
        message=message,
        started_at=current.get("started_at") or time.time(),
    )


def mark_job_succeeded(job_id: str, config: AppConfig, result: dict[str, Any]) -> None:
    update_job(
        job_id,
        config,
        state="succeeded",
        stage="complete",
        message="Local transcript merged and graph updated",
        finished_at=time.time(),
        result=result,
        error=None,
    )


def mark_job_failed(job_id: str, config: AppConfig, error: str) -> None:
    update_job(
        job_id,
        config,
        state="failed",
        stage="failed",
        message="Local transcription merge failed",
        finished_at=time.time(),
        error=error,
    )
