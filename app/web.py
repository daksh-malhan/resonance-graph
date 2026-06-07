from __future__ import annotations

import argparse
import json
import mimetypes
import shutil
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from app.background_jobs import list_jobs, read_job
from app.config import AppConfig, load_config
from app.errors import AppError
from app.neo4j_store import Neo4jStore
from app.ollama import OllamaClient
from app.pipeline import ingest_url_pipeline
from app.retrieval import answer_question
from app.transcription import local_transcription_backend_status
from app.utils import configure_logging, require_executable
from app.youtube import discover_channel_videos

STATIC_DIR = Path(__file__).parent / "static"

STAGE_PROGRESS = {
    "queued": 2,
    "downloading": 8,
    "extracting_audio": 18,
    "fetching_captions": 30,
    "caption_ingesting": 44,
    "caption_ready": 58,
    "local_transcription_queued": 64,
    "local_transcribing": 70,
    "merging_transcripts": 82,
    "chunking": 88,
    "embedding_chunks": 93,
    "writing_graph": 97,
    "complete": 100,
    "failed": 100,
}


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def create(self, kind: str) -> str:
        job_id = uuid.uuid4().hex
        with self._lock:
            self._jobs[job_id] = {
                "id": job_id,
                "kind": kind,
                "state": "queued",
                "message": "Queued",
                "started_at": None,
                "finished_at": None,
                "result": None,
                "error": None,
                "progress": {
                    "percent": 0,
                    "stage": "queued",
                    "label": "Queued",
                    "detail": "",
                },
                "items": [],
            }
        return job_id

    def update(self, job_id: str, **values: Any) -> None:
        with self._lock:
            self._jobs[job_id].update(values)

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None


class WebState:
    def __init__(self, config: AppConfig):
        self.config = config
        self.jobs = JobStore()


def json_response(handler: BaseHTTPRequestHandler, status: int, payload: Any) -> None:
    body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def read_json_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length", "0") or "0")
    if length == 0:
        return {}
    raw = handler.rfile.read(length)
    return json.loads(raw.decode("utf-8"))


class RequestHandler(BaseHTTPRequestHandler):
    state: WebState

    def log_message(self, format: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                self._serve_static("index.html")
            elif parsed.path.startswith("/static/"):
                self._serve_static(parsed.path.removeprefix("/static/"))
            elif parsed.path == "/api/status":
                json_response(self, HTTPStatus.OK, status_payload(self.state.config))
            elif parsed.path == "/api/episodes":
                json_response(self, HTTPStatus.OK, {"episodes": list_episodes(self.state.config)})
            elif parsed.path.startswith("/api/episodes/"):
                video_id = parsed.path.removeprefix("/api/episodes/")
                episode = inspect_episode(self.state.config, video_id)
                json_response(self, HTTPStatus.OK, {"episode": episode})
            elif parsed.path == "/api/graph":
                qs = parse_qs(parsed.query)
                limit = int((qs.get("limit") or ["250"])[0])
                video_id = (qs.get("video_id") or [None])[0] or None
                json_response(
                    self,
                    HTTPStatus.OK,
                    graph_overview(self.state.config, limit=limit, video_id=video_id),
                )
            elif parsed.path.startswith("/api/jobs/"):
                job_id = parsed.path.removeprefix("/api/jobs/")
                job = self.state.jobs.get(job_id)
                if not job:
                    json_response(self, HTTPStatus.NOT_FOUND, {"error": "Job not found"})
                else:
                    json_response(self, HTTPStatus.OK, {"job": job})
            elif parsed.path == "/api/background-jobs":
                qs = parse_qs(parsed.query)
                limit = int((qs.get("limit") or ["25"])[0])
                json_response(
                    self,
                    HTTPStatus.OK,
                    {"jobs": list_jobs(self.state.config, limit=limit)},
                )
            elif parsed.path.startswith("/api/background-jobs/"):
                job_id = parsed.path.removeprefix("/api/background-jobs/")
                job = read_job(job_id, self.state.config)
                if not job:
                    json_response(self, HTTPStatus.NOT_FOUND, {"error": "Background job not found"})
                else:
                    json_response(self, HTTPStatus.OK, {"job": job})
            else:
                json_response(self, HTTPStatus.NOT_FOUND, {"error": "Not found"})
        except Exception as exc:
            json_response(self, HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            body = read_json_body(self)
            if parsed.path == "/api/setup-db":
                setup_db(self.state.config)
                json_response(self, HTTPStatus.OK, {"ok": True})
            elif parsed.path == "/api/ingest-url":
                job_id = start_ingest_url_job(self.state, body)
                json_response(self, HTTPStatus.ACCEPTED, {"job_id": job_id})
            elif parsed.path == "/api/ingest-channel":
                job_id = start_ingest_channel_job(self.state, body)
                json_response(self, HTTPStatus.ACCEPTED, {"job_id": job_id})
            elif parsed.path == "/api/channel-preview":
                videos = channel_preview(self.state.config, body)
                json_response(self, HTTPStatus.OK, {"videos": videos})
            elif parsed.path == "/api/ask":
                json_response(self, HTTPStatus.OK, ask_question(self.state.config, body))
            elif parsed.path == "/api/reset-db":
                if not body.get("confirm"):
                    json_response(self, HTTPStatus.BAD_REQUEST, {"error": "Confirmation required"})
                    return
                reset_db(self.state.config)
                json_response(self, HTTPStatus.OK, {"ok": True})
            elif parsed.path == "/api/clear-cache":
                if not body.get("confirm"):
                    json_response(self, HTTPStatus.BAD_REQUEST, {"error": "Confirmation required"})
                    return
                clear_cache(self.state.config, include_models=bool(body.get("include_models")))
                json_response(self, HTTPStatus.OK, {"ok": True})
            else:
                json_response(self, HTTPStatus.NOT_FOUND, {"error": "Not found"})
        except Exception as exc:
            json_response(self, HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

    def _serve_static(self, relative: str) -> None:
        path = (STATIC_DIR / relative).resolve()
        if not path.is_file() or STATIC_DIR.resolve() not in path.parents:
            json_response(self, HTTPStatus.NOT_FOUND, {"error": "Not found"})
            return
        body = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def status_payload(config: AppConfig) -> dict[str, Any]:
    checks = []
    for label, check in [
        ("yt-dlp", lambda: require_executable("yt-dlp", "Install yt-dlp.")),
        ("FFmpeg", lambda: require_executable("ffmpeg", "Install FFmpeg.")),
        ("Transcription", lambda: local_transcription_backend_status(config)),
        ("Ollama", lambda: OllamaClient(config).ensure_models()),
        ("Neo4j", lambda: _neo4j_health(config)),
    ]:
        try:
            message = check()
            checks.append({"label": label, "ok": True, "message": message or "OK"})
        except Exception as exc:
            checks.append({"label": label, "ok": False, "message": str(exc)})
    return {"checks": checks}


def _neo4j_health(config: AppConfig) -> None:
    store = Neo4jStore(config)
    try:
        store.healthcheck()
    finally:
        store.close()


def setup_db(config: AppConfig) -> None:
    ollama = OllamaClient(config)
    ollama.ensure_models()
    dimension = ollama.embedding_dimension()
    store = Neo4jStore(config)
    try:
        store.setup_schema(dimension)
    finally:
        store.close()


def list_episodes(config: AppConfig) -> list[dict]:
    store = Neo4jStore(config)
    try:
        return store.list_episodes()
    finally:
        store.close()


def inspect_episode(config: AppConfig, video_id: str) -> dict | None:
    store = Neo4jStore(config)
    try:
        return store.inspect_episode(video_id)
    finally:
        store.close()


def graph_overview(config: AppConfig, limit: int, video_id: str | None) -> dict:
    store = Neo4jStore(config)
    try:
        return store.graph_overview(limit=limit, video_id=video_id)
    finally:
        store.close()


def ask_question(config: AppConfig, body: dict[str, Any]) -> dict:
    question = str(body.get("question") or "").strip()
    if not question:
        raise AppError("Question is required.")
    top_k = int(body.get("top_k") or config.retrieval_top_k)
    neighbors = int(body.get("neighbors") if body.get("neighbors") is not None else 1)
    video_id = str(body.get("video_id") or "").strip() or None
    ollama = OllamaClient(config)
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
    return result.model_dump(mode="json")


def channel_preview(config: AppConfig, body: dict[str, Any]) -> list[dict]:
    url = str(body.get("url") or "").strip()
    if not url:
        raise AppError("Channel URL is required.")
    videos = discover_channel_videos(
        url,
        config,
        min_duration_seconds=_optional_int(body.get("min_duration")),
        limit=_optional_int(body.get("limit")),
    )
    return [video.model_dump(mode="json") for video in videos]


def start_ingest_url_job(state: WebState, body: dict[str, Any]) -> str:
    url = str(body.get("url") or "").strip()
    if not url:
        raise AppError("Video URL is required.")
    force = bool(body.get("force"))
    job_id = state.jobs.create("ingest-url")

    def run() -> None:
        def on_stage(stage: str, message: str) -> None:
            state.jobs.update(
                job_id,
                state="running",
                stage=stage,
                message=message,
                progress=progress_payload(stage, message),
                items=[
                    progress_item(
                        title="Video ingest",
                        stage=stage,
                        message=message,
                        state="running",
                    )
                ],
            )

        state.jobs.update(
            job_id,
            state="running",
            stage="queued",
            message="Ingesting video",
            started_at=time.time(),
            progress=progress_payload("queued", "Ingesting video"),
            items=[
                progress_item(
                    title="Video ingest",
                    stage="queued",
                    message="Preparing ingest",
                    state="running",
                )
            ],
        )
        try:
            download, chunks = ingest_url_pipeline(
                url,
                state.config,
                force=force,
                stage_callback=on_stage,
                background_local=True,
            )
            state.jobs.update(
                job_id,
                state="succeeded",
                stage="complete",
                message="Video ingested",
                finished_at=time.time(),
                progress=progress_payload("complete", "Video ingested"),
                items=[
                    progress_item(
                        title=download.episode.title,
                        video_id=download.episode.video_id,
                        stage="complete",
                        message=download.episode.transcript_status or "Video ingested",
                        state="succeeded",
                        percent=100,
                    )
                ],
                result={
                    "video_id": download.episode.video_id,
                    "title": download.episode.title,
                    "chunks": len(chunks),
                    "source_url": download.episode.source_url,
                    "transcript_source": download.episode.transcript_source,
                    "transcript_status": download.episode.transcript_status,
                    "local_transcription_job_id": download.episode.local_transcription_job_id,
                },
            )
        except Exception as exc:
            state.jobs.update(
                job_id,
                state="failed",
                stage="failed",
                message="Video ingest failed",
                finished_at=time.time(),
                error=str(exc),
                progress=progress_payload("failed", "Video ingest failed"),
                items=[
                    progress_item(
                        title="Video ingest",
                        stage="failed",
                        message=str(exc),
                        state="failed",
                        percent=100,
                    )
                ],
            )

    threading.Thread(target=run, daemon=True).start()
    return job_id


def start_ingest_channel_job(state: WebState, body: dict[str, Any]) -> str:
    url = str(body.get("url") or "").strip()
    if not url:
        raise AppError("Channel URL is required.")
    force = bool(body.get("force"))
    stop_on_error = bool(body.get("stop_on_error"))
    limit = _optional_int(body.get("limit"))
    min_duration = _optional_int(body.get("min_duration"))
    job_id = state.jobs.create("ingest-channel")

    def run() -> None:
        state.jobs.update(
            job_id,
            state="running",
            message="Discovering channel videos",
            stage="discovering",
            started_at=time.time(),
            progress={
                "percent": 1,
                "stage": "discovering",
                "label": "Discovering channel videos",
                "detail": "",
            },
            result={"succeeded": 0, "failed": 0, "total": 0, "items": []},
        )
        try:
            videos = discover_channel_videos(
                url,
                state.config,
                min_duration_seconds=min_duration,
                limit=limit,
            )
            result = {"succeeded": 0, "failed": 0, "total": len(videos), "items": []}
            progress_items = [
                progress_item(
                    title=video.title,
                    video_id=video.video_id,
                    stage="queued",
                    message="Waiting",
                    state="queued",
                    percent=0,
                )
                for video in videos
            ]
            state.jobs.update(
                job_id,
                stage="queued",
                progress=channel_progress_payload(progress_items),
                items=progress_items,
                result=result,
            )
            for index, video in enumerate(videos, start=1):
                state.jobs.update(
                    job_id,
                    message=f"Ingesting {index}/{len(videos)}: {video.title}",
                    progress=channel_progress_payload(progress_items),
                    items=progress_items,
                    result=result,
                )
                try:
                    def on_stage(stage: str, message: str) -> None:
                        progress_items[index - 1] = progress_item(
                            title=video.title,
                            video_id=video.video_id,
                            stage=stage,
                            message=message,
                            state="running",
                        )
                        state.jobs.update(
                            job_id,
                            state="running",
                            stage=stage,
                            message=f"{index}/{len(videos)}: {message}",
                            progress=channel_progress_payload(progress_items),
                            items=progress_items,
                            result=result,
                        )

                    download, chunks = ingest_url_pipeline(
                        video.url,
                        state.config,
                        force=force,
                        stage_callback=on_stage,
                        background_local=True,
                    )
                    result["succeeded"] += 1
                    progress_items[index - 1] = progress_item(
                        title=download.episode.title,
                        video_id=download.episode.video_id,
                        stage="complete",
                        message=download.episode.transcript_status or "Video ingested",
                        state="succeeded",
                        percent=100,
                    )
                    result["items"].append(
                        {
                            "video_id": download.episode.video_id,
                            "title": download.episode.title,
                            "chunks": len(chunks),
                            "transcript_status": download.episode.transcript_status,
                            "local_transcription_job_id": download.episode.local_transcription_job_id,
                            "ok": True,
                        }
                    )
                except Exception as exc:
                    result["failed"] += 1
                    progress_items[index - 1] = progress_item(
                        title=video.title,
                        video_id=video.video_id,
                        stage="failed",
                        message=str(exc),
                        state="failed",
                        percent=100,
                    )
                    result["items"].append(
                        {
                            "video_id": video.video_id,
                            "title": video.title,
                            "ok": False,
                            "error": str(exc),
                        }
                    )
                    if stop_on_error:
                        raise

            state.jobs.update(
                job_id,
                state="succeeded" if result["failed"] == 0 else "failed",
                stage="complete" if result["failed"] == 0 else "failed",
                message="Channel ingest complete",
                finished_at=time.time(),
                progress=channel_progress_payload(progress_items),
                items=progress_items,
                result=result,
                error=None if result["failed"] == 0 else f"{result['failed']} video(s) failed",
            )
        except Exception as exc:
            state.jobs.update(
                job_id,
                state="failed",
                stage="failed",
                message="Channel ingest failed",
                finished_at=time.time(),
                error=str(exc),
                progress=progress_payload("failed", "Channel ingest failed"),
            )

    threading.Thread(target=run, daemon=True).start()
    return job_id


def reset_db(config: AppConfig) -> None:
    store = Neo4jStore(config)
    try:
        store.reset_database()
    finally:
        store.close()


def clear_cache(config: AppConfig, include_models: bool = False) -> None:
    targets = [
        config.youtube_download_dir,
        config.audio_output_dir,
        config.transcript_output_dir,
        config.chunk_output_dir,
        config.embedding_cache_dir,
        config.job_output_dir,
    ]
    if include_models:
        targets.append(config.model_cache_dir)
    for target in targets:
        shutil.rmtree(target, ignore_errors=True)
        target.mkdir(parents=True, exist_ok=True)


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    parsed = int(value)
    if parsed < 0:
        raise AppError("Numeric values must be zero or greater.")
    return parsed


def progress_payload(stage: str, message: str, percent: int | None = None) -> dict[str, Any]:
    return {
        "percent": percent if percent is not None else STAGE_PROGRESS.get(stage, 5),
        "stage": stage,
        "label": stage.replace("_", " ").title(),
        "detail": message,
    }


def progress_item(
    title: str,
    stage: str,
    message: str,
    state: str,
    video_id: str | None = None,
    percent: int | None = None,
) -> dict[str, Any]:
    return {
        "title": title,
        "video_id": video_id,
        "state": state,
        "stage": stage,
        "message": message,
        "percent": percent if percent is not None else STAGE_PROGRESS.get(stage, 5),
    }


def channel_progress_payload(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        return {
            "percent": 100,
            "stage": "complete",
            "label": "Channel ingest",
            "detail": "No matching videos found",
        }
    percent = round(sum(int(item.get("percent") or 0) for item in items) / len(items))
    running = next((item for item in items if item.get("state") == "running"), None)
    completed = sum(1 for item in items if item.get("state") == "succeeded")
    failed = sum(1 for item in items if item.get("state") == "failed")
    detail = f"{completed} complete"
    if failed:
        detail += f", {failed} failed"
    if running:
        detail += f" · {running.get('title')}"
    return {
        "percent": percent,
        "stage": running.get("stage") if running else "complete" if completed + failed == len(items) else "queued",
        "label": "Channel ingest",
        "detail": detail,
    }


def run_server(host: str, port: int) -> None:
    configure_logging()
    config = load_config()
    config.ensure_directories()
    RequestHandler.state = WebState(config)
    server = ThreadingHTTPServer((host, port), RequestHandler)
    print(f"Local GraphRAG web app running at http://{host}:{port}")
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local GraphRAG web app.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    run_server(args.host, args.port)


if __name__ == "__main__":
    main()
