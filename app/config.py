from __future__ import annotations

import os
import re
import tomllib
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _load_config_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    if path.suffix.lower() in {".yaml", ".yml"}:
        return yaml.safe_load(path.read_text()) or {}
    if path.suffix.lower() == ".toml":
        return tomllib.loads(path.read_text())
    raise ValueError("APP_CONFIG_FILE must point to a .yaml, .yml, or .toml file")


class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = "local-graphrag-password"

    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = "llama3.1:8b"
    ollama_embedding_model: str = "nomic-embed-text"

    youtube_download_dir: Path = Path("data/youtube")
    audio_output_dir: Path = Path("data/audio")
    transcript_output_dir: Path = Path("data/transcripts")
    chunk_output_dir: Path = Path("data/chunks")
    embedding_cache_dir: Path = Path("data/embeddings")
    job_output_dir: Path = Path("data/jobs")
    model_cache_dir: Path = Path("data/models")

    chunk_size: int = 900
    chunk_overlap: int = 200
    retrieval_top_k: int = 8
    max_youtube_resolution: int = 720
    channel_min_video_duration_seconds: int = 61
    channel_max_videos: int = 0

    transcription_backend: str = "faster-whisper"
    transcript_fast_path: str = "youtube_captions"
    background_local_transcription: bool = True
    local_transcription_backend: str = "whisper_cpp_metal"
    whisper_model_size: str = "base"
    transcript_merge_strategy: str = "time_overlap_best_text"
    faster_whisper_model: str = "base"
    faster_whisper_device: str = "cpu"
    faster_whisper_compute_type: str = "int8"
    whisper_cpp_binary: str = "whisper-cli"
    whisper_cpp_model: Path | None = Path("data/models/whisper.cpp/ggml-base.bin")

    vector_index_name: str = "chunk_embedding_index"
    app_config_file: Path | None = Field(default=None, exclude=True)

    @field_validator(
        "youtube_download_dir",
        "audio_output_dir",
        "transcript_output_dir",
        "chunk_output_dir",
        "embedding_cache_dir",
        "job_output_dir",
        "model_cache_dir",
        "whisper_cpp_model",
        mode="before",
    )
    @classmethod
    def expand_path(cls, value: str | Path | None) -> Path | None:
        if value is None:
            return None
        return Path(value).expanduser()

    @field_validator(
        "chunk_size",
        "retrieval_top_k",
        "max_youtube_resolution",
        "channel_min_video_duration_seconds",
    )
    @classmethod
    def positive_int(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("must be greater than zero")
        return value

    @field_validator("chunk_overlap", "channel_max_videos")
    @classmethod
    def non_negative_int(cls, value: int) -> int:
        if value < 0:
            raise ValueError("must be zero or greater")
        return value

    @field_validator("vector_index_name")
    @classmethod
    def valid_cypher_identifier(cls, value: str) -> str:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
            raise ValueError("must be a simple Neo4j identifier")
        return value

    def ensure_directories(self) -> None:
        for directory in [
            self.youtube_download_dir,
            self.audio_output_dir,
            self.transcript_output_dir,
            self.chunk_output_dir,
            self.embedding_cache_dir,
            self.job_output_dir,
            self.model_cache_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)


def load_config() -> AppConfig:
    load_dotenv()
    config_file = os.getenv("APP_CONFIG_FILE")
    if not config_file:
        return AppConfig()

    file_values = _load_config_file(Path(config_file).expanduser())
    normalized = {str(key).lower(): value for key, value in file_values.items()}
    return AppConfig(**normalized)
