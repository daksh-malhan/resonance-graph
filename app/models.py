from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, HttpUrl, field_validator


class SourceMetadata(BaseModel):
    id: str
    url: str
    kind: str = "youtube"


class EpisodeMetadata(BaseModel):
    video_id: str
    title: str
    source_url: str
    channel: str | None = None
    duration: float | None = None
    upload_date: str | None = None
    local_video_path: Path | None = None
    info_json_path: Path | None = None


class DownloadResult(BaseModel):
    source: SourceMetadata
    episode: EpisodeMetadata
    episode_dir: Path


class ChannelVideo(BaseModel):
    video_id: str
    title: str
    url: str
    duration: float | None = None
    channel: str | None = None


class TranscriptSegment(BaseModel):
    segment_id: str
    video_id: str
    start_time: float
    end_time: float
    text: str


class Transcript(BaseModel):
    video_id: str
    segments: list[TranscriptSegment]


class TranscriptChunk(BaseModel):
    chunk_id: str
    video_id: str
    ordinal: int
    text: str
    start_time: float
    end_time: float
    segment_ids: list[str] = Field(default_factory=list)
    embedding: list[float] | None = None


class RetrievedChunk(BaseModel):
    chunk_id: str
    video_id: str
    episode_title: str
    source_url: str
    text: str
    start_time: float
    end_time: float
    score: float


class RagAnswer(BaseModel):
    answer: str
    contexts: list[RetrievedChunk]


class YouTubeUrl(BaseModel):
    url: HttpUrl

    @field_validator("url")
    @classmethod
    def must_be_youtube(cls, value: HttpUrl) -> HttpUrl:
        host = (value.host or "").lower()
        if host not in {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"}:
            raise ValueError("URL must be a YouTube URL")
        return value
