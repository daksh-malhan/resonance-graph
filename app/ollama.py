from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import requests

from app.config import AppConfig
from app.errors import AppError
from app.models import TranscriptChunk

logger = logging.getLogger(__name__)


class OllamaClient:
    def __init__(self, config: AppConfig):
        self.config = config
        self.base_url = config.ollama_base_url.rstrip("/")

    def healthcheck(self) -> None:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise AppError(
                f"Ollama is not reachable at {self.base_url}. Start Ollama and pull the models."
            ) from exc

    def list_models(self) -> set[str]:
        self.healthcheck()
        response = requests.get(f"{self.base_url}/api/tags", timeout=10)
        response.raise_for_status()
        payload = response.json()
        return {model.get("name", "") for model in payload.get("models", [])}

    def ensure_models(self) -> None:
        models = self.list_models()
        missing = [
            model
            for model in [self.config.ollama_chat_model, self.config.ollama_embedding_model]
            if model not in models and f"{model}:latest" not in models
        ]
        if missing:
            pulls = "\n".join(f"  ollama pull {model}" for model in missing)
            raise AppError(f"Missing Ollama model(s): {', '.join(missing)}\nRun:\n{pulls}")

    def embed_text(self, text: str) -> list[float]:
        payload = {"model": self.config.ollama_embedding_model, "prompt": text}
        try:
            response = requests.post(f"{self.base_url}/api/embeddings", json=payload, timeout=120)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise AppError(f"Ollama embedding request failed: {exc}") from exc

        data = response.json()
        embedding = data.get("embedding")
        if not isinstance(embedding, list) or not embedding:
            raise AppError("Ollama embedding response did not include an embedding vector.")
        return [float(value) for value in embedding]

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        payload: dict[str, Any] = {
            "model": self.config.ollama_chat_model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        try:
            response = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=180)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise AppError(f"Ollama chat request failed: {exc}") from exc

        data = response.json()
        content = (data.get("message") or {}).get("content")
        if not content:
            raise AppError("Ollama chat response did not include answer content.")
        return str(content).strip()

    def embedding_dimension(self) -> int:
        return len(self.embed_text("dimension check"))


def embed_chunks(
    chunks: list[TranscriptChunk],
    client: OllamaClient,
    config: AppConfig,
    force: bool = False,
) -> list[TranscriptChunk]:
    config.embedding_cache_dir.mkdir(parents=True, exist_ok=True)
    embedded: list[TranscriptChunk] = []
    for index, chunk in enumerate(chunks, start=1):
        cache_path = _embedding_cache_path(chunk.text, config)
        if cache_path.exists() and not force:
            embedding = json.loads(cache_path.read_text())["embedding"]
        else:
            logger.info("Embedding chunk %s/%s", index, len(chunks))
            embedding = client.embed_text(chunk.text)
            cache_path.write_text(json.dumps({"embedding": embedding}) + "\n")
        embedded.append(chunk.model_copy(update={"embedding": embedding}))
    return embedded


def _embedding_cache_path(text: str, config: AppConfig) -> Path:
    digest = hashlib.sha256(
        f"{config.ollama_embedding_model}\0{text}".encode("utf-8")
    ).hexdigest()
    return config.embedding_cache_dir / f"{digest}.json"
