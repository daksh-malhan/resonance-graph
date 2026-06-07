from __future__ import annotations

from app.config import AppConfig
from app.models import RagAnswer, RetrievedChunk
from app.neo4j_store import Neo4jStore
from app.ollama import OllamaClient
from app.prompts import SYSTEM_PROMPT, build_answer_prompt
from app.utils import format_timestamp


def retrieve_context(
    question: str,
    store: Neo4jStore,
    ollama: OllamaClient,
    config: AppConfig,
    top_k: int | None = None,
    neighbor_window: int = 1,
    video_id: str | None = None,
) -> list[RetrievedChunk]:
    embedding = ollama.embed_text(question)
    return store.vector_search(
        question_embedding=embedding,
        top_k=top_k or config.retrieval_top_k,
        neighbor_window=neighbor_window,
        video_id=video_id,
    )


def answer_question(
    question: str,
    store: Neo4jStore,
    ollama: OllamaClient,
    config: AppConfig,
    top_k: int | None = None,
    neighbor_window: int = 1,
    video_id: str | None = None,
) -> RagAnswer:
    contexts = retrieve_context(question, store, ollama, config, top_k, neighbor_window, video_id)
    prompt = build_answer_prompt(question, contexts)
    answer = ollama.chat(SYSTEM_PROMPT, prompt)
    if contexts:
        sources = "\n".join(
            f"- {chunk.episode_title} at "
            f"{format_timestamp(chunk.start_time)}-{format_timestamp(chunk.end_time)}"
            for chunk in contexts
        )
        answer = f"{answer}\n\nSources:\n{sources}"
    return RagAnswer(answer=answer, contexts=contexts)
