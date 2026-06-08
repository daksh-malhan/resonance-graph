from __future__ import annotations

import re

from app.config import AppConfig
from app.models import RagAnswer, RetrievedChunk
from app.neo4j_store import Neo4jStore
from app.ollama import OllamaClient
from app.prompts import SYSTEM_PROMPT, build_answer_prompt
from app.reranking import rerank_chunks
from app.utils import format_timestamp

CORPUS_OVERVIEW_PHRASES = {
    "what is the data about",
    "what is this data about",
    "what is the dataset about",
    "what is this dataset about",
    "what is in the database",
    "what is in this database",
    "what is in the library",
    "what is this library about",
    "summarize the data",
    "summarise the data",
    "summarize this data",
    "summarise this data",
}


def retrieve_context(
    question: str,
    store: Neo4jStore,
    ollama: OllamaClient,
    config: AppConfig,
    top_k: int | None = None,
    neighbor_window: int = 0,
    video_id: str | None = None,
) -> list[RetrievedChunk]:
    final_top_k = top_k or config.retrieval_top_k
    candidate_top_k = max(final_top_k, config.retrieval_candidate_top_k)
    embedding = ollama.embed_text(question)
    candidates = store.vector_search(
        question_embedding=embedding,
        top_k=candidate_top_k,
        neighbor_window=neighbor_window,
        video_id=video_id,
    )
    return rerank_chunks(question, candidates, final_top_k)


def answer_question(
    question: str,
    store: Neo4jStore,
    ollama: OllamaClient,
    config: AppConfig,
    top_k: int | None = None,
    neighbor_window: int = 0,
    video_id: str | None = None,
) -> RagAnswer:
    if video_id is None and is_corpus_overview_question(question):
        return RagAnswer(answer=build_corpus_overview_answer(store.list_episodes()), contexts=[])

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


def is_corpus_overview_question(question: str) -> bool:
    normalized = _normalize_question(question)
    if normalized in CORPUS_OVERVIEW_PHRASES:
        return True
    return (
        ("data" in normalized or "dataset" in normalized or "library" in normalized)
        and any(term in normalized for term in ["about", "summarize", "summarise", "overview"])
        and len(normalized.split()) <= 8
    )


def build_corpus_overview_answer(episodes: list[dict]) -> str:
    if not episodes:
        return "There are no ingested episodes in the local database yet."

    ready = [
        episode
        for episode in episodes
        if int(episode.get("chunk_count") or 0) > 0
    ]
    selected = ready or episodes
    titles = [str(episode.get("title") or episode.get("video_id")) for episode in selected]
    topics = _topic_hints_from_titles(titles)

    lines = [
        f"The local database currently contains {len(selected)} ingested episode(s).",
    ]
    if topics:
        lines.append("From the video titles, the collection appears to cover: " + ", ".join(topics) + ".")
    else:
        lines.append("From the video titles, it appears to be a collection of podcast/video episodes.")

    lines.append("")
    lines.append("Episode title sources:")
    for episode in selected[:10]:
        title = episode.get("title") or episode.get("video_id")
        status = episode.get("transcript_status") or "unknown transcript"
        chunks = int(episode.get("chunk_count") or 0)
        lines.append(f"- {title} ({status}, {chunks} chunks)")

    if len(selected) > 10:
        lines.append(f"- ...and {len(selected) - 10} more episode(s).")

    lines.append("")
    lines.append(
        "For a transcript-grounded answer with timestamp citations, choose one episode in Scope "
        "or ask a more specific question."
    )
    return "\n".join(lines)


def _normalize_question(question: str) -> str:
    return " ".join(
        question.lower()
        .replace("?", " ")
        .replace("!", " ")
        .replace(".", " ")
        .replace(",", " ")
        .split()
    )


def _topic_hints_from_titles(titles: list[str]) -> list[str]:
    hints: list[str] = []
    keyword_groups = [
        ("neuroscience and brain health", ["neuroscience", "brain", "neurostimulation", "learning", "psychedelics"]),
        ("medicine and health", ["science", "safety", "peptides", "speaking languages"]),
        ("investing and markets", ["investing", "equity", "debt", "credit", "artificial intelligence"]),
        ("entrepreneurship and business", ["startup", "founder", "business", "industries"]),
        ("jobs and development", ["jobs", "world bank", "infrastructure"]),
        ("personal growth and psychology", ["trauma", "chaos", "growth"]),
    ]
    title_blob = " ".join(titles).lower()
    for label, keywords in keyword_groups:
        if any(_contains_keyword(title_blob, keyword) for keyword in keywords):
            hints.append(label)
    return hints


def _contains_keyword(text: str, keyword: str) -> bool:
    escaped = re.escape(keyword.lower())
    return re.search(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", text) is not None
