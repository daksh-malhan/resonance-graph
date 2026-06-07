from __future__ import annotations

from app.config import AppConfig
from app.models import RagAnswer, RetrievedChunk
from app.neo4j_store import Neo4jStore
from app.ollama import OllamaClient
from app.prompts import SYSTEM_PROMPT, build_answer_prompt
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
    if is_metadata_identity_question(question):
        episodes = [store.inspect_episode(video_id)] if video_id else store.list_episodes()
        return RagAnswer(
            answer=build_metadata_identity_answer([episode for episode in episodes if episode]),
            contexts=[],
        )

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


def is_metadata_identity_question(question: str) -> bool:
    normalized = _normalize_question(question)
    words = set(normalized.split())
    identity_terms = {
        "host",
        "hosts",
        "hosted",
        "hosting",
        "owner",
        "owns",
        "owned",
        "channel",
        "uploader",
        "publisher",
        "podcast",
        "show",
    }
    question_terms = {"who", "whose", "which", "what", "is", "are"}
    return bool(words & identity_terms) and bool(words & question_terms) and len(words) <= 14


def build_metadata_identity_answer(episodes: list[dict]) -> str:
    if not episodes:
        return "I do not have stored episode metadata for that selection."

    owners = sorted({_youtube_owner_name(episode) for episode in episodes if _youtube_owner_name(episode)})
    titles = [str(episode.get("title") or episode.get("video_id")) for episode in episodes]

    if not owners:
        return (
            "The stored YouTube metadata does not include channel, uploader, or creator fields "
            "for that selection. I cannot identify the channel owner or publisher from metadata alone."
        )

    lines: list[str] = []
    if len(owners) == 1:
        owner = owners[0]
        citations = _youtube_owner_citations(episodes[0])
        lines.append(f"The stored YouTube metadata identifies `{owner}` as the channel owner or publisher.")
        if citations:
            lines.append("Metadata source: " + "; ".join(citations))
    else:
        lines.append(
            "The selected data has multiple YouTube channel/uploader metadata values: "
            + ", ".join(f"`{owner}`" for owner in owners)
            + ". Treat these as YouTube metadata, not transcript proof."
        )

    title_host_hints = [title for title in titles if any(owner in title for owner in owners)]
    if title_host_hints:
        lines.append("")
        lines.append("Title metadata also mentions the same owner/publisher name:")
        for title in title_host_hints[:5]:
            lines.append(f"- {title} (Episode title: {title})")

    if len(titles) > len(title_host_hints):
        lines.append("")
        lines.append("Episode metadata checked:")
        for title in titles[:5]:
            lines.append(f"- {title}")

    return "\n".join(lines)


def _youtube_owner_name(episode: dict) -> str:
    for key in ["uploader", "channel", "creator"]:
        value = episode.get(key)
        if value and str(value).strip():
            return str(value).strip()
    return ""


def _youtube_owner_citations(episode: dict) -> list[str]:
    citations: list[str] = []
    for label, key in [
        ("YouTube uploader", "uploader"),
        ("YouTube channel", "channel"),
        ("YouTube creator", "creator"),
    ]:
        value = episode.get(key)
        if value and str(value).strip():
            citations.append(f"{label}: {str(value).strip()}")
    return citations


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
        ("investing and markets", ["investing", "equity", "debt", "credit", "ai"]),
        ("entrepreneurship and business", ["startup", "founder", "business", "industries"]),
        ("jobs and development", ["jobs", "world bank", "infrastructure"]),
        ("personal growth and psychology", ["trauma", "chaos", "growth"]),
    ]
    title_blob = " ".join(titles).lower()
    for label, keywords in keyword_groups:
        if any(keyword in title_blob for keyword in keywords):
            hints.append(label)
    return hints
