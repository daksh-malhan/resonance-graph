from __future__ import annotations

from app.models import RetrievedChunk
from app.utils import format_timestamp

SYSTEM_PROMPT = (
    "You answer questions about a video or podcast transcript. Use only the provided "
    "context. If the context does not contain the answer, say that the video transcript "
    "does not provide enough evidence. Include timestamp citations and the episode title "
    "for supported claims. Prefer merged transcript context when present. Do not invent "
    "facts outside the transcript context."
)


def format_retrieval_context(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "No transcript context was retrieved."

    blocks = []
    for index, chunk in enumerate(chunks, start=1):
        timestamp = f"{format_timestamp(chunk.start_time)}-{format_timestamp(chunk.end_time)}"
        blocks.append(
            "\n".join(
                [
                    f"[{index}] Episode: {chunk.episode_title}",
                    f"Time: {timestamp}",
                    f"Transcript source: {chunk.transcript_source}",
                    f"Citation: {chunk.episode_title} at {timestamp}",
                    f"URL: {chunk.source_url}",
                    f"Score: {chunk.score:.4f}",
                    f"Text: {chunk.text}",
                ]
            )
        )
    return "\n\n".join(blocks)


def build_answer_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    context = format_retrieval_context(chunks)
    return (
        "Answer the question using only this transcript context.\n\n"
        f"Question:\n{question}\n\n"
        f"Transcript context:\n{context}\n\n"
        "Answer with concise reasoning. For every citation, copy the exact Citation field "
        "from the relevant context block. Tie together adjacent context blocks when they "
        "describe the same point. Do not write placeholder citation text."
    )
