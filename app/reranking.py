from __future__ import annotations

import math
import re
from collections import Counter

from app.models import RetrievedChunk

TOKEN_RE = re.compile(r"[a-z0-9']+")

STOPWORDS = {
    "a",
    "about",
    "after",
    "all",
    "also",
    "am",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "she",
    "so",
    "that",
    "the",
    "their",
    "them",
    "there",
    "they",
    "this",
    "to",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "would",
    "you",
    "your",
}


def rerank_chunks(question: str, chunks: list[RetrievedChunk], limit: int) -> list[RetrievedChunk]:
    """Rerank vector candidates with local lexical and metadata signals."""
    if limit <= 0 or not chunks:
        return []

    final_limit = min(limit, len(chunks))
    query_terms = _meaningful_terms(question)
    if not query_terms:
        return chunks[:final_limit]

    vector_scores = _normalized_vector_scores(chunks)
    scored = [
        (
            _rerank_score(question, query_terms, chunk, vector_scores[index]),
            vector_scores[index],
            -index,
            chunk,
        )
        for index, chunk in enumerate(chunks)
    ]
    scored.sort(reverse=True)
    return [chunk for _, _, _, chunk in scored[:final_limit]]


def _rerank_score(
    question: str,
    query_terms: list[str],
    chunk: RetrievedChunk,
    vector_score: float,
) -> float:
    text = chunk.text.lower()
    metadata = _metadata_text(chunk).lower()
    combined = f"{metadata} {text}"
    term_counts = Counter(_tokens(combined))
    unique_terms = set(term_counts)

    coverage = sum(1 for term in query_terms if term in unique_terms) / len(query_terms)
    density = sum(min(3, term_counts.get(term, 0)) for term in query_terms) / (3 * len(query_terms))
    metadata_coverage = sum(1 for term in query_terms if term in set(_tokens(metadata))) / len(query_terms)
    phrase_score = _phrase_score(question, combined)

    return (
        0.55 * vector_score
        + 0.22 * coverage
        + 0.10 * density
        + 0.08 * metadata_coverage
        + 0.05 * phrase_score
    )


def _meaningful_terms(text: str) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for token in _tokens(text):
        if token in STOPWORDS or len(token) < 2:
            continue
        if token not in seen:
            terms.append(token)
            seen.add(token)
    return terms


def _tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def _metadata_text(chunk: RetrievedChunk) -> str:
    role_text = " ".join(
        f"{candidate.name} {candidate.role} {candidate.evidence_text}"
        for candidate in chunk.episode_role_candidates
    )
    return " ".join(
        value
        for value in [
            chunk.episode_title,
            chunk.episode_channel,
            chunk.episode_uploader,
            chunk.episode_creator,
            role_text,
        ]
        if value
    )


def _phrase_score(question: str, text: str) -> float:
    question_tokens = _tokens(question)
    if len(question_tokens) < 2:
        return 0.0

    phrases = [
        " ".join(question_tokens[index : index + size])
        for size in (2, 3)
        for index in range(0, len(question_tokens) - size + 1)
        if all(token not in STOPWORDS for token in question_tokens[index : index + size])
    ]
    if not phrases:
        return 0.0
    matches = sum(1 for phrase in phrases if phrase in text)
    return matches / len(phrases)


def _normalized_vector_scores(chunks: list[RetrievedChunk]) -> list[float]:
    raw_scores = [chunk.score for chunk in chunks]
    finite_scores = [score for score in raw_scores if math.isfinite(score)]
    if not finite_scores:
        return [0.0 for _ in chunks]

    min_score = min(finite_scores)
    max_score = max(finite_scores)
    if 0.0 <= min_score and max_score <= 1.0:
        return [max(0.0, min(1.0, score)) if math.isfinite(score) else 0.0 for score in raw_scores]
    if max_score == min_score:
        return [1.0 if math.isfinite(score) else 0.0 for score in raw_scores]
    return [
        ((score - min_score) / (max_score - min_score)) if math.isfinite(score) else 0.0
        for score in raw_scores
    ]
