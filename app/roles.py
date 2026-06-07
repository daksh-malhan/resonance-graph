from __future__ import annotations

import re

from app.models import DownloadResult, RoleCandidate, Transcript

MAX_EVIDENCE_LENGTH = 280


def extract_role_candidates(
    download: DownloadResult,
    transcript: Transcript | None = None,
) -> list[RoleCandidate]:
    """Extract generic, evidence-backed role candidates from metadata and transcript.

    This intentionally avoids dataset-specific names. YouTube channel/uploader/creator
    fields are preserved as their actual metadata roles. Host/guest roles are only
    added as cautious candidates when generic title, description, or intro patterns
    provide evidence.
    """
    episode = download.episode
    candidates: list[RoleCandidate] = []

    _add_candidate(
        candidates,
        name=episode.channel,
        role="publishing_channel",
        confidence=1.0,
        evidence_source="youtube_channel",
        evidence_text=episode.channel,
        clean=False,
    )
    _add_candidate(
        candidates,
        name=episode.uploader,
        role="uploader",
        confidence=1.0,
        evidence_source="youtube_uploader",
        evidence_text=episode.uploader,
        clean=False,
    )
    _add_candidate(
        candidates,
        name=episode.creator,
        role="creator",
        confidence=0.95,
        evidence_source="youtube_creator",
        evidence_text=episode.creator,
        clean=False,
    )

    _extract_title_candidates(candidates, episode.title, episode.channel, episode.uploader)
    _extract_description_candidates(candidates, episode.description)
    if transcript:
        _extract_intro_candidates(candidates, transcript)

    return _dedupe_candidates(candidates)


def _extract_title_candidates(
    candidates: list[RoleCandidate],
    title: str,
    channel: str | None,
    uploader: str | None,
) -> None:
    if not title:
        return

    primary_title = title.split("|", 1)[0].strip()
    x_match = re.search(r"(.+?)\s+[xX×]\s+(.+)", primary_title)
    if x_match:
        left = _clean_name_phrase(x_match.group(1))
        right = _clean_name_phrase(x_match.group(2))
        _add_candidate(
            candidates,
            name=left,
            role="possible_guest",
            confidence=0.62,
            evidence_source="title_pattern",
            evidence_text=title,
        )
        confidence = 0.72 if _same_name(right, channel) or _same_name(right, uploader) else 0.62
        _add_candidate(
            candidates,
            name=right,
            role="possible_host",
            confidence=confidence,
            evidence_source="title_pattern",
            evidence_text=title,
        )

    with_match = re.search(r"\bwith\s+([^|:,-]{3,80})", primary_title, flags=re.IGNORECASE)
    if with_match:
        _add_candidate(
            candidates,
            name=_clean_name_phrase(with_match.group(1)),
            role="possible_guest",
            confidence=0.55,
            evidence_source="title_pattern",
            evidence_text=title,
        )


def _extract_description_candidates(
    candidates: list[RoleCandidate],
    description: str | None,
) -> None:
    if not description:
        return

    for pattern in [
        r"\bhosted by\s+([A-Z][A-Za-z .'-]{2,80})",
        r"\bhost[:\s-]+([A-Z][A-Za-z .'-]{2,80})",
        r"\byour host[,:\s-]+([A-Z][A-Za-z .'-]{2,80})",
    ]:
        for match in re.finditer(pattern, description[:2500], flags=re.IGNORECASE):
            _add_candidate(
                candidates,
                name=_clean_name_phrase(match.group(1)),
                role="host",
                confidence=0.9,
                evidence_source="youtube_description",
                evidence_text=_nearby_text(description, match.start(), match.end()),
            )


def _extract_intro_candidates(candidates: list[RoleCandidate], transcript: Transcript) -> None:
    intro_text = " ".join(segment.text for segment in transcript.segments[:20])
    if not intro_text:
        return

    for pattern in [
        r"\bI[' ]?m\s+([A-Z][A-Za-z .'-]{2,60})",
        r"\bI am\s+([A-Z][A-Za-z .'-]{2,60})",
        r"\bmy name is\s+([A-Z][A-Za-z .'-]{2,60})",
    ]:
        for match in re.finditer(pattern, intro_text[:4000]):
            _add_candidate(
                candidates,
                name=_clean_name_phrase(match.group(1)),
                role="possible_host",
                confidence=0.7,
                evidence_source="transcript_intro",
                evidence_text=_nearby_text(intro_text, match.start(), match.end()),
            )


def _add_candidate(
    candidates: list[RoleCandidate],
    name: str | None,
    role: str,
    confidence: float,
    evidence_source: str,
    evidence_text: str | None,
    clean: bool = True,
) -> None:
    cleaned = _clean_name_phrase(name or "") if clean else re.sub(r"\s+", " ", name or "").strip()
    if not cleaned or len(cleaned) < 2:
        return
    candidates.append(
        RoleCandidate(
            name=cleaned,
            role=role,
            confidence=max(0.0, min(confidence, 1.0)),
            evidence_source=evidence_source,
            evidence_text=(evidence_text or cleaned).strip()[:MAX_EVIDENCE_LENGTH],
        )
    )


def _dedupe_candidates(candidates: list[RoleCandidate]) -> list[RoleCandidate]:
    merged: dict[tuple[str, str, str], RoleCandidate] = {}
    for candidate in candidates:
        key = (
            _normalize_name(candidate.name),
            candidate.role,
            candidate.evidence_source,
        )
        existing = merged.get(key)
        if existing is None or candidate.confidence > existing.confidence:
            merged[key] = candidate
    return sorted(
        merged.values(),
        key=lambda item: (-item.confidence, item.role, item.name.lower()),
    )


def _clean_name_phrase(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", value).strip(" -:|,.\n\t")
    cleaned = re.sub(
        r"\b(on|about|discusses|discussing|talks|explains|with|and today|and we|today we)\b.*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip(" -:|,.")
    return cleaned


def _nearby_text(text: str, start: int, end: int) -> str:
    window_start = max(0, start - 90)
    window_end = min(len(text), end + 90)
    return re.sub(r"\s+", " ", text[window_start:window_end]).strip()


def _same_name(left: str | None, right: str | None) -> bool:
    if not left or not right:
        return False
    return _normalize_name(left) == _normalize_name(right)


def _normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())
