from pathlib import Path

from app.models import DownloadResult, EpisodeMetadata, SourceMetadata, Transcript, TranscriptSegment
from app.roles import extract_role_candidates


def _download(
    title: str = "Guest Expert x Podcast Host | Example Show",
    channel: str | None = "Example Show",
    uploader: str | None = "Podcast Host",
    creator: str | None = None,
    description: str | None = None,
) -> DownloadResult:
    return DownloadResult(
        source=SourceMetadata(id="https://www.youtube.com/watch?v=abc123", url="https://www.youtube.com/watch?v=abc123"),
        episode=EpisodeMetadata(
            video_id="abc123",
            title=title,
            channel=channel,
            uploader=uploader,
            creator=creator,
            description=description,
            source_url="https://www.youtube.com/watch?v=abc123",
        ),
        episode_dir=Path("data/youtube/abc123"),
    )


def test_extract_role_candidates_preserves_actual_youtube_metadata_roles() -> None:
    candidates = extract_role_candidates(
        _download(channel="Cooking With Casey", uploader="Open Podcast Uploader", creator="Open Podcast Creator")
    )

    role_map = {(candidate.name, candidate.role) for candidate in candidates}
    assert ("Cooking With Casey", "publishing_channel") in role_map
    assert ("Open Podcast Uploader", "uploader") in role_map
    assert ("Open Podcast Creator", "creator") in role_map


def test_extract_role_candidates_uses_generic_title_pattern_for_possible_roles() -> None:
    candidates = extract_role_candidates(
        _download(title="Guest Expert x Podcast Host | Example Show", uploader="Podcast Host")
    )

    possible_host = [
        candidate for candidate in candidates if candidate.name == "Podcast Host" and candidate.role == "possible_host"
    ]
    possible_guest = [
        candidate for candidate in candidates if candidate.name == "Guest Expert" and candidate.role == "possible_guest"
    ]
    assert possible_host
    assert possible_guest
    assert possible_host[0].evidence_source == "title_pattern"


def test_extract_role_candidates_uses_description_hosted_by_evidence() -> None:
    candidates = extract_role_candidates(
        _download(
            title="A Conversation About Local AI",
            description="A weekly interview series hosted by Casey Rivera with builders and researchers.",
        )
    )

    assert any(
        candidate.name == "Casey Rivera"
        and candidate.role == "host"
        and candidate.confidence >= 0.85
        and candidate.evidence_source == "youtube_description"
        for candidate in candidates
    )


def test_extract_role_candidates_uses_intro_as_possible_host_only() -> None:
    transcript = Transcript(
        video_id="abc123",
        segments=[
            TranscriptSegment(
                segment_id="abc123:segment:000000",
                video_id="abc123",
                start_time=0,
                end_time=4,
                text="Welcome back. I'm Morgan Lee and today we are discussing memory.",
            )
        ],
    )

    candidates = extract_role_candidates(_download(title="A Conversation About Memory"), transcript)

    assert any(
        candidate.name == "Morgan Lee"
        and candidate.role == "possible_host"
        and candidate.evidence_source == "transcript_intro"
        for candidate in candidates
    )
