import json
from pathlib import Path

from app.youtube import parse_youtube_metadata


def test_parse_youtube_metadata_fixture(tmp_path: Path) -> None:
    fixture_path = Path(__file__).parent / "fixtures" / "youtube_info.json"
    info = json.loads(fixture_path.read_text())
    episode_dir = tmp_path / "abc123"
    episode_dir.mkdir()
    video_file = episode_dir / "abc123.mp4"
    video_file.write_text("placeholder")
    info_json = episode_dir / "abc123.info.json"
    info_json.write_text(json.dumps(info))

    metadata = parse_youtube_metadata(info, episode_dir)

    assert metadata.video_id == "abc123"
    assert metadata.title == "Example Creative Commons Talk"
    assert metadata.channel == "Open Media Channel"
    assert metadata.channel_id == "UCexample"
    assert metadata.channel_url == "https://www.youtube.com/channel/UCexample"
    assert metadata.uploader == "Open Media Uploader"
    assert metadata.uploader_id == "@openmedia"
    assert metadata.uploader_url == "https://www.youtube.com/@openmedia"
    assert metadata.creator == "Open Media Creator"
    assert metadata.local_video_path == video_file
    assert metadata.info_json_path == info_json
