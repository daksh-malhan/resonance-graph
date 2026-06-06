from pathlib import Path

from app.config import AppConfig


def test_config_defaults_are_valid(tmp_path: Path) -> None:
    config = AppConfig(
        youtube_download_dir=tmp_path / "youtube",
        audio_output_dir=tmp_path / "audio",
        transcript_output_dir=tmp_path / "transcripts",
        chunk_output_dir=tmp_path / "chunks",
        embedding_cache_dir=tmp_path / "embeddings",
    )

    assert config.neo4j_uri == "bolt://localhost:7687"
    assert config.chunk_size > config.chunk_overlap
    config.ensure_directories()
    assert config.youtube_download_dir.exists()
    assert config.embedding_cache_dir.exists()


def test_config_expands_paths() -> None:
    config = AppConfig(youtube_download_dir="~/local-graphrag-test")
    assert isinstance(config.youtube_download_dir, Path)
    assert str(config.youtube_download_dir).startswith(str(Path.home()))
