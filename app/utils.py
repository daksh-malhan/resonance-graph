from __future__ import annotations

import json
import logging
import shutil
import sys
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from app.errors import AppError

T = TypeVar("T", bound=BaseModel)


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def require_executable(name: str, install_hint: str) -> str:
    path = shutil.which(name)
    if path:
        return path

    for candidate_dir in [Path(sys.prefix) / "bin", Path(sys.executable).parent]:
        venv_path = candidate_dir / name
        if venv_path.exists() and venv_path.is_file():
            return str(venv_path)

    if name == "ffmpeg":
        try:
            import imageio_ffmpeg
        except ImportError:
            pass
        else:
            bundled_path = Path(imageio_ffmpeg.get_ffmpeg_exe())
            if bundled_path.exists() and bundled_path.is_file():
                return str(bundled_path)

    raise AppError(f"Missing required executable '{name}'. {install_hint}")


def write_json(path: Path, value: BaseModel | list[BaseModel] | dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, BaseModel):
        payload = value.model_dump(mode="json")
    elif isinstance(value, list):
        payload = [
            item.model_dump(mode="json") if isinstance(item, BaseModel) else item for item in value
        ]
    else:
        payload = value
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def read_model(path: Path, model: type[T]) -> T:
    return model.model_validate_json(path.read_text())


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def format_timestamp(seconds: float) -> str:
    total = max(0, int(seconds))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"
