"""Configuration loading and validation. Secrets live only in .env, never in source."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _positive_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be > 0")
    return value


@dataclass(frozen=True)
class Settings:
    """Immutable runtime settings read from environment variables."""

    api_key: str
    model: str
    image_detail: str
    openai_timeout_seconds: int
    whisper_model: str
    whisper_device: str
    whisper_compute_type: str
    whisper_language: str | None
    whisper_vad_min_silence_ms: int
    frame_interval_seconds: int
    batch_duration_seconds: int
    frame_width: int
    max_duration_minutes: int
    max_upload_mb: int
    runs_dir: Path
    save_results: bool

    @classmethod
    def from_environment(cls) -> Settings:
        """Load settings without requiring the OpenAI key until inference starts."""
        detail = os.getenv("OPENAI_IMAGE_DETAIL", "low")
        if detail not in {"low", "high", "auto"}:
            raise ValueError("OPENAI_IMAGE_DETAIL must be low, high, or auto")
        language = os.getenv("WHISPER_LANGUAGE", "en").strip()
        runs_path = Path(os.getenv("RUNS_DIR", "./runs")).expanduser()
        return cls(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            image_detail=detail,
            openai_timeout_seconds=_positive_int("OPENAI_TIMEOUT_SECONDS", 180),
            whisper_model=os.getenv("WHISPER_MODEL", "small"),
            whisper_device=os.getenv("WHISPER_DEVICE", "cpu"),
            whisper_compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
            whisper_language=language or None,
            whisper_vad_min_silence_ms=_positive_int("WHISPER_VAD_MIN_SILENCE_MS", 500),
            frame_interval_seconds=_positive_int("FRAME_INTERVAL_SECONDS", 10),
            batch_duration_seconds=_positive_int("BATCH_DURATION_SECONDS", 120),
            frame_width=_positive_int("FRAME_WIDTH", 640),
            max_duration_minutes=_positive_int("MAX_DURATION_MINUTES", 60),
            max_upload_mb=_positive_int("MAX_UPLOAD_MB", 1024),
            runs_dir=runs_path if runs_path.is_absolute() else ROOT / runs_path,
            save_results=os.getenv("SAVE_RESULTS", "true").lower() == "true",
        )
