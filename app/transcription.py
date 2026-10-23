"""Local ASR adapter. No speaker diarization or speaker naming is attempted."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from app.models import Transcript, TranscriptSegment


def clean_transcript_text(raw_text: str) -> str:
    """Remove excess whitespace only; retain questions, filler words and repetitions."""
    return re.sub(r"\s+", " ", raw_text).strip()


class FasterWhisperTranscriber:
    """Lazily initialize faster-whisper to avoid model download during imports/tests."""

    def __init__(
        self, model_name: str, device: str, compute_type: str,
        language: str | None, min_silence_ms: int,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.min_silence_ms = min_silence_ms
        self._model: Any = None

    def transcribe(self, audio_path: Path, duration_seconds: float) -> Transcript:
        """Transcribe full audio once; preserve original ASR segment timestamps."""
        if self._model is None:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(
                self.model_name, device=self.device, compute_type=self.compute_type,
            )
        segments, info = self._model.transcribe(
            str(audio_path), language=self.language, beam_size=5, temperature=0.0,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": self.min_silence_ms, "speech_pad_ms": 300},
            word_timestamps=True, condition_on_previous_text=False,
        )
        entries = [
            TranscriptSegment(
                start=round(s.start, 3), end=round(s.end, 3),
                raw_text=s.text, text=clean_transcript_text(s.text),
            )
            for s in segments if clean_transcript_text(s.text)
        ]
        if not entries:
            raise ValueError("No intelligible speech was transcribed")
        return Transcript(
            language=info.language or self.language or "unknown",
            duration_seconds=duration_seconds,
            segments=entries,
        )
