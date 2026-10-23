"""Integration regression for the faster-whisper/PyAV decoder boundary."""

from __future__ import annotations

import wave
from pathlib import Path

import pytest


def test_faster_whisper_decodes_wav_with_pinned_pyav(tmp_path: Path) -> None:
    """Catch PyAV metadata_errors signature regressions without model download."""
    pytest.importorskip("av")
    audio = pytest.importorskip("faster_whisper.audio")

    wav_path = tmp_path / "silence.wav"
    with wave.open(str(wav_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 16000)

    samples = audio.decode_audio(str(wav_path), sampling_rate=16000)
    assert len(samples) == 16000
