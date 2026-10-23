"""Small Protocol interfaces: replace an implementation without editing orchestration."""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

from app.models import (
    AnalysedBatch, Batch, BatchAnalysis, ClassEvaluation, Frame,
    Transcript, VideoInfo,
)


class MediaPort(Protocol):
    def inspect(self, video: Path) -> VideoInfo: ...
    def extract(self, video: Path, output_dir: Path) -> tuple[list[Frame], Path]: ...


class TranscriptionPort(Protocol):
    def transcribe(self, audio_path: Path, duration_seconds: float) -> Transcript: ...


class EvaluationPort(Protocol):
    def analyse_batch(self, batch: Batch) -> BatchAnalysis: ...
    def evaluate_class(self, batches: list[AnalysedBatch], duration_seconds: float) -> ClassEvaluation: ...


class ArtifactPort(Protocol):
    def save(self, run_id: str, filename: str, data: object) -> None: ...
