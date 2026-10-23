"""Orchestration only: video -> transcription -> batch evidence -> final rating."""
from __future__ import annotations

from pathlib import Path

from app.batching import TemporalBatcher
from app.models import AnalysedBatch, PredictionResponse
from app.ports import ArtifactPort, EvaluationPort, MediaPort, TranscriptionPort
from app.rubric import Rubric


class ClassQualityPipeline:
    """Depend on substitutable service interfaces rather than concrete SDKs."""

    def __init__(
        self, media: MediaPort, transcription: TranscriptionPort,
        batching: TemporalBatcher, evaluator: EvaluationPort,
        rubric: Rubric, artifacts: ArtifactPort,
        max_duration_seconds: int, model_name: str,
    ) -> None:
        self.media = media
        self.transcription = transcription
        self.batching = batching
        self.evaluator = evaluator
        self.rubric = rubric
        self.artifacts = artifacts
        self.max_duration_seconds = max_duration_seconds
        self.model_name = model_name

    def run(self, video_path: Path, workspace: Path, run_id: str) -> PredictionResponse:
        """Process one recording synchronously. Fail rather than silently omit windows."""
        info = self.media.inspect(video_path)
        if not info.has_audio:
            raise ValueError("Video must contain an audio track")
        if info.duration_seconds > self.max_duration_seconds:
            raise ValueError(f"Video must be <= {self.max_duration_seconds / 60:g} minutes")
        frames, audio = self.media.extract(video_path, workspace)
        transcript = self.transcription.transcribe(audio, info.duration_seconds)
        self.artifacts.save(run_id, "transcript.json", transcript)
        windows = self.batching.build(info.duration_seconds, frames, transcript.segments)
        analyses: list[AnalysedBatch] = []
        for window in windows:
            analysis = self.evaluator.analyse_batch(window)
            analyses.append(AnalysedBatch(
                batch_id=window.batch_id, start=window.start, end=window.end,
                analysis=analysis,
                frame_timestamps={f.frame_id: f.timestamp_seconds for f in window.frames},
            ))
            # Save progress for debugging; no raw frames or audio are retained.
            self.artifacts.save(run_id, "batch_evidence.json", analyses)
        final = self.evaluator.evaluate_class(analyses, info.duration_seconds)
        response = self.rubric.apply(
            final, run_id=run_id, model=self.model_name,
            duration_seconds=info.duration_seconds, batch_count=len(windows),
        )
        self.artifacts.save(run_id, "assessment.json", response)
        return response
