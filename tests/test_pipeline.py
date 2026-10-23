"""End-to-end orchestration test with in-memory ports, no API or ASR downloads."""
from pathlib import Path

from app.batching import TemporalBatcher
from app.models import (
    Batch, BatchAnalysis, ClassEvaluation, Confidence, Evidence, EvidencePolarity,
    EvidenceSource, Frame, Signal, SignalRating, Transcript, TranscriptSegment, VideoInfo,
)
from app.pipeline import ClassQualityPipeline
from app.rubric import Rubric
from app.settings import ROOT
from conftest import make_evaluation


class FakeMedia:
    def inspect(self, video: Path) -> VideoInfo:
        return VideoInfo(duration_seconds=130, has_audio=True)

    def extract(self, video: Path, output_dir: Path) -> tuple[list[Frame], Path]:
        return ([Frame(frame_id="f0", timestamp_seconds=0, path=Path("/tmp/unneeded"))],
                output_dir / "audio.wav")


class FakeTranscriber:
    def transcribe(self, audio_path: Path, duration_seconds: float) -> Transcript:
        return Transcript(duration_seconds=duration_seconds, language="en", segments=[
            TranscriptSegment(start=1, end=5, raw_text="Hello!", text="Hello!")
        ])


class FakeEvaluator:
    def __init__(self) -> None:
        self.calls: list[Batch] = []

    def analyse_batch(self, batch: Batch) -> BatchAnalysis:
        self.calls.append(batch)
        return BatchAnalysis(summary="Greeting", frame_observations=[], evidence=[Evidence(
            signal=Signal.RAPPORT, criterion_id="welcome", polarity=EvidencePolarity.POSITIVE,
            source=EvidenceSource.TRANSCRIPT, timestamp_seconds=batch.start,
            observation="A greeting", transcript_quote="Hello!",
        )])

    def evaluate_class(self, batches, duration_seconds: float) -> ClassEvaluation:
        assert len(batches) == 2
        return make_evaluation(2)


class FakeStore:
    def __init__(self) -> None:
        self.files: list[str] = []

    def save(self, run_id: str, filename: str, data: object) -> None:
        self.files.append(filename)


def test_pipeline_calls_and_artifact_outputs(tmp_path: Path) -> None:
    evaluator = FakeEvaluator()
    store = FakeStore()
    pipeline = ClassQualityPipeline(
        media=FakeMedia(), transcription=FakeTranscriber(),
        batching=TemporalBatcher(120), evaluator=evaluator,
        rubric=Rubric(ROOT / "config" / "rubric.json"), artifacts=store,
        max_duration_seconds=3600, model_name="gpt-4o",
    )
    result = pipeline.run(tmp_path / "dummy.mp4", tmp_path, "abc")
    assert 0 < result.score < 75
    assert result.label in {"Poor", "Average", "Good"}
    assert result.batch_count == 2
    assert len(evaluator.calls) == 2
    assert store.files == [
        "transcript.json", "batch_evidence.json", "batch_evidence.json", "assessment.json",
    ]
