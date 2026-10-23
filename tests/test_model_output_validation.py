"""Do not silently accept model-fabricated timestamps or frame references."""

from pathlib import Path

import pytest

from app.gpt_evaluator import GPT4oEvaluator, ModelEvaluationError
from app.models import (
    Batch,
    BatchAnalysis,
    Evidence,
    EvidencePolarity,
    EvidenceSource,
    Frame,
    FrameObservation,
    FrameStatus,
    Signal,
)


def test_missing_frame_is_marked_unverifiable_not_invented() -> None:
    batch = Batch(
        batch_id=1,
        start=0,
        end=120,
        frames=[
            Frame(
                frame_id="f1",
                timestamp_seconds=0,
                path=Path("unused.jpg"),
            )
        ],
        transcript=[],
    )
    analysis = BatchAnalysis(summary="Nothing", frame_observations=[], evidence=[])
    GPT4oEvaluator._validate_batch(analysis, batch)
    assert analysis.frame_observations[0].status == FrameStatus.UNVERIFIABLE


def test_out_of_range_event_is_discarded() -> None:
    batch = Batch(
        batch_id=1,
        start=0,
        end=120,
        frames=[],
        transcript=[],
    )

    analysis = BatchAnalysis(
        summary="Timestamp validation",
        frame_observations=[],
        evidence=[
            Evidence(
                signal=Signal.HOMEWORK,
                criterion_id="assigned",
                polarity=EvidencePolarity.POSITIVE,
                source=EvidenceSource.TRANSCRIPT,
                timestamp_seconds=130,
                observation="Outside window",
                transcript_quote="Homework",
            ),
            Evidence(
                signal=Signal.HOMEWORK,
                criterion_id="assigned",
                polarity=EvidencePolarity.POSITIVE,
                source=EvidenceSource.TRANSCRIPT,
                timestamp_seconds=110,
                observation="Inside window",
                transcript_quote="Homework",
            ),
        ],
    )

    GPT4oEvaluator._validate_batch(analysis, batch)

    assert len(analysis.evidence) == 1
    assert analysis.evidence[0].timestamp_seconds == 110


def test_generic_visual_signal_is_unverified_not_a_500() -> None:
    from app.rubric import Rubric
    from app.settings import ROOT

    evaluator = object.__new__(GPT4oEvaluator)
    evaluator.criteria = Rubric(ROOT / "config/rubric.json").criteria
    analysis = BatchAnalysis(
        summary="Classroom image",
        frame_observations=[
            FrameObservation(
                frame_id="f1",
                criterion_id="visual",
                status=FrameStatus.POSITIVE,
                focused_signal="visual",
                observation="Generic visual observation",
            )
        ],
        evidence=[],
    )
    evaluator._sanitize_batch_criteria(analysis)
    assert analysis.frame_observations[0].criterion_id == "none"
    assert analysis.frame_observations[0].status == FrameStatus.UNVERIFIABLE


def test_valid_visual_criterion_is_preserved() -> None:
    from app.rubric import Rubric
    from app.settings import ROOT

    evaluator = object.__new__(GPT4oEvaluator)
    evaluator.criteria = Rubric(ROOT / "config/rubric.json").criteria
    analysis = BatchAnalysis(
        summary="Tutor visible",
        frame_observations=[
            FrameObservation(
                frame_id="f1",
                criterion_id="tutor_visibility",
                status=FrameStatus.POSITIVE,
                focused_signal="visual",
                observation="Face clearly visible",
            )
        ],
        evidence=[],
    )
    evaluator._sanitize_batch_criteria(analysis)
    assert analysis.frame_observations[0].criterion_id == "tutor_visibility"
    assert analysis.frame_observations[0].status == FrameStatus.POSITIVE


def test_generic_signal_event_is_not_scored() -> None:
    from app.rubric import Rubric
    from app.settings import ROOT

    evaluator = object.__new__(GPT4oEvaluator)
    evaluator.criteria = Rubric(ROOT / "config/rubric.json").criteria
    analysis = BatchAnalysis(
        summary="Generic evidence",
        frame_observations=[],
        evidence=[
            Evidence(
                signal=Signal.VISUAL,
                criterion_id="visual",
                polarity=EvidencePolarity.POSITIVE,
                source=EvidenceSource.VIDEO,
                timestamp_seconds=10,
                observation="Generic visual",
                transcript_quote="",
            )
        ],
    )
    evaluator._sanitize_batch_criteria(analysis)
    assert analysis.evidence == []
