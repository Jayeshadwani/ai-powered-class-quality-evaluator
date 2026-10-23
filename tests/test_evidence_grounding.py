"""A model may judge frequency/severity, but it cannot fabricate citations."""
from __future__ import annotations

from app.gpt_evaluator import GPT4oEvaluator
from app.models import (
    AnalysedBatch, BatchAnalysis, CriterionStatus, Evidence, EvidencePolarity,
    EvidenceSource, IssueSeverity, Signal,
)
from app.rubric import Rubric
from app.settings import ROOT
from conftest import make_evaluation


def fake_evaluator(evaluation) -> GPT4oEvaluator:
    instance = object.__new__(GPT4oEvaluator)
    instance.criteria = Rubric(ROOT / "config/rubric.json").criteria
    instance._parse = lambda **kwargs: evaluation
    return instance


def batch_with_evidence(*items: Evidence) -> list[AnalysedBatch]:
    return [AnalysedBatch(
        batch_id=1, start=0, end=120, frame_timestamps={},
        analysis=BatchAnalysis(summary="synthetic", frame_observations=[], evidence=list(items)),
    )]


def evidence(criterion_id: str, polarity: EvidencePolarity, timestamp: float) -> Evidence:
    return Evidence(
        signal=Signal.INTERACTION, criterion_id=criterion_id,
        polarity=polarity, source=EvidenceSource.TRANSCRIPT,
        timestamp_seconds=timestamp, observation="synthetic test evidence",
        transcript_quote="Example sentence",
    )


def test_unrelated_criterion_cannot_justify_positive_credit() -> None:
    evaluation = make_evaluation(0)
    item = evaluation.ratings[0].criteria[0]
    item.status = CriterionStatus.CONSISTENT
    item.positive_timestamps = [30.0]
    result = fake_evaluator(evaluation).evaluate_class(
        batch_with_evidence(evidence("student_responses", EvidencePolarity.POSITIVE, 30.0)), 120,
    )
    assert result.ratings[0].criteria[0].status == CriterionStatus.UNVERIFIED
    assert result.ratings[0].criteria[0].positive_timestamps == []


def test_positive_and_negative_evidence_coexist_as_partial() -> None:
    evaluation = make_evaluation(0)
    item = evaluation.ratings[0].criteria[0]
    item.status = CriterionStatus.PARTIAL
    item.issue_severity = IssueSeverity.MODERATE
    item.positive_timestamps = [20.0]
    item.negative_timestamps = [45.0]
    result = fake_evaluator(evaluation).evaluate_class(
        batch_with_evidence(
            evidence("individual_questions", EvidencePolarity.POSITIVE, 20.0),
            evidence("individual_questions", EvidencePolarity.NEGATIVE, 45.0),
        ), 120,
    )
    c = result.ratings[0].criteria[0]
    assert c.status == CriterionStatus.PARTIAL
    assert c.positive_timestamps == [20.0] and c.negative_timestamps == [45.0]


def test_minor_negative_does_not_automatically_override_consistent() -> None:
    evaluation = make_evaluation(0)
    item = evaluation.ratings[0].criteria[0]
    item.status = CriterionStatus.CONSISTENT
    item.issue_severity = IssueSeverity.MINOR
    item.positive_timestamps = [20.0]
    item.negative_timestamps = [45.0]
    result = fake_evaluator(evaluation).evaluate_class(
        batch_with_evidence(
            evidence("individual_questions", EvidencePolarity.POSITIVE, 20.0),
            evidence("individual_questions", EvidencePolarity.NEGATIVE, 45.0),
        ), 120,
    )
    assert result.ratings[0].criteria[0].status == CriterionStatus.CONSISTENT


def test_fabricated_negative_timestamp_does_not_pass() -> None:
    evaluation = make_evaluation(0)
    item = evaluation.ratings[0].criteria[0]
    item.status = CriterionStatus.NOT_MET
    item.negative_timestamps = [11.0]
    result = fake_evaluator(evaluation).evaluate_class(
        batch_with_evidence(evidence("individual_questions", EvidencePolarity.POSITIVE, 11.0)), 120,
    )
    assert result.ratings[0].criteria[0].status == CriterionStatus.UNVERIFIED
    assert result.ratings[0].criteria[0].negative_timestamps == []


def test_nearby_timestamp_snaps_to_actual_evidence_time() -> None:
    evaluation = make_evaluation(0)
    item = evaluation.ratings[0].criteria[0]
    item.status = CriterionStatus.PARTIAL
    item.positive_timestamps = [20.3]
    result = fake_evaluator(evaluation).evaluate_class(
        batch_with_evidence(evidence("individual_questions", EvidencePolarity.POSITIVE, 20.1)), 120,
    )
    assert result.ratings[0].criteria[0].positive_timestamps == [20.1]
