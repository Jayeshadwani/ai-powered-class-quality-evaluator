"""Deterministic fake class judgements; never call the model in unit tests."""
from app.models import (
    ClassEvaluation, Confidence, CriterionAssessment, CriterionStatus,
    IssueSeverity, Signal, SignalRating,
)
from app.rubric import Rubric
from app.settings import ROOT


def make_evaluation(consistent_count: int | None = None) -> ClassEvaluation:
    """Synthetic: optionally mark only the first N criteria per signal consistent."""
    rubric = Rubric(ROOT / "config/rubric.json")
    ratings = []
    for signal in Signal:
        checks = rubric.criteria[signal.value]
        ratings.append(SignalRating(
            signal=signal,
            criteria=[CriterionAssessment(
                criterion_id=criterion["id"],
                status=(CriterionStatus.CONSISTENT if consistent_count is None or i < consistent_count
                        else CriterionStatus.UNVERIFIED),
                issue_severity=(IssueSeverity.NONE if consistent_count is None or i < consistent_count
                                else IssueSeverity.UNKNOWN),
                positive_timestamps=([12.0] if consistent_count is None or i < consistent_count else []),
                negative_timestamps=[],
                observation="Synthetic criterion test",
            ) for i, criterion in enumerate(checks)],
            confidence=Confidence.HIGH,
            rationale="Synthetic example",
            limitations="",
        ))
    return ClassEvaluation(ratings=ratings, overall_observation="Synthetic", recommendations=[])
