"""Scoring is a pure function of whole-class judgements and configured weights."""
import json
import pytest

from app.models import CriterionStatus, IssueSeverity, Signal
from app.rubric import Rubric
from app.settings import ROOT
from conftest import make_evaluation


def apply(evaluation):
    return Rubric(ROOT / "config/rubric.json").apply(
        evaluation, run_id="test", model="gpt-4o", duration_seconds=120, batch_count=1,
    )


def test_label_cutoffs() -> None:
    rubric = Rubric(ROOT / "config/rubric.json")
    for score, label in [(0, "Poor"), (24, "Poor"), (25, "Average"), (49, "Average"),
                         (50, "Good"), (74, "Good"), (75, "Excellent"), (100, "Excellent")]:
        assert rubric._label(score) == label


def test_original_signal_weights_preserved() -> None:
    data = json.loads((ROOT / "config/rubric.json").read_text())
    assert data["weights"] == {
        "interaction": 25, "rapport": 20, "communication": 20,
        "structure": 15, "homework": 10, "visual": 10,
    }
    assert data["fulfilment_points"] == {
        "consistent": 100, "partial": 50, "not_met": 0, "unverified": 0,
    }


def test_all_unverified_zero_credit() -> None:
    result = apply(make_evaluation(0))
    assert result.score == 0 and result.label == "Poor"
    assert all(r.score == 0 for r in result.ratings)


def test_all_consistent_full_credit() -> None:
    result = apply(make_evaluation())
    assert result.score == 100 and result.label == "Excellent"


def test_partial_credit_with_positive_and_negative_events() -> None:
    evaluation = make_evaluation(0)
    for rating in evaluation.ratings:
        for item in rating.criteria:
            item.status = CriterionStatus.PARTIAL
            item.issue_severity = IssueSeverity.MODERATE
            item.positive_timestamps = [10.0]
            item.negative_timestamps = [50.0]
    result = apply(evaluation)
    assert result.score == 50
    assert result.label == "Good"
    assert all(r.score == 50 for r in result.ratings)
    assert all(c.positive_timestamps == [10.0] and c.negative_timestamps == [50.0]
               for r in result.ratings for c in r.criterion_deductions)


def test_one_negative_event_does_not_zero_consistent_credit() -> None:
    evaluation = make_evaluation()
    item = evaluation.ratings[0].criteria[0]
    item.negative_timestamps = [40.0]
    item.issue_severity = IssueSeverity.MINOR
    result = apply(evaluation)
    assert result.ratings[0].criterion_deductions[0].points_awarded == 20
    assert result.score == 100


def test_partial_criterion_is_half_of_its_available_points() -> None:
    evaluation = make_evaluation(0)
    item = evaluation.ratings[0].criteria[0]
    item.status = CriterionStatus.PARTIAL
    item.positive_timestamps = [10.0]
    item.issue_severity = IssueSeverity.MODERATE
    result = apply(evaluation)
    assert result.ratings[0].score == 10  # 50% of one 20-point interaction criterion
    assert result.ratings[0].criterion_deductions[0].points_deducted == 10


def test_missing_proof_cannot_earn_partial_credit() -> None:
    evaluation = make_evaluation(0)
    evaluation.ratings[0].criteria[0].status = CriterionStatus.PARTIAL
    with pytest.raises(ValueError, match="requires cited positive evidence"):
        apply(evaluation)


def test_missing_signal_is_rejected() -> None:
    evaluation = make_evaluation(0)
    evaluation.ratings.pop()
    with pytest.raises(ValueError, match="every rubric signal"):
        apply(evaluation)


def test_two_synthetic_classes_have_distinct_scores() -> None:
    # Not empirical accuracy: protects against collapse of partial and consistent.
    mostly_partial = make_evaluation()
    for rating in mostly_partial.ratings:
        for item in rating.criteria:
            item.status = CriterionStatus.PARTIAL
    assert apply(mostly_partial).score < apply(make_evaluation()).score
