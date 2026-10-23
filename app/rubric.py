"""Data-driven, deterministic scoring; the LLM only judges fulfilment of criteria."""
from __future__ import annotations

import json
from pathlib import Path

from app.models import (
    ClassEvaluation, CriterionDeduction, CriterionStatus, PredictionResponse,
    Signal, WeightedRating,
)


class Rubric:
    """Validate criteria, configurable partial credit, user weights and label bands."""

    def __init__(self, rubric_file: Path) -> None:
        data = json.loads(rubric_file.read_text(encoding="utf-8"))
        self.version: str = data["rubric_version"]
        self.weights: dict[str, int] = data["weights"]
        self.labels: list[dict[str, int | str]] = data["labels"]
        self.criteria: dict[str, list[dict[str, str]]] = data["criteria"]
        self.fulfilment_points: dict[str, int] = data["fulfilment_points"]
        expected = {signal.value for signal in Signal}
        if set(self.weights) != expected or sum(self.weights.values()) != 100:
            raise ValueError("Rubric must define all six signals with weights totaling 100")
        if any(type(w) is not int or w < 0 for w in self.weights.values()):
            raise ValueError("Weights must be non-negative integers")
        if set(self.criteria) != expected:
            raise ValueError("Every weighted signal must define its criteria")
        for signal, items in self.criteria.items():
            names = [item["id"] for item in items]
            if not names or len(names) != len(set(names)):
                raise ValueError(f"Criteria must be nonempty and unique: {signal}")
        if set(self.fulfilment_points) != {status.value for status in CriterionStatus}:
            raise ValueError("Exactly one fulfilment value required for each criterion status")
        if any(type(p) is not int or not 0 <= p <= 100 for p in self.fulfilment_points.values()):
            raise ValueError("Fulfilment points must be integers from 0 to 100")
        if not (self.fulfilment_points["consistent"] > self.fulfilment_points["partial"] >
                self.fulfilment_points["not_met"]):
            raise ValueError("Fulfilment points must descend: consistent > partial > not_met")
        for score in range(101):
            self._label(score)

    def _label(self, score: int) -> str:
        matches = [item["name"] for item in self.labels if item["min"] <= score <= item["max"]]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one label for score {score}")
        return str(matches[0])

    def apply(
        self, evaluation: ClassEvaluation, *, run_id: str,
        model: str, duration_seconds: float, batch_count: int,
    ) -> PredictionResponse:
        """Score verified fulfilment consistently; no single negative zeroes all credit."""
        by_signal = {r.signal.value: r for r in evaluation.ratings}
        if len(evaluation.ratings) != len(Signal) or set(by_signal) != set(self.weights):
            raise ValueError("Model must return one evaluation for every rubric signal")
        ratings: list[WeightedRating] = []
        weighted_total = 0.0
        for key, weight in self.weights.items():
            rating = by_signal[key]
            required_ids = {c["id"] for c in self.criteria[key]}
            reported_ids = [c.criterion_id for c in rating.criteria]
            if set(reported_ids) != required_ids or len(reported_ids) != len(required_ids):
                raise ValueError(f"Missing or duplicated criterion evaluations for {key}")
            points_per_criterion = 100 / len(required_ids)
            deductions: list[CriterionDeduction] = []
            earned = 0.0
            for criterion in rating.criteria:
                credit_pct = self.fulfilment_points[criterion.status.value]
                if criterion.status in (CriterionStatus.CONSISTENT, CriterionStatus.PARTIAL) and not criterion.positive_timestamps:
                    raise ValueError(f"Earned credit requires cited positive evidence: {criterion.criterion_id}")
                credit = points_per_criterion * credit_pct / 100
                earned += credit
                all_times = sorted(set(criterion.positive_timestamps + criterion.negative_timestamps))
                deductions.append(CriterionDeduction(
                    criterion_id=criterion.criterion_id,
                    status=criterion.status,
                    issue_severity=criterion.issue_severity,
                    points_possible=round(points_per_criterion, 3),
                    points_awarded=round(credit, 3),
                    points_deducted=round(points_per_criterion - credit, 3),
                    evidence_timestamps=all_times,
                    positive_timestamps=criterion.positive_timestamps,
                    negative_timestamps=criterion.negative_timestamps,
                    observation=criterion.observation,
                ))
            # Do not round individual criteria before aggregating; this avoids bias.
            signal_score = round(earned)
            contribution = earned * weight / 100
            weighted_total += contribution
            supporting = sorted({t for c in rating.criteria for t in c.positive_timestamps})
            ratings.append(WeightedRating(
                signal=rating.signal, score=signal_score, weight=weight,
                contribution=round(contribution, 3), confidence=rating.confidence,
                rationale=rating.rationale, supporting_timestamps=supporting,
                limitations=rating.limitations, criterion_deductions=deductions,
            ))
        score = round(weighted_total)
        return PredictionResponse(
            run_id=run_id, model=model, duration_seconds=duration_seconds,
            batch_count=batch_count, score=score, label=self._label(score),
            weights_version=self.version, ratings=ratings,
            overall_observation=evaluation.overall_observation,
            recommendations=evaluation.recommendations,
        )
