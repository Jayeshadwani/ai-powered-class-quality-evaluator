"""GPT-4o implementation of the EvaluationPort; prompts and schema are injectable."""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Any

from app.models import (
    AnalysedBatch,
    Batch,
    BatchAnalysis,
    ClassEvaluation,
    FrameObservation,
    FrameStatus,
    Signal,
    CriterionStatus,
    CriterionAssessment,
    EvidencePolarity,
    IssueSeverity,
)
from app.prompts import BATCH_SYSTEM_PROMPT, CLASS_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class ModelEvaluationError(RuntimeError):
    """Indicates a model refusal, invalid response or out-of-window evidence."""


class GPT4oEvaluator:
    """Extract per-batch evidence and evaluate class-level signal ratings."""

    def __init__(
        self,
        api_key: str,
        model: str,
        image_detail: str,
        timeout_seconds: int,
        criteria: dict[str, list[dict[str, str]]],
    ) -> None:
        if not api_key or api_key == "replace_with_your_openai_api_key":
            raise ValueError("Set a valid OPENAI_API_KEY in .env")
        # Delayed import allows offline tests without requiring the OpenAI client.
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, timeout=timeout_seconds, max_retries=2)
        self.model = model
        self.image_detail = image_detail
        self.criteria = criteria

    def analyse_batch(self, batch: Batch) -> BatchAnalysis:
        """Send exactly this batch's frames and utterances to the multimodal LLM."""
        # Clip overlapping transcript timestamps to this batch.
        lines: list[dict[str, Any]] = [
            {
                "start": max(s.start, batch.start),
                "end": min(s.end, batch.end),
                "text": s.text,
            }
            for s in batch.transcript
            if s.start < batch.end and s.end > batch.start
        ]
        text_payload = {
            "batch_id": batch.batch_id,
            "start": batch.start,
            "end": batch.end,
            "required_criteria": self.criteria,
            "frames": [
                {"frame_id": f.frame_id, "timestamp_seconds": f.timestamp_seconds}
                for f in batch.frames
            ],
            "transcript": lines,
        }
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": "Input window JSON:\n"
                + json.dumps(text_payload, ensure_ascii=False),
            },
        ]
        for frame in batch.frames:
            raw_bytes = frame.path.read_bytes()
            encoded = base64.b64encode(raw_bytes).decode("ascii")
            content.extend(
                [
                    {
                        "type": "text",
                        "text": f"Frame {frame.frame_id} at {frame.timestamp_seconds:.2f}s",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded}",
                            "detail": self.image_detail,
                        },
                    },
                ]
            )
        result = self._parse(
            system_prompt=BATCH_SYSTEM_PROMPT,
            user_content=content,
            response_format=BatchAnalysis,
        )
        self._validate_batch(result, batch)
        # A generic signal name (e.g. "visual") is not a rubric criterion.
        # Ignore ambiguous observations rather than scoring them or aborting the upload.
        self._sanitize_batch_criteria(result)
        return result

    def _sanitize_batch_criteria(self, result: BatchAnalysis) -> None:
        """Keep only rubric-matched evidence; safely mark ambiguous frames unknown."""
        valid_evidence = []
        for item in result.evidence:
            allowed = {
                criterion["id"] for criterion in self.criteria[item.signal.value]
            }
            if item.criterion_id in allowed:
                valid_evidence.append(item)
            else:
                logger.warning(
                    "Discarded non-rubric evidence: signal=%s criterion_id=%s",
                    item.signal.value,
                    item.criterion_id,
                )
        result.evidence = valid_evidence

        allowed_visual = {
            criterion["id"] for criterion in self.criteria[Signal.VISUAL.value]
        }
        for observation in result.frame_observations:
            if observation.criterion_id in allowed_visual:
                continue
            if (
                observation.criterion_id != "none"
                or observation.status != FrameStatus.UNVERIFIABLE
            ):
                logger.warning(
                    "Unverified frame %s: expected a visual criterion, got %r",
                    observation.frame_id,
                    observation.criterion_id,
                )
            observation.criterion_id = "none"
            observation.status = FrameStatus.UNVERIFIABLE
            observation.observation = "No specific rubric criterion verified; visual observation excluded from scoring."

    def evaluate_class(
        self,
        batches: list[AnalysedBatch],
        duration_seconds: float,
    ) -> ClassEvaluation:
        """Judge full-class coverage and issue severity; ground both evidence polarities."""
        # Aggregation uses the full evidence ledger, not a mean of batch scores.
        positive: dict[tuple[Signal, str], list[float]] = {}
        negative: dict[tuple[Signal, str], list[float]] = {}
        seen_batches: dict[tuple[Signal, str], set[int]] = {}
        for batch in batches:
            for event in batch.analysis.evidence:
                key = (event.signal, event.criterion_id)
                if event.polarity == EvidencePolarity.POSITIVE:
                    positive.setdefault(key, []).append(event.timestamp_seconds)
                elif event.polarity == EvidencePolarity.NEGATIVE:
                    negative.setdefault(key, []).append(event.timestamp_seconds)
                seen_batches.setdefault(key, set()).add(batch.batch_id)
            for obs in batch.analysis.frame_observations:
                timestamp = batch.frame_timestamps.get(obs.frame_id)
                if timestamp is None or obs.criterion_id == "none":
                    continue
                key = (Signal.VISUAL, obs.criterion_id)
                if obs.status == FrameStatus.POSITIVE:
                    positive.setdefault(key, []).append(timestamp)
                elif obs.status == FrameStatus.ISSUE:
                    negative.setdefault(key, []).append(timestamp)
                if obs.status != FrameStatus.UNVERIFIABLE:
                    seen_batches.setdefault(key, set()).add(batch.batch_id)

        criterion_coverage = []
        for signal in Signal:
            for criterion in self.criteria[signal.value]:
                key = (signal, criterion["id"])
                criterion_coverage.append(
                    {
                        "signal": signal.value,
                        "criterion_id": criterion["id"],
                        "positive_count": len(positive.get(key, [])),
                        "negative_count": len(negative.get(key, [])),
                        "batches_with_observations": len(seen_batches.get(key, set())),
                        "total_batches": len(batches),
                    }
                )
        payload = {
            "duration_seconds": duration_seconds,
            "required_criteria": self.criteria,
            "criterion_coverage": criterion_coverage,
            "batches": [
                {
                    "batch_id": b.batch_id,
                    "start": b.start,
                    "end": b.end,
                    "summary": b.analysis.summary,
                    "frame_timestamps": b.frame_timestamps,
                    "evidence": [
                        e.model_dump(mode="json") for e in b.analysis.evidence
                    ],
                    "frame_observations": [
                        f.model_dump(mode="json") for f in b.analysis.frame_observations
                    ],
                }
                for b in batches
            ],
        }
        result = self._parse(
            system_prompt=CLASS_SYSTEM_PROMPT,
            user_content=[
                {"type": "text", "text": json.dumps(payload, ensure_ascii=False)}
            ],
            response_format=ClassEvaluation,
        )
        if len(result.ratings) != len(Signal) or {
            r.signal for r in result.ratings
        } != set(Signal):
            raise ModelEvaluationError(
                "GPT-4o did not rate exactly the six required signals"
            )
        for rating in result.ratings:
            required = {c["id"] for c in self.criteria[rating.signal.value]}
            received = [c.criterion_id for c in rating.criteria]
            if len(received) != len(set(received)) or not set(received).issubset(
                required
            ):
                raise ModelEvaluationError(
                    f"Invalid or duplicate criteria for {rating.signal.value}"
                )
            for missing_id in sorted(required - set(received)):
                rating.criteria.append(
                    CriterionAssessment(
                        criterion_id=missing_id,
                        status=CriterionStatus.UNVERIFIED,
                        issue_severity=IssueSeverity.UNKNOWN,
                        positive_timestamps=[],
                        negative_timestamps=[],
                        observation="No criterion judgement returned by the model.",
                    )
                )
            for item in rating.criteria:
                key = (rating.signal, item.criterion_id)
                # Never allow guessed timestamps to justify a positive or negative claim.
                item.positive_timestamps = self._ground_timestamps(
                    item.positive_timestamps,
                    positive.get(key, []),
                    duration_seconds,
                )
                item.negative_timestamps = self._ground_timestamps(
                    item.negative_timestamps,
                    negative.get(key, []),
                    duration_seconds,
                )
                if (
                    item.status in (CriterionStatus.CONSISTENT, CriterionStatus.PARTIAL)
                    and not item.positive_timestamps
                ):
                    item.status = CriterionStatus.UNVERIFIED
                    item.issue_severity = IssueSeverity.UNKNOWN
                    item.observation = (
                        "No positive evidence matched this criterion in the batch ledger; "
                        "class-level credit withheld."
                    )
                # If a negative judgement cites events that cannot be grounded,
                # it cannot claim observed failure. Missing credit remains unverified.
                elif (
                    item.status == CriterionStatus.NOT_MET
                    and not item.negative_timestamps
                ):
                    item.status = CriterionStatus.UNVERIFIED
                    item.issue_severity = IssueSeverity.UNKNOWN
                    item.observation = (
                        "No cited negative event was grounded; criterion unverified."
                    )
                # Do NOT let one negative observation override all positive evidence.
                # GPT judges its severity/frequency, and Python grants partial credit.
        return result

    @staticmethod
    def _ground_timestamps(
        reported: list[float],
        known: list[float],
        duration_seconds: float,
    ) -> list[float]:
        """Keep only criterion-matched, within-recording citations from the ledger."""
        return sorted(
            {
                round(nearest, 3)
                for timestamp in reported
                if 0 <= timestamp <= duration_seconds
                if known
                for nearest in [min(known, key=lambda k: abs(k - timestamp))]
                if abs(nearest - timestamp) <= 0.5
            }
        )

    def _parse(
        self,
        system_prompt: str,
        user_content: list[dict[str, Any]],
        response_format: Any,
    ) -> Any:
        """One structured GPT-4o call; SDK parses directly into Pydantic."""
        try:
            completion = self.client.beta.chat.completions.parse(
                model=self.model,
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                response_format=response_format,
            )
            message = completion.choices[0].message
            if message.parsed is None:
                raise ModelEvaluationError(
                    f"Model did not return structured output: {message.refusal or 'empty'}"
                )
            return message.parsed
        except ModelEvaluationError:
            raise
        except Exception as exc:
            raise ModelEvaluationError(
                "GPT-4o request or schema parsing failed"
            ) from exc

    @staticmethod
    def _validate_batch(result: BatchAnalysis, batch: Batch) -> None:
        """Reject fabricated frame IDs/timestamps; pad missing frames as unknown."""
        known_ids = {f.frame_id for f in batch.frames}
        observed_ids: set[str] = set()
        for obs in result.frame_observations:
            if obs.frame_id not in known_ids:
                raise ModelEvaluationError(
                    f"Unknown frame ID returned by model: {obs.frame_id}"
                )
            if obs.frame_id in observed_ids:
                raise ModelEvaluationError(
                    f"Duplicated frame ID returned: {obs.frame_id}"
                )
            observed_ids.add(obs.frame_id)
        # Incomplete model coverage is recorded as unknown, never invented evidence.
        for frame in batch.frames:
            if frame.frame_id not in observed_ids:
                result.frame_observations.append(
                    FrameObservation(
                        frame_id=frame.frame_id,
                        criterion_id="none",
                        status=FrameStatus.UNVERIFIABLE,
                        focused_signal="no_model_observation",
                        observation="Model did not assess this sampled frame.",
                    )
                )
        valid_events = []

        for event in result.evidence:
            if batch.start <= event.timestamp_seconds <= batch.end:
                valid_events.append(event)
            else:
                logger.warning(
                    "Discarding out-of-window evidence: "
                    "batch=%s window=[%.2f, %.2f] "
                    "timestamp=%.3f signal=%s criterion=%s",
                    batch.batch_id,
                    batch.start,
                    batch.end,
                    event.timestamp_seconds,
                    event.signal.value,
                    event.criterion_id,
                )

        result.evidence = valid_events
