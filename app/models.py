"""Types and API schemas shared by preprocessing, inference and presentation."""
from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field


class Signal(str, Enum):
    INTERACTION = "interaction"
    RAPPORT = "rapport"
    COMMUNICATION = "communication"
    STRUCTURE = "structure"
    HOMEWORK = "homework"
    VISUAL = "visual"


class EvidencePolarity(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    UNCERTAIN = "uncertain"


class EvidenceSource(str, Enum):
    TRANSCRIPT = "transcript"
    VIDEO = "video"
    BOTH = "both"


class FrameStatus(str, Enum):
    POSITIVE = "positive"
    ISSUE = "issue"
    UNVERIFIABLE = "unverifiable"


class CriterionStatus(str, Enum):
    """Whole-class fulfilment judgement, not a per-event yes/no vote."""
    CONSISTENT = "consistent"
    PARTIAL = "partial"
    NOT_MET = "not_met"
    UNVERIFIED = "unverified"


class IssueSeverity(str, Enum):
    """Qualitative importance of observed negative evidence (not its count)."""
    NONE = "none"
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"
    UNKNOWN = "unknown"


class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TranscriptSegment(BaseModel):
    start: float
    end: float
    raw_text: str
    text: str


class Transcript(BaseModel):
    language: str
    duration_seconds: float
    segments: list[TranscriptSegment]


class Frame(BaseModel):
    frame_id: str
    timestamp_seconds: float
    path: Path


class Batch(BaseModel):
    batch_id: int
    start: float
    end: float
    frames: list[Frame]
    transcript: list[TranscriptSegment]


# Strict Pydantic schemas parsed by GPT-4o. The application validates ranges
# and evidence references separately, because not all JSON Schema keywords
# are supported by Structured Outputs.
class FrameObservation(BaseModel):
    frame_id: str
    criterion_id: str  # Specific VISUAL criterion, or "none" when unverifiable
    status: FrameStatus
    focused_signal: str
    observation: str


class Evidence(BaseModel):
    signal: Signal
    criterion_id: str  # Exact rubric criterion supported or contradicted
    polarity: EvidencePolarity
    source: EvidenceSource
    timestamp_seconds: float
    observation: str
    transcript_quote: str


class BatchAnalysis(BaseModel):
    summary: str
    frame_observations: list[FrameObservation]
    evidence: list[Evidence]


class AnalysedBatch(BaseModel):
    batch_id: int
    start: float
    end: float
    analysis: BatchAnalysis
    frame_timestamps: dict[str, float] = Field(default_factory=dict)


class CriterionAssessment(BaseModel):
    """A whole-class judgement tied to positive AND negative evidence events."""
    criterion_id: str
    status: CriterionStatus
    issue_severity: IssueSeverity
    positive_timestamps: list[float]
    negative_timestamps: list[float]
    observation: str


class SignalRating(BaseModel):
    """GPT judges coverage and severity; Python exclusively calculates points."""
    signal: Signal
    criteria: list[CriterionAssessment]
    confidence: Confidence
    rationale: str
    limitations: str


class ClassEvaluation(BaseModel):
    ratings: list[SignalRating]
    overall_observation: str
    recommendations: list[str]


class CriterionDeduction(BaseModel):
    criterion_id: str
    status: CriterionStatus
    issue_severity: IssueSeverity
    points_possible: float
    points_awarded: float
    points_deducted: float
    evidence_timestamps: list[float]
    positive_timestamps: list[float]
    negative_timestamps: list[float]
    observation: str


class WeightedRating(BaseModel):
    signal: Signal
    score: int
    weight: int
    contribution: float
    confidence: Confidence
    rationale: str
    supporting_timestamps: list[float]
    limitations: str
    criterion_deductions: list[CriterionDeduction]


class PredictionResponse(BaseModel):
    run_id: str
    model: str
    duration_seconds: float
    batch_count: int
    score: int
    label: str
    weights_version: str
    ratings: list[WeightedRating]
    overall_observation: str
    recommendations: list[str]


class VideoInfo(BaseModel):
    duration_seconds: float
    has_audio: bool
