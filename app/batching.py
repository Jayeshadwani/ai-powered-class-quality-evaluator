"""Align video frames and ASR utterances to non-overlapping 2-minute windows."""
from __future__ import annotations

import math

from app.models import Batch, Frame, TranscriptSegment


class TemporalBatcher:
    """Build windows, preserving utterances that overlap a window boundary."""

    def __init__(self, batch_seconds: int) -> None:
        if batch_seconds <= 0:
            raise ValueError("batch_seconds must be positive")
        self.batch_seconds = batch_seconds

    def build(
        self, duration_seconds: float, frames: list[Frame],
        transcript: list[TranscriptSegment],
    ) -> list[Batch]:
        """Return chronologically ordered batches; one utterance may overlap two."""
        if duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")
        batches: list[Batch] = []
        for index in range(math.ceil(duration_seconds / self.batch_seconds)):
            start = float(index * self.batch_seconds)
            end = min(duration_seconds, start + self.batch_seconds)
            window_frames = [f for f in frames if start <= f.timestamp_seconds < end]
            window_text = [s for s in transcript if s.start < end and s.end > start]
            batches.append(Batch(
                batch_id=index + 1, start=start, end=end,
                frames=window_frames, transcript=window_text,
            ))
        return batches
