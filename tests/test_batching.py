"""Batch boundaries, frame assignments, and transcript alignment."""
from pathlib import Path

from app.batching import TemporalBatcher
from app.models import Frame, TranscriptSegment


def test_sixty_minutes_produces_thirty_buckets_and_twelve_frames_each() -> None:
    frames = [Frame(frame_id=f"f{i}", timestamp_seconds=i * 10.0, path=Path("ignored.jpg"))
              for i in range(360)]
    segments = [TranscriptSegment(start=15, end=25, raw_text="Hello", text="Hello")]
    result = TemporalBatcher(120).build(3600, frames, segments)
    assert len(result) == 30
    assert all(len(batch.frames) == 12 for batch in result)
    assert result[0].end == result[1].start == 120
    assert len(result[0].transcript) == 1
    assert len(result[1].transcript) == 0


def test_utterance_overlapping_boundary_kept_in_both_buckets() -> None:
    segment = TranscriptSegment(start=118.0, end=123.0, raw_text="Continue", text="Continue")
    result = TemporalBatcher(120).build(240, [], [segment])
    assert len(result[0].transcript) == 1
    assert len(result[1].transcript) == 1


def test_partial_last_window_and_last_frame() -> None:
    result = TemporalBatcher(120).build(125, [
        Frame(frame_id="last", timestamp_seconds=120, path=Path("missing.jpg"))
    ], [])
    assert len(result) == 2
    assert result[-1].end == 125
    assert result[-1].frames[0].frame_id == "last"
