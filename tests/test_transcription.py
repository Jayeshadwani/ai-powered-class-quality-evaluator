"""Transcript normalization should not erase teaching signals."""
from app.transcription import clean_transcript_text


def test_cleaning_only_removes_redundant_whitespace() -> None:
    raw = " Hello,  Gautam!  \n Um, um... Why?  "
    assert clean_transcript_text(raw) == "Hello, Gautam! Um, um... Why?"
