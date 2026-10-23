"""Optional real FFmpeg integration test; requires ffmpeg and ffprobe in PATH."""
import shutil
import subprocess
from pathlib import Path

import pytest

from app.media import FFmpegMediaExtractor


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg missing")
def test_extract_synthetic_video(tmp_path: Path) -> None:
    video = tmp_path / "sample.mp4"
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "color=c=blue:s=320x240:r=10",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000",
        "-t", "4", "-c:v", "mpeg4", "-c:a", "aac", str(video),
    ], check=True, capture_output=True)
    extractor = FFmpegMediaExtractor(interval_seconds=10, frame_width=320)
    info = extractor.inspect(video)
    frames, audio = extractor.extract(video, tmp_path)
    assert 3.9 < info.duration_seconds < 4.1
    assert info.has_audio
    assert len(frames) == 1
    assert audio.is_file() and audio.stat().st_size > 0
