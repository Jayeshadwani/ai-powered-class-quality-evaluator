"""FFprobe/FFmpeg adapter. No speech or classification logic belongs here."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from app.models import Frame, VideoInfo


class MediaError(RuntimeError):
    """Raised when media validation or extraction fails."""


class FFmpegMediaExtractor:
    """Inspect video, extract local WAV and fixed-interval JPEG frames."""

    def __init__(self, interval_seconds: int, frame_width: int) -> None:
        self.interval_seconds = interval_seconds
        self.frame_width = frame_width

    def inspect(self, video: Path) -> VideoInfo:
        """Return duration and audio availability from FFprobe metadata."""
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(video)],
                capture_output=True, text=True, check=True, timeout=120,
            )
            metadata = json.loads(result.stdout)
            streams = metadata.get("streams", [])
            if not any(s.get("codec_type") == "video" for s in streams):
                raise MediaError("File does not contain a video stream")
            duration = float(metadata["format"]["duration"])
            if duration <= 0:
                raise MediaError("Video duration must be positive")
            return VideoInfo(
                duration_seconds=duration,
                has_audio=any(s.get("codec_type") == "audio" for s in streams),
            )
        except (subprocess.CalledProcessError, OSError, KeyError, ValueError, subprocess.TimeoutExpired) as exc:
            raise MediaError("Unable to inspect the uploaded video; check FFprobe and the file") from exc

    def extract(self, video: Path, output_dir: Path) -> tuple[list[Frame], Path]:
        """Save JPEGs at nominal 10s offsets and uncompressed 16k mono WAV."""
        frames_dir = output_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        audio_path = output_dir / "audio.wav"
        self._execute([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
            "-vf", f"fps=1/{self.interval_seconds}:round=up,scale={self.frame_width}:-2,format=yuvj420p",
            "-q:v", "3", str(frames_dir / "frame_%05d.jpg"),
        ])
        self._execute([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
            "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(audio_path),
        ])
        paths = sorted(frames_dir.glob("frame_*.jpg"))
        if not paths or not audio_path.is_file():
            raise MediaError("FFmpeg did not produce frames and audio")
        # Nominal timestamps for constant-rate sampling. The actual decoded frame
        # may deviate by up to a source-frame interval, especially for variable FPS.
        frames = [
            Frame(frame_id=f"frame_{i:05d}", timestamp_seconds=float(i * self.interval_seconds), path=path)
            for i, path in enumerate(paths)
        ]
        return frames, audio_path

    @staticmethod
    def _execute(command: list[str]) -> None:
        try:
            subprocess.run(command, check=True, capture_output=True, text=True, timeout=3600)
        except (subprocess.CalledProcessError, OSError, subprocess.TimeoutExpired) as exc:
            raise MediaError("FFmpeg failed while processing the recording") from exc
