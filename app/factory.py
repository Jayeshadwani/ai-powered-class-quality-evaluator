"""Composition root. This is the only place binding concrete implementations."""
from __future__ import annotations

from functools import lru_cache

from app.artifacts import JSONArtifactStore
from app.batching import TemporalBatcher
from app.gpt_evaluator import GPT4oEvaluator
from app.media import FFmpegMediaExtractor
from app.pipeline import ClassQualityPipeline
from app.rubric import Rubric
from app.settings import ROOT, Settings
from app.transcription import FasterWhisperTranscriber


@lru_cache(maxsize=1)
def create_pipeline() -> ClassQualityPipeline:
    """Construct services once per API worker; defer ASR loading until first upload."""
    settings = Settings.from_environment()
    rubric = Rubric(ROOT / "config" / "rubric.json")
    return ClassQualityPipeline(
        media=FFmpegMediaExtractor(settings.frame_interval_seconds, settings.frame_width),
        transcription=FasterWhisperTranscriber(
            settings.whisper_model, settings.whisper_device,
            settings.whisper_compute_type, settings.whisper_language,
            settings.whisper_vad_min_silence_ms,
        ),
        batching=TemporalBatcher(settings.batch_duration_seconds),
        evaluator=GPT4oEvaluator(
            api_key=settings.api_key,
            model=settings.model, image_detail=settings.image_detail,
            timeout_seconds=settings.openai_timeout_seconds,
            criteria=rubric.criteria,
        ),
        rubric=rubric,
        artifacts=JSONArtifactStore(settings.runs_dir, settings.save_results),
        max_duration_seconds=settings.max_duration_minutes * 60,
        model_name=settings.model,
    )
