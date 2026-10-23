"""FastAPI HTTP layer. Audio/video is uploaded, not linked from remote URLs."""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from uuid import uuid4

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile

from app.factory import create_pipeline
from app.media import MediaError
from app.models import PredictionResponse
from app.pipeline import ClassQualityPipeline
from app.settings import Settings

logger = logging.getLogger(__name__)
app = FastAPI(title="Class Quality MVP", version="0.1.0")
ALLOWED_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm"}


@app.get("/health")
def health() -> dict[str, str]:
    """Return process health without exposing credentials or model state."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(
    file: UploadFile = File(...),
    pipeline: ClassQualityPipeline = Depends(create_pipeline),
) -> PredictionResponse:
    """Accept upload, run local preprocessing + GPT-4o, return weighted result."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=400, detail="Supported formats: MP4, MOV, MKV, WEBM")
    settings = Settings.from_environment()
    limit = settings.max_upload_mb * 1024 * 1024
    run_id = uuid4().hex
    try:
        with tempfile.TemporaryDirectory(prefix="class-quality-") as tmp:
            workspace = Path(tmp)
            video = workspace / f"upload{suffix}"
            total_bytes = 0
            with video.open("wb") as handle:
                while chunk := file.file.read(1024 * 1024):
                    total_bytes += len(chunk)
                    if total_bytes > limit:
                        raise HTTPException(status_code=413, detail="Video exceeds MAX_UPLOAD_MB")
                    handle.write(chunk)
            if total_bytes == 0:
                raise HTTPException(status_code=400, detail="Uploaded video is empty")
            return pipeline.run(video, workspace, run_id)
    except (MediaError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Prediction failed for run %s", run_id)
        raise HTTPException(
            status_code=500, detail="Processing failed. Review the FastAPI server log.",
        ) from exc
    finally:
        file.file.close()
