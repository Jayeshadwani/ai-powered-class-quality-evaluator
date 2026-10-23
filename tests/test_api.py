"""Verify FastAPI contract without invoking Whisper or GPT-4o."""
from pathlib import Path

from fastapi.testclient import TestClient

from app.api import app
from app.factory import create_pipeline
from conftest import make_evaluation
from app.rubric import Rubric
from app.settings import ROOT


class StubPipeline:
    def run(self, video_path: Path, workspace: Path, run_id: str):
        assert video_path.exists()
        assert video_path.read_bytes() == b"stub content"
        rubric = Rubric(ROOT / "config" / "rubric.json")
        return rubric.apply(make_evaluation(2), run_id=run_id, model="gpt-4o", duration_seconds=120, batch_count=1)


def test_http_upload_contract() -> None:
    app.dependency_overrides[create_pipeline] = lambda: StubPipeline()
    try:
        with TestClient(app) as client:
            response = client.post("/predict", files={"file": ("class.mp4", b"stub content", "video/mp4")})
        assert response.status_code == 200, response.text
        data = response.json()
        assert 0 <= data["score"] <= 100
        assert data["label"] in {"Poor", "Average", "Good", "Excellent"}
        assert len(data["ratings"]) == 6
    finally:
        app.dependency_overrides.clear()
