"""Local JSON output storage. Raw media is held only in a temporary workspace."""
from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel


class JSONArtifactStore:
    """Persist auditable, non-media intermediate data in local runs/<id>/ folder."""

    def __init__(self, runs_dir: Path, enabled: bool) -> None:
        self.runs_dir = runs_dir
        self.enabled = enabled

    def save(self, run_id: str, filename: str, data: object) -> None:
        """Save a JSON file; never accept caller-controlled arbitrary filenames."""
        if not self.enabled:
            return
        if not run_id.isalnum() or filename not in {"transcript.json", "batch_evidence.json", "assessment.json"}:
            raise ValueError("Invalid artifact name")
        directory = self.runs_dir / run_id
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        if isinstance(data, BaseModel):
            payload = data.model_dump(mode="json")
        elif isinstance(data, list):
            payload = [item.model_dump(mode="json") if isinstance(item, BaseModel) else item for item in data]
        else:
            payload = data
        (directory / filename).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
        )
