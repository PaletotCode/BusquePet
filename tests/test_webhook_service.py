from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

import webhook_service


class DummyPipeline:
    def __init__(self, results: List[Dict[str, Any]]):
        self.results = results
        self.calls: List[Dict[str, Any]] = []

    def search(self, query_image_path: str, k: int, use_hybrid: bool, query_phash: str | None = None):
        self.calls.append(
            {
                "image_path": query_image_path,
                "k": k,
                "use_hybrid": use_hybrid,
                "phash": query_phash,
            }
        )
        return self.results


@pytest.fixture()
def client_bundle(tmp_path, monkeypatch):
    """Prepare a FastAPI TestClient wired to a dummy pipeline."""
    # Create placeholder assets required by Settings
    model_path = tmp_path / "models" / "contrastive" / "best_model.pt"
    metadata_path = tmp_path / "data" / "embeddings" / "embedding_metadata.csv"
    faiss_index_path = tmp_path / "models" / "faiss_index"
    for required in (model_path, metadata_path):
        required.parent.mkdir(parents=True, exist_ok=True)
        required.write_text("stub")
    faiss_index_path.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("BUSQUEPET_MODEL_PATH", str(model_path))
    monkeypatch.setenv("BUSQUEPET_METADATA_PATH", str(metadata_path))
    monkeypatch.setenv("BUSQUEPET_FAISS_INDEX", str(faiss_index_path))
    monkeypatch.setenv("BUSQUEPET_WEBHOOK_PORT", "8100")

    image_path = tmp_path / "shared" / "normalized" / "sample.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"data")

    webhook_service.get_settings.cache_clear()
    webhook_service.get_pipeline.cache_clear()

    dummy_pipeline = DummyPipeline(
        results=[
            {
                "rank": 1,
                "image_id": "pet-1",
                "breed": "husky",
                "score": 0.91,
                "distance": 0.08,
                "image_path": "/tmp/pet.jpg",
                "phash_score": 0.5,
                "embedding_score": 0.8,
            }
        ]
    )

    def fake_get_pipeline():
        return dummy_pipeline
    fake_get_pipeline.cache_clear = lambda: None

    monkeypatch.setattr(webhook_service, "get_pipeline", fake_get_pipeline)

    app = webhook_service.create_app()
    client = TestClient(app)
    try:
        yield client, dummy_pipeline, image_path
    finally:
        client.close()
        webhook_service.get_settings.cache_clear()
        if hasattr(webhook_service.get_pipeline, "cache_clear"):
            webhook_service.get_pipeline.cache_clear()



def test_webhook_returns_matches_and_respects_phash(client_bundle):
    client, pipeline, image_path = client_bundle

    payload = {
        "job_id": "abc123",
        "image_path": str(image_path),
        "phash": "feedface",
        "k": 5,
        "use_hybrid": True,
        "metadata": {"original_filename": "example.jpg"},
    }

    response = client.post("/v1/match", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["job_id"] == "abc123"
    assert data["matches"][0]["image_id"] == "pet-1"
    assert data["duration_ms"] >= 0

    assert len(pipeline.calls) == 1
    assert pipeline.calls[0]["phash"] == "feedface"
    assert pipeline.calls[0]["k"] == 5


def test_webhook_validates_missing_image(client_bundle):
    client, pipeline, image_path = client_bundle
    missing_path = Path(image_path).with_name("missing.jpg")

    payload = {
        "job_id": "job-404",
        "image_path": str(missing_path),
        "phash": None,
    }

    response = client.post("/v1/match", json=payload)
    assert response.status_code == 400
    assert "does not exist" in response.json()["detail"]
    assert pipeline.calls == []
