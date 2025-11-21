from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pytest

from inference import PetMatchingPipeline


def _stub_pipeline():
    pipeline = object.__new__(PetMatchingPipeline)
    pipeline.embedding_extractor = types.SimpleNamespace(
        extract_single_embedding=lambda path: [0.42]
    )
    pipeline.matcher = None
    pipeline.compute_phash = lambda path, hash_size=16: "auto"
    return pipeline


def test_pipeline_search_uses_provided_phash():
    pipeline = _stub_pipeline()

    captured = {}

    def fake_hybrid_search(embedding, phash, k):
        captured["embedding"] = embedding
        captured["phash"] = phash
        return [
            {
                "hybrid_score": 0.77,
                "image_id": "pet-42",
                "breed": "mixed",
                "image_path": "/tmp/pet.jpg",
            }
        ]

    pipeline.matcher = types.SimpleNamespace(hybrid_search=fake_hybrid_search)

    results = pipeline.search("foo.jpg", k=1, use_hybrid=True, query_phash="deadbeef")

    assert captured["phash"] == "deadbeef"
    assert results[0]["score"] == pytest.approx(0.77)
    assert results[0]["rank"] == 1


def test_pipeline_search_computes_phash_when_missing():
    pipeline = _stub_pipeline()

    captured = {}

    def fake_hybrid_search(embedding, phash, k):
        captured["phash"] = phash
        return [
            {
                "hybrid_score": 0.55,
                "image_id": "pet-99",
                "breed": "pug",
                "image_path": "/tmp/another.jpg",
            }
        ]

    pipeline.matcher = types.SimpleNamespace(hybrid_search=fake_hybrid_search)
    pipeline.compute_phash = lambda path, hash_size=16: "computed"

    pipeline.search("foo.jpg", k=1, use_hybrid=True, query_phash=None)

    assert captured["phash"] == "computed"


def test_pipeline_falls_back_to_embedding_only():
    pipeline = _stub_pipeline()

    pipeline.faiss_indexer = types.SimpleNamespace(
        search=lambda embedding, k: (np.array([0.2]), np.array([0]))
    )
    pipeline.metadata_df = pd.DataFrame(
        [
            {"image_id": "pet-0", "breed": "akita", "processed_path": "/tmp/pet.png"},
        ]
    )

    results = pipeline.search("foo.jpg", k=1, use_hybrid=False)

    assert results[0]["image_id"] == "pet-0"
    assert results[0]["score"] == pytest.approx(0.8)
    assert "distance" in results[0]
