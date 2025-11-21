"""FAISS index helpers for BusquePet."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np
import pandas as pd

from system_utils import configure_logging

LOGGER = configure_logging(__name__)


class FAISSIndexer:
    """Thin wrapper around FAISS HNSW for cosine similarity search."""

    def __init__(
        self,
        embedding_dim: int = 768,
        index_type: str = "HNSW",
        M: int = 32,
        ef_construction: int = 200,
        ef_search: int = 128,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.index: faiss.Index | None = None

    def _normalise(self, embeddings: np.ndarray) -> np.ndarray:
        faiss.normalize_L2(embeddings)
        return embeddings

    def create_index(self, embeddings: np.ndarray) -> faiss.Index:
        embeddings = embeddings.astype(np.float32)
        embeddings = self._normalise(embeddings)
        if self.index_type == "HNSW":
            index = faiss.IndexHNSWFlat(self.embedding_dim, self.M)
            index.hnsw.efConstruction = self.ef_construction
            index.hnsw.efSearch = self.ef_search
        elif self.index_type == "IVF":
            nlist = min(int(np.sqrt(len(embeddings))), 100)
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
            index.train(embeddings)
        elif self.index_type == "Flat":
            index = faiss.IndexFlatL2(self.embedding_dim)
        else:
            raise ValueError(f"Unsupported index_type: {self.index_type}")
        index.add(embeddings)
        self.index = index
        LOGGER.info("FAISS index created with %s vectors", index.ntotal)
        return index

    def _ensure_index(self) -> faiss.Index:
        if self.index is None:
            raise RuntimeError("FAISS index is not initialised. Call create_index or load_index first.")
        return self.index

    def search(self, query_embedding: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        index = self._ensure_index()
        query = query_embedding.astype(np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        self._normalise(query)
        distances, indices = index.search(query, k)
        return distances[0], indices[0]

    def batch_search(self, query_embeddings: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        index = self._ensure_index()
        queries = query_embeddings.astype(np.float32)
        self._normalise(queries)
        return index.search(queries, k)

    def save_index(self, output_path: str) -> None:
        if self.index is None:
            raise RuntimeError("Cannot save a non-existent index.")
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(output_dir / "faiss_index.bin"))
        LOGGER.info("FAISS index saved to %s", output_dir)

    def load_index(self, index_path: str) -> faiss.Index:
        index_file = Path(index_path)
        if index_file.is_dir():
            index_file = index_file / "faiss_index.bin"
        if not index_file.exists():
            raise FileNotFoundError(f"FAISS index not found at {index_file}")
        self.index = faiss.read_index(str(index_file))
        LOGGER.info("FAISS index loaded from %s", index_file)
        return self.index

    def benchmark_search(self, query_embeddings: np.ndarray, k: int = 10) -> Dict[str, float]:
        distances, _ = self.batch_search(query_embeddings, k)
        return {
            "mean_distance": float(distances.mean()),
            "std_distance": float(distances.std()),
            "min_distance": float(distances.min()),
            "max_distance": float(distances.max()),
        }


class HybridMatcher:
    """Combine FAISS search with perceptual hash reranking."""

    def __init__(
        self,
        faiss_indexer: FAISSIndexer,
        metadata_df: pd.DataFrame,
        embedding_weight: float = 0.7,
        phash_weight: float = 0.3,
    ) -> None:
        self.faiss_indexer = faiss_indexer
        self.metadata_df = metadata_df.reset_index(drop=True)
        self.embedding_weight = embedding_weight
        self.phash_weight = phash_weight
        self.phash_dict = dict(zip(self.metadata_df.index, self.metadata_df["phash"]))

    @staticmethod
    def _hamming_distance(hash1: str, hash2: str) -> int:
        return bin(int(hash1, 16) ^ int(hash2, 16)).count("1")

    def hybrid_search(self, query_embedding: np.ndarray, query_phash: str, k: int = 10, candidate_multiplier: int = 3) -> List[Dict]:
        distances, indices = self.faiss_indexer.search(query_embedding, k * candidate_multiplier)
        results: List[Dict] = []
        for dist, idx in zip(distances, indices):
            emb_score = 1.0 - dist
            candidate_phash = self.phash_dict.get(int(idx))
            if candidate_phash and query_phash:
                hamming = self._hamming_distance(query_phash, candidate_phash)
                phash_score = 1.0 - min(hamming / 256.0, 1.0)
            else:
                phash_score = 0.0
            hybrid_score = self.embedding_weight * emb_score + self.phash_weight * phash_score
            row = self.metadata_df.iloc[idx]
            results.append(
                {
                    "index": int(idx),
                    "hybrid_score": float(hybrid_score),
                    "embedding_score": float(emb_score),
                    "phash_score": float(phash_score),
                    "embedding_distance": float(dist),
                    "image_id": row["image_id"],
                    "breed": row["breed"],
                    "image_path": row["processed_path"],
                }
            )
        results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return results[:k]


def main() -> None:
    embeddings_file = Path("data/embeddings/embeddings.npy")
    metadata_file = Path("data/embeddings/embedding_metadata.csv")
    if not embeddings_file.exists() or not metadata_file.exists():
        raise FileNotFoundError("Embeddings or metadata missing. Run the embedding extractor first.")
    embeddings = np.load(embeddings_file)
    metadata = pd.read_csv(metadata_file)
    indexer = FAISSIndexer()
    indexer.create_index(embeddings)
    indexer.save_index("models/faiss_index")
    benchmark = indexer.benchmark_search(embeddings[: min(100, len(embeddings))])
    with (Path("models/faiss_index") / "benchmark_results.json").open("w", encoding="utf-8") as fp:
        json.dump(benchmark, fp, indent=2)
    LOGGER.info("FAISS indexing completed")


if __name__ == "__main__":
    main()
