"""Inference pipeline combining FAISS search with perceptual hashes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import imagehash
import pandas as pd
from PIL import Image

from embeddings import EmbeddingExtractor
from index_faiss import FAISSIndexer, HybridMatcher
from system_utils import configure_logging

LOGGER = configure_logging(__name__)


class PetMatchingPipeline:
    """Full hybrid retrieval pipeline for BusquePet."""

    def __init__(
        self,
        model_path: str,
        faiss_index_path: str,
        metadata_path: str,
        base_model_name: str = "ISxOdin/vit-base-oxford-iiit-pets",
        embedding_dim: int = 768,
        embedding_weight: float = 0.7,
        phash_weight: float = 0.3,
    ) -> None:
        self.model_path = Path(model_path)
        self.faiss_index_path = Path(faiss_index_path)
        self.metadata_path = Path(metadata_path)
        if not self.model_path.exists():
            raise FileNotFoundError(self.model_path)
        if not self.metadata_path.exists():
            raise FileNotFoundError(self.metadata_path)
        self.metadata_df = pd.read_csv(self.metadata_path)
        self.embedding_extractor = EmbeddingExtractor(
            model_path=str(self.model_path),
            base_model_name=base_model_name,
            embedding_dim=embedding_dim,
            batch_size=1,
        )
        self.faiss_indexer = FAISSIndexer(embedding_dim=embedding_dim)
        self.faiss_indexer.load_index(str(self.faiss_index_path))
        self.matcher = HybridMatcher(
            faiss_indexer=self.faiss_indexer,
            metadata_df=self.metadata_df,
            embedding_weight=embedding_weight,
            phash_weight=phash_weight,
        )

    def compute_phash(self, image_path: str, hash_size: int = 16) -> str:
        image = Image.open(image_path).convert("RGB")
        return str(imagehash.phash(image, hash_size=hash_size))

    def search(
        self,
        query_image_path: str,
        k: int = 10,
        use_hybrid: bool = True,
        query_phash: str | None = None,
    ) -> List[Dict]:
        LOGGER.info("Running search for %s", query_image_path)
        embedding = self.embedding_extractor.extract_single_embedding(query_image_path)
        if use_hybrid:
            phash = query_phash or self.compute_phash(query_image_path)
            scored = self.matcher.hybrid_search(embedding, phash, k=k)
            for rank, result in enumerate(scored, 1):
                result.update({"rank": rank, "score": result.pop("hybrid_score")})
            return scored
        distances, indices = self.faiss_indexer.search(embedding, k)
        results = []
        for rank, (idx, dist) in enumerate(zip(indices, distances), 1):
            row = self.metadata_df.iloc[idx]
            results.append(
                {
                    "rank": rank,
                    "index": int(idx),
                    "score": float(1.0 - dist),
                    "distance": float(dist),
                    "image_id": row["image_id"],
                    "breed": row["breed"],
                    "image_path": row["processed_path"],
                }
            )
        return results

    def batch_search(self, query_image_paths: List[str], k: int = 10) -> List[List[Dict]]:
        return [self.search(path, k=k) for path in query_image_paths]

    def visualize_results(self, query_image_path: str, results: List[Dict], output_path: str | None = None) -> None:
        import matplotlib.pyplot as plt

        if not results:
            LOGGER.warning("No results to visualise")
            return
        n_cols = min(5, len(results))
        n_rows = (len(results) + n_cols - 1) // n_cols + 1
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
        axes = axes.ravel()
        axes[0].imshow(Image.open(query_image_path))
        axes[0].set_title("Query", fontweight="bold")
        axes[0].axis("off")
        for idx in range(1, n_cols):
            axes[idx].axis("off")
        for idx, result in enumerate(results):
            axis = axes[n_cols + idx]
            axis.imshow(Image.open(result["image_path"]))
            axis.set_title(f"#{result['rank']} - {result['breed']}\nScore: {result['score']:.3f}", fontsize=9)
            axis.axis("off")
        for axis in axes[n_cols + len(results) :]:
            axis.axis("off")
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            LOGGER.info("Visualisation saved to %s", output_path)
        else:
            plt.show()
        plt.close()


def main() -> None:
    pipeline = PetMatchingPipeline(
        model_path="models/contrastive/best_model.pt",
        faiss_index_path="models/faiss_index",
        metadata_path="data/embeddings/embedding_metadata.csv",
    )
    query = "data/processed/example_query.jpg"
    results = pipeline.search(query, k=10, use_hybrid=True)
    output_dir = Path("outputs/inference")
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline.visualize_results(query, results, output_path=str(output_dir / "search_results.png"))
    with (output_dir / "search_results.json").open("w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)
    LOGGER.info("Inference demo completed")


if __name__ == "__main__":
    main()
