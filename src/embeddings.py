"""Embedding extraction utilities for BusquePet."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from augmentations import RealisticAugmentations
from system_utils import auto_optimize_for_macos, configure_logging, mixed_precision_context
from train_contrastive import EmbeddingModel

LOGGER = configure_logging(__name__)


class EmbeddingExtractor:
    """Extract L2-normalised embeddings for a metadata dataframe."""

    def __init__(
        self,
        model_path: str,
        base_model_name: str = "ISxOdin/vit-base-oxford-iiit-pets",
        embedding_dim: int = 768,
        batch_size: int = 64,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")
        self.exec_config = auto_optimize_for_macos(batch_size=batch_size, num_workers=2)
        self.device = self.exec_config.device
        self.base_model_name = base_model_name
        self.embedding_dim = embedding_dim
        self.batch_size = self.exec_config.batch_size
        LOGGER.info("Embedding extraction on %s | batch=%s", self.device, self.batch_size)
        self.model = self._load_model()
        self.transform = RealisticAugmentations(image_size=self.exec_config.image_size).get_validation_transform()

    def _load_model(self) -> EmbeddingModel:
        LOGGER.info("Loading embedding model from %s", self.model_path)
        model = EmbeddingModel(
            base_model_name=self.base_model_name,
            embedding_dim=self.embedding_dim,
        )
        checkpoint = torch.load(self.model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict)
        model = model.to(self.device)
        model.eval()
        return model

    def _prepare_tensor(self, image_path: str) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")
        array = np.array(image)
        tensor = self.transform(image=array)["image"]
        return tensor

    def extract_single_embedding(self, image_path: str) -> np.ndarray:
        tensor = self._prepare_tensor(image_path).unsqueeze(0).to(self.device)
        with torch.no_grad():
            with mixed_precision_context(self.device, self.exec_config.precision, self.exec_config.use_mixed_precision):
                embedding = self.model(tensor).float()
        return embedding.cpu().numpy()[0]

    def extract_batch_embeddings(self, image_paths: List[str]) -> np.ndarray:
        tensors = []
        for path in image_paths:
            try:
                tensors.append(self._prepare_tensor(path))
            except Exception as err:
                LOGGER.error("Failed to process %s: %s", path, err)
                tensors.append(torch.zeros(3, self.exec_config.image_size, self.exec_config.image_size))
        batch_tensor = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            with mixed_precision_context(self.device, self.exec_config.precision, self.exec_config.use_mixed_precision):
                embeddings = self.model(batch_tensor).float()
        return embeddings.cpu().numpy()

    def extract_dataset_embeddings(
        self, metadata_df: pd.DataFrame, output_path: str, resume: bool = False
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        existing_embeddings = None
        start_idx = 0
        if resume:
            emb_path = output_dir / "embeddings.npy"
            id_map_path = output_dir / "id_to_index.json"
            if emb_path.exists() and id_map_path.exists():
                existing_embeddings = np.load(emb_path)
                start_idx = existing_embeddings.shape[0]
                LOGGER.info("Resuming embeddings: found %s vectors, continuing from index %s", start_idx, start_idx)
            else:
                LOGGER.info("Resume flag set but no prior embeddings found; starting fresh.")

        embeddings_list: List[np.ndarray] = []
        image_paths = metadata_df["processed_path"].tolist()
        image_ids = metadata_df["image_id"].tolist()
        if start_idx >= len(image_paths):
            LOGGER.info("All embeddings already present; skipping extraction.")
        for start in tqdm(range(start_idx, len(image_paths), self.batch_size), desc="Embeddings"):
            batch_paths = image_paths[start : start + self.batch_size]
            embeddings = self.extract_batch_embeddings(batch_paths)
            embeddings_list.append(embeddings)
        new_embeddings = np.vstack(embeddings_list) if embeddings_list else np.empty((0, self.embedding_dim))
        if existing_embeddings is not None:
            all_embeddings = np.vstack([existing_embeddings, new_embeddings])
        else:
            all_embeddings = new_embeddings
        np.save(output_dir / "embeddings.npy", all_embeddings)
        id_to_index = {img_id: idx for idx, img_id in enumerate(image_ids)}
        with (output_dir / "id_to_index.json").open("w", encoding="utf-8") as fp:
            json.dump(id_to_index, fp, indent=2)
        metadata_df.assign(embedding_index=list(range(len(metadata_df)))).to_csv(
            output_dir / "embedding_metadata.csv", index=False
        )
        return all_embeddings, id_to_index

    def compute_embedding_statistics(self, embeddings: np.ndarray) -> Dict[str, float]:
        norms = np.linalg.norm(embeddings, axis=1)
        return {
            "shape": list(embeddings.shape),
            "mean_norm": float(norms.mean()),
            "std_norm": float(norms.std()),
            "mean_value": float(embeddings.mean()),
            "std_value": float(embeddings.std()),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract embeddings for the BusquePet dataset.")
    parser.add_argument("--metadata", default="data/metadata.csv", help="Path to the metadata CSV.")
    parser.add_argument("--output", default="data/embeddings", help="Directory to store embeddings and metadata.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing embeddings if they are already partially computed.",
    )
    args = parser.parse_args()

    metadata_csv = Path(args.metadata)
    if not metadata_csv.exists():
        raise FileNotFoundError(metadata_csv)
    extractor = EmbeddingExtractor(model_path="models/contrastive/best_model.pt")
    embeddings, _ = extractor.extract_dataset_embeddings(
        pd.read_csv(metadata_csv), output_path=args.output, resume=args.resume
    )
    stats = extractor.compute_embedding_statistics(embeddings)
    with Path(args.output).joinpath("embedding_stats.json").open("w", encoding="utf-8") as fp:
        json.dump(stats, fp, indent=2)
    LOGGER.info("Embedding extraction complete")


if __name__ == "__main__":
    main()
