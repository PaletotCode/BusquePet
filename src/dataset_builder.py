"""Dataset builder responsible for ETL, perceptual hashing, and metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import imagehash
import pandas as pd
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from system_utils import configure_logging

logger = configure_logging(__name__)


class DatasetBuilder:
    """Construct processed datasets enriched with perceptual hashes."""

    def __init__(self, raw_data_path: str, output_path: str, image_size: int = 224) -> None:
        self.raw_data_path = Path(raw_data_path)
        self.output_path = Path(output_path)
        self.image_size = image_size

        if not self.raw_data_path.exists():
            raise FileNotFoundError(f"Raw data path does not exist: {self.raw_data_path}")

        self.output_path.mkdir(parents=True, exist_ok=True)
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def compute_phash(self, image_path: str, hash_size: int = 16) -> Optional[str]:
        """Return the perceptual hash of *image_path*, handling corrupt files."""
        try:
            img = Image.open(image_path).convert("RGB")
            phash = imagehash.phash(img, hash_size=hash_size)
            return str(phash)
        except Exception as err:  # pragma: no cover - pillow/cv2 failures
            logger.error("Failed to compute pHash for %s: %s", image_path, err)
            return None

    def extract_metadata_from_path(self, image_path: Path) -> Dict:
        """Infer dataset metadata based on folder structure."""
        parts = image_path.parts
        metadata = {
            "image_path": str(image_path),
            "filename": image_path.name,
            "breed": parts[-2] if len(parts) > 1 else "unknown",
            "image_id": image_path.stem,
        }
        return metadata

    def validate_image(self, image_path: Path) -> bool:
        """Validate image integrity before processing."""
        try:
            img = Image.open(image_path)
            img.verify()
            if img.size[0] < 50 or img.size[1] < 50:
                return False
            return True
        except Exception as err:  # pragma: no cover - pillow/cv2 failures
            logger.warning("Invalid image %s: %s", image_path, err)
            return False

    def normalize_and_save_image(self, image_path: Path, output_dir: Path) -> Optional[str]:
        """Resize and persist a normalized RGB copy of the original image."""
        try:
            img = Image.open(image_path).convert("RGB")
            img_resized = img.resize((self.image_size, self.image_size), Image.LANCZOS)

            output_subdir = output_dir / image_path.parent.name
            output_subdir.mkdir(parents=True, exist_ok=True)

            output_path = output_subdir / image_path.name
            img_resized.save(output_path, quality=95)

            return str(output_path)
        except Exception as err:
            logger.error("Failed to normalize %s: %s", image_path, err)
            return None

    def build_dataset(self, extensions: List[str] = (".jpg", ".jpeg", ".png")) -> pd.DataFrame:
        """Build the processed dataset and save metadata + statistics."""
        logger.info("Starting dataset build from %s", self.raw_data_path)

        image_files = []
        for ext in extensions:
            image_files.extend(self.raw_data_path.rglob(f"*{ext}"))
            image_files.extend(self.raw_data_path.rglob(f"*{ext.upper()}"))

        logger.info("Found %d candidate images", len(image_files))

        records = []
        processed_dir = self.output_path / "processed"
        processed_dir.mkdir(exist_ok=True)

        for img_path in tqdm(image_files, desc="Processing images"):
            if not self.validate_image(img_path):
                continue

            metadata = self.extract_metadata_from_path(img_path)
            phash = self.compute_phash(img_path)

            if phash is None:
                continue

            normalized_path = self.normalize_and_save_image(img_path, processed_dir)

            if normalized_path is None:
                continue

            record = {
                "image_id": metadata["image_id"],
                "original_path": metadata["image_path"],
                "processed_path": normalized_path,
                "breed": metadata["breed"],
                "phash": phash,
                "filename": metadata["filename"],
            }

            records.append(record)

        df = pd.DataFrame(records)

        csv_path = self.output_path / "metadata.csv"
        df.to_csv(csv_path, index=False)
        logger.info("Dataset stored at %s with %d clean images", csv_path, len(df))

        stats = self.compute_dataset_statistics(df)
        self.save_statistics(stats)

        return df

    def compute_dataset_statistics(self, df: pd.DataFrame) -> Dict:
        """Compute useful summary statistics about the processed dataset."""
        stats = {
            "total_images": len(df),
            "total_breeds": df["breed"].nunique(),
            "breed_distribution": df["breed"].value_counts().to_dict(),
            "unique_phashes": df["phash"].nunique(),
            "duplicate_phashes": len(df) - df["phash"].nunique(),
        }
        return stats

    def save_statistics(self, stats: Dict) -> None:
        """Persist dataset statistics for future audits."""
        stats_path = self.output_path / "dataset_stats.json"
        with open(stats_path, "w", encoding="utf-8") as fp:
            json.dump(stats, fp, indent=2)
        logger.info("Dataset statistics saved to %s", stats_path)

    def create_train_val_split(
        self, df: pd.DataFrame, val_ratio: float = 0.2, seed: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create a stratified train/validation split."""
        from sklearn.model_selection import train_test_split

        train_df, val_df = train_test_split(
            df,
            test_size=val_ratio,
            stratify=df["breed"],
            random_state=seed,
        )

        train_path = self.output_path / "train_metadata.csv"
        val_path = self.output_path / "val_metadata.csv"

        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)

        logger.info("Train split: %d | Validation split: %d", len(train_df), len(val_df))

        return train_df, val_df


def main() -> None:
    """Entry point for manual dataset generation."""
    builder = DatasetBuilder(
        raw_data_path="data/raw",
        output_path="data",
        image_size=224,
    )

    df = builder.build_dataset()

    builder.create_train_val_split(df, val_ratio=0.2)

    logger.info("ETL finished successfully! Processed images: %d", len(df))


if __name__ == "__main__":
    main()
