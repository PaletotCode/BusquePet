"""Download the Oxford-IIIT Pets dataset from Hugging Face and export to data/raw."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

from datasets import ClassLabel, Dataset, get_dataset_split_names, load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch Oxford Pets images and save under data/raw/<breed>/")
    parser.add_argument(
        "--dataset",
        type=str,
        default="enterprise-explorers/oxford-pets",
        help="Dataset identifier on Hugging Face Hub.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Destination root for the class folders.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Dataset splits to download (default: todos disponíveis).",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face access token (optional if HF_TOKEN env is set or CLI login already done).",
    )
    return parser.parse_args()


def detect_columns(dataset: Dataset) -> tuple[str, str]:
    """Return (image_column, label_column)."""
    columns = dataset.column_names
    image_candidates = ["image", "img", "photo"]
    label_candidates = ["label", "labels", "breed", "class"]

    image_col = next((c for c in image_candidates if c in columns), None)
    if image_col is None:
        raise ValueError(f"Não encontrei coluna de imagem nas colunas: {columns}")

    label_col = next((c for c in label_candidates if c in columns), None)
    if label_col is None:
        raise ValueError(f"Não encontrei coluna de label nas colunas: {columns}")

    return image_col, label_col


def normalise_label(label_value, label_feature: Optional[ClassLabel]) -> str:
    if isinstance(label_feature, ClassLabel):
        return label_feature.int2str(int(label_value))
    if isinstance(label_value, (list, tuple)) and label_value:
        return str(label_value[0])
    return str(label_value)


def export_split(dataset: Dataset, image_col: str, label_col: str, output_dir: Path, split_name: str) -> None:
    label_feature = dataset.features.get(label_col)
    for idx, sample in enumerate(dataset):
        image = sample[image_col]
        if image is None:
            continue
        label_name = normalise_label(sample[label_col], label_feature)
        safe_label = label_name.replace(" ", "_").replace("/", "-").lower()
        dest_dir = output_dir / safe_label
        dest_dir.mkdir(parents=True, exist_ok=True)
        filename = dest_dir / f"{split_name}_{idx:05d}.jpg"
        image.convert("RGB").save(filename, format="JPEG", quality=95)


def main() -> None:
    args = parse_args()
    token = args.token or os.getenv("HF_TOKEN")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    available_splits = get_dataset_split_names(args.dataset, use_auth_token=token)
    requested_splits = args.splits or available_splits
    splits = []
    for split in requested_splits:
        if split in available_splits:
            splits.append(split)
        else:
            print(f'[aviso] split "{split}" não existe neste dataset. Disponíveis: {available_splits}')
    if not splits:
        raise ValueError("Nenhum split válido informado.")

    for split in splits:
        dataset = load_dataset(
            args.dataset,
            split=split,
            use_auth_token=token,
        )
        image_col, label_col = detect_columns(dataset)
        print(f"[{split}] imagens={len(dataset)} coluna_imagem={image_col} coluna_label={label_col}")
        export_split(dataset, image_col, label_col, args.output_dir, split_name=split)


if __name__ == "__main__":
    main()
