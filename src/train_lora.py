"""LoRA fine-tuning pipeline for BusquePet breed classification."""

from __future__ import annotations

import argparse
import os
import logging
from pathlib import Path
from typing import Dict, Optional

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import albumentations as A
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from transformers import (
    EarlyStoppingCallback,
    TrainingArguments,
    Trainer,
    ViTForImageClassification,
)
from peft import LoraConfig, TaskType, get_peft_model

from augmentations import RealisticAugmentations
from system_utils import auto_optimize_for_macos, configure_logging, dump_execution_plan

LOGGER = configure_logging(__name__)


class PetDataset(Dataset):
    """Dataset for supervised breed classification."""

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        transform: Optional[A.BasicTransform] = None,
        label_encoder: Optional[LabelEncoder] = None,
    ) -> None:
        self.df = metadata_df.reset_index(drop=True)
        self.transform = transform
        self.label_encoder = label_encoder or LabelEncoder()
        if label_encoder is None:
            self.label_encoder.fit(self.df["breed"])
        else:
            missing = set(self.df["breed"]) - set(self.label_encoder.classes_)
            if missing:
                raise ValueError(f"Validation set has unseen breeds: {missing}")
        self.label_map: Dict[str, int] = {
            breed: idx for idx, breed in enumerate(self.label_encoder.classes_)
        }

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        image = Image.open(row["processed_path"]).convert("RGB")
        array = np.array(image)
        if self.transform:
            array = self.transform(image=array)["image"]
        else:
            array = torch.from_numpy(array).permute(2, 0, 1).float() / 255.0
        label = self.label_map[row["breed"]]
        return {"pixel_values": array, "labels": torch.tensor(label, dtype=torch.long)}


def compute_metrics(eval_pred) -> Dict[str, float]:
    """Compute standard metrics for classification."""
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted"),
        "precision": precision_score(labels, predictions, average="weighted"),
        "recall": recall_score(labels, predictions, average="weighted"),
    }


class LoRATrainer:
    """Wrap the Hugging Face Trainer with BusquePet defaults and CPU optimisations."""

    def __init__(
        self,
        model_name: str = "ISxOdin/vit-base-oxford-iiit-pets",
        output_dir: str = "models/lora",
        num_epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 5e-4,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        num_workers: int = 2,
    ) -> None:
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.exec_config = auto_optimize_for_macos(batch_size=batch_size, num_workers=num_workers)
        self.batch_size = self.exec_config.batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.num_workers = self.exec_config.num_workers
        dump_execution_plan(self.exec_config, self.output_dir / "execution_plan.json")
        LOGGER.info(
            "LoRA training on %s | batch=%s | workers=%s", self.exec_config.device, self.batch_size, self.num_workers
        )
        self.lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["query", "value"],
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.IMAGE_CLASSIFICATION,
        )
        self.label_encoder: Optional[LabelEncoder] = None

    def prepare_datasets(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> tuple[PetDataset, PetDataset]:
        augmentations = RealisticAugmentations(image_size=self.exec_config.image_size, p=0.5)
        train_dataset = PetDataset(train_df, transform=augmentations.get_training_transform())
        val_dataset = PetDataset(val_df, transform=augmentations.get_validation_transform(), label_encoder=train_dataset.label_encoder)
        self.label_encoder = train_dataset.label_encoder
        self.num_labels = len(self.label_encoder.classes_)
        LOGGER.info("Train samples: %s | Val samples: %s | Classes: %s", len(train_dataset), len(val_dataset), self.num_labels)
        return train_dataset, val_dataset

    def setup_model(self) -> ViTForImageClassification:
        LOGGER.info("Loading pretrained model %s", self.model_name)
        model = ViTForImageClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            ignore_mismatched_sizes=True,
        )
        model = get_peft_model(model, self.lora_config)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        LOGGER.info("Trainable params: %.2f%% (%s/%s)", (trainable / total) * 100, trainable, total)
        return model

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame):
        train_dataset, val_dataset = self.prepare_datasets(train_df, val_df)
        model = self.setup_model()
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            warmup_steps=200,
            weight_decay=0.01,
            logging_dir=str(self.output_dir / "logs"),
            logging_steps=25,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            fp16=False,
            bf16=self.exec_config.use_mixed_precision and self.exec_config.precision == torch.bfloat16,
            dataloader_num_workers=self.num_workers,
            remove_unused_columns=False,
            report_to=["tensorboard"],
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )
        LOGGER.info("Starting LoRA fine-tuning")
        trainer.train()
        trainer.save_model(str(self.output_dir / "final"))
        self.save_label_encoder()
        return model, trainer

    def save_label_encoder(self) -> None:
        if not self.label_encoder:
            return
        import pickle

        encoder_path = self.output_dir / "label_encoder.pkl"
        with encoder_path.open("wb") as fp:
            pickle.dump(self.label_encoder, fp)
        LOGGER.info("Label encoder stored at %s", encoder_path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune the LoRA classifier.")
    parser.add_argument("--train-metadata", default="data/train_metadata.csv", help="Training metadata CSV path.")
    parser.add_argument("--val-metadata", default="data/val_metadata.csv", help="Validation metadata CSV path.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--learning-rate", type=float, default=5e-4, help="Learning rate.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    train_csv = Path(args.train_metadata)
    val_csv = Path(args.val_metadata)
    if not train_csv.exists() or not val_csv.exists():
        raise FileNotFoundError("Train/validation metadata CSV files are required.")
    trainer = LoRATrainer(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    trainer.train(pd.read_csv(train_csv), pd.read_csv(val_csv))


if __name__ == "__main__":
    main()
