"""Contrastive training with triplet loss for BusquePet embeddings."""

from __future__ import annotations

import argparse
import os
import json
import logging
import random
from pathlib import Path
from typing import List, Tuple

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import albumentations as A
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TaskID,
)
from rich.text import Text
from transformers import ViTModel
from peft import PeftModel

from augmentations import RealisticAugmentations
from system_utils import (
    CheckpointManager,
    EarlyStopping,
    TrainingMetricsRecorder,
    auto_optimize_for_macos,
    configure_logging,
    dump_execution_plan,
    maybe_compile_model,
    mixed_precision_context,
)

LOGGER = configure_logging(__name__)


class RateColumn(ProgressColumn):
    """Safely render iterations per second even before the first batch."""

    def render(self, task) -> Text:
        speed = task.speed
        if not speed:
            return Text("-- it/s")
        return Text(f"{speed:5.2f} it/s")


class TripletDataset(Dataset):
    """Return (anchor, positive, negative) image tensors for triplet loss."""

    def __init__(self, metadata_df: pd.DataFrame, transform: A.BasicTransform | None = None) -> None:
        self.df = metadata_df.reset_index(drop=True)
        self.transform = transform
        self.breed_to_indices: dict[str, List[int]] = {}
        for idx, row in self.df.iterrows():
            self.breed_to_indices.setdefault(row["breed"], []).append(idx)
        self.valid_breeds = [breed for breed, indices in self.breed_to_indices.items() if len(indices) >= 2]
        if not self.valid_breeds:
            raise ValueError("Dataset requires at least one class with two samples.")

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, path: str) -> torch.Tensor:
        try:
            image = Image.open(path).convert("RGB")
        except FileNotFoundError as err:
            raise FileNotFoundError(f"Sample not found: {path}") from err
        array = np.array(image)
        if self.transform:
            transformed = self.transform(image=array)
            return transformed["image"]
        tensor = torch.from_numpy(array).permute(2, 0, 1).float() / 255.0
        return tensor

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        anchor_row = self.df.iloc[index]
        anchor_breed = anchor_row["breed"]
        anchor_image = self._load_image(anchor_row["processed_path"])

        positive_candidates = [idx for idx in self.breed_to_indices[anchor_breed] if idx != index]
        positive_idx = random.choice(positive_candidates) if positive_candidates else index
        positive_image = self._load_image(self.df.iloc[positive_idx]["processed_path"])

        negative_breed = random.choice([breed for breed in self.valid_breeds if breed != anchor_breed])
        negative_idx = random.choice(self.breed_to_indices[negative_breed])
        negative_image = self._load_image(self.df.iloc[negative_idx]["processed_path"])

        return anchor_image, positive_image, negative_image


class TripletLoss(nn.Module):
    """Margin-based triplet loss."""

    def __init__(self, margin: float = 1.0) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        distance_positive = F.pairwise_distance(anchor, positive, p=2)
        distance_negative = F.pairwise_distance(anchor, negative, p=2)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()


class EmbeddingModel(nn.Module):
    """ViT backbone plus projection head that produces L2-normalised embeddings."""

    def __init__(self, base_model_name: str, lora_path: str | None = None, embedding_dim: int = 768) -> None:
        super().__init__()
        self.vit = ViTModel.from_pretrained(base_model_name)
        if lora_path and Path(lora_path).exists():
            LOGGER.info("Loading LoRA weights from %s", lora_path)
            self.vit = PeftModel.from_pretrained(self.vit, lora_path)
        self.projection = nn.Sequential(
            nn.Linear(self.vit.config.hidden_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, embedding_dim),
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.vit(pixel_values=pixel_values)
        cls_output = outputs.last_hidden_state[:, 0]
        embeddings = self.projection(cls_output)
        return F.normalize(embeddings, p=2, dim=1)


class ContrastiveTrainer:
    """High-level training orchestration for the contrastive head."""

    def __init__(
        self,
        base_model_name: str = "ISxOdin/vit-base-oxford-iiit-pets",
        lora_path: str | None = None,
        output_dir: str = "models/contrastive",
        num_epochs: int = 20,
        batch_size: int = 64,
        learning_rate: float = 1e-4,
        margin: float = 1.0,
        embedding_dim: int = 768,
        num_workers: int = 4,
        early_stop_patience: int = 5,
        resume: str | None = None,
    ) -> None:
        self.base_model_name = base_model_name
        self.lora_path = lora_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.margin = margin
        self.embedding_dim = embedding_dim
        self.exec_config = auto_optimize_for_macos(batch_size=batch_size, num_workers=num_workers)
        self.batch_size = self.exec_config.batch_size
        self.num_workers = self.exec_config.num_workers
        self.device = self.exec_config.device
        self.resume = resume
        dump_execution_plan(self.exec_config, self.output_dir / "execution_plan.json")
        LOGGER.info("Training on %s | batch=%s | workers=%s", self.device, self.batch_size, self.num_workers)
        self.early_stopping = EarlyStopping(patience=early_stop_patience)
        self.checkpoints = CheckpointManager(self.output_dir / "checkpoints")
        self.metrics_recorder = TrainingMetricsRecorder(self.output_dir)
        self.console = Console()
        self.progress_columns = (
            TextColumn("[bold blue]{task.description}", justify="left"),
            BarColumn(bar_width=None, complete_style="green", finished_style="green"),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            RateColumn(),
        )

    def _progress(self, description: str, total: int) -> tuple[Progress, TaskID]:
        progress = Progress(
            *self.progress_columns,
            console=self.console,
            refresh_per_second=5,
            transient=True,
            expand=True,
        )
        task_id = progress.add_task(description, total=total)
        return progress, task_id

    def prepare_dataloaders(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> Tuple[DataLoader, DataLoader]:
        augmentations = RealisticAugmentations(image_size=self.exec_config.image_size, p=0.5)
        train_dataset = TripletDataset(train_df, transform=augmentations.get_training_transform())
        val_dataset = TripletDataset(val_df, transform=augmentations.get_validation_transform())
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=max(1, self.num_workers // 2),
            pin_memory=False,
        )
        return train_loader, val_loader

    def setup_model(self) -> nn.Module:
        model = EmbeddingModel(
            base_model_name=self.base_model_name,
            lora_path=self.lora_path,
            embedding_dim=self.embedding_dim,
        ).to(self.device)
        if self.exec_config.use_compile:
            model = maybe_compile_model(model)
        return model

    def train_epoch(self, model: nn.Module, dataloader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer) -> float:
        model.train()
        total_loss = 0.0
        progress, task_id = self._progress("[green]Treino[/]", len(dataloader))
        with progress:
            for step, (anchor, positive, negative) in enumerate(dataloader, start=1):
                anchor = anchor.to(self.device)
                positive = positive.to(self.device)
                negative = negative.to(self.device)
                optimizer.zero_grad()
                with mixed_precision_context(
                    self.device, self.exec_config.precision, self.exec_config.use_mixed_precision
                ):
                    anchor_emb = model(anchor)
                    positive_emb = model(positive)
                    negative_emb = model(negative)
                    loss = criterion(anchor_emb, positive_emb, negative_emb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                avg_loss = total_loss / step
                progress.update(
                    task_id,
                    advance=1,
                    description=f"[green]Treino[/] loss {avg_loss:.4f}",
                )
        return total_loss / max(1, len(dataloader))

    def validate(self, model: nn.Module, dataloader: DataLoader, criterion: nn.Module) -> float:
        model.eval()
        total_loss = 0.0
        progress, task_id = self._progress("[cyan]Validação[/]", len(dataloader))
        with progress:
            with torch.no_grad():
                for step, (anchor, positive, negative) in enumerate(dataloader, start=1):
                    anchor = anchor.to(self.device)
                    positive = positive.to(self.device)
                    negative = negative.to(self.device)
                    with mixed_precision_context(
                        self.device, self.exec_config.precision, self.exec_config.use_mixed_precision
                    ):
                        anchor_emb = model(anchor)
                        positive_emb = model(positive)
                        negative_emb = model(negative)
                        loss = criterion(anchor_emb, positive_emb, negative_emb)
                    total_loss += loss.item()
                    avg_loss = total_loss / step
                    progress.update(
                        task_id,
                        advance=1,
                        description=f"[cyan]Validação[/] loss {avg_loss:.4f}",
                    )
        return total_loss / max(1, len(dataloader))

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> Tuple[nn.Module, dict]:
        train_loader, val_loader = self.prepare_dataloaders(train_df, val_df)
        model = self.setup_model()
        criterion = TripletLoss(margin=self.margin)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_epochs)
        start_epoch, best_val = self._maybe_resume(model, optimizer, scheduler)
        history: dict[str, List[float]] = {"train_loss": [], "val_loss": []}

        for epoch in range(start_epoch, self.num_epochs + 1):
            LOGGER.info("Epoch %s/%s", epoch, self.num_epochs)
            train_loss = self.train_epoch(model, train_loader, criterion, optimizer)
            val_loss = self.validate(model, val_loader, criterion)
            scheduler.step()
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            metrics = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
            self.metrics_recorder.record(metrics)
            checkpoint_state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": val_loss,
            }
            self.checkpoints.save(f"epoch_{epoch:03d}", checkpoint_state)
            if val_loss < best_val:
                best_val = val_loss
                torch.save(checkpoint_state, self.output_dir / "best_model.pt")
                LOGGER.info("New best validation loss: %.4f", val_loss)
            if self.early_stopping.step(val_loss):
                LOGGER.info("Early stopping triggered at epoch %s", epoch)
                break

        final_path = self.output_dir / "final_model.pt"
        torch.save(model.state_dict(), final_path)
        with (self.output_dir / "training_history.json").open("w", encoding="utf-8") as fp:
            json.dump(history, fp, indent=2)
        LOGGER.info("Training finished. Final model stored at %s", final_path)
        return model, history

    def _find_latest_checkpoint(self) -> Path | None:
        checkpoints = sorted(self.checkpoints.directory.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        return checkpoints[0] if checkpoints else None

    def _maybe_resume(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
    ) -> tuple[int, float]:
        if not self.resume:
            return 1, float("inf")

        resume_path: Path | None = None
        if self.resume.lower() in {"auto", "latest"}:
            resume_path = self._find_latest_checkpoint()
            if resume_path is None:
                LOGGER.warning("Resume requested but no checkpoints found, starting fresh.")
                return 1, float("inf")
        else:
            resume_path = Path(self.resume)
            if not resume_path.exists():
                LOGGER.warning("Resume path %s not found, starting fresh.", resume_path)
                return 1, float("inf")

        LOGGER.info("Resuming training from %s", resume_path)
        checkpoint = torch.load(resume_path, map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        best_val = checkpoint.get("val_loss", float("inf"))
        if best_val != float("inf"):
            self.early_stopping.best = best_val
        start_epoch = checkpoint.get("epoch", 0) + 1
        return start_epoch, best_val


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the BusquePet contrastive model.")
    parser.add_argument("--train-metadata", default="data/train_metadata.csv", help="Path to the train metadata CSV.")
    parser.add_argument("--val-metadata", default="data/val_metadata.csv", help="Path to the validation metadata CSV.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--margin", type=float, default=1.0, help="Triplet margin.")
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to a checkpoint to resume from (or use 'auto' to pick the latest).",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    train_metadata = Path(args.train_metadata)
    val_metadata = Path(args.val_metadata)
    if not train_metadata.exists() or not val_metadata.exists():
        raise FileNotFoundError("Train/validation metadata CSV files are required.")
    train_df = pd.read_csv(train_metadata)
    val_df = pd.read_csv(val_metadata)
    trainer = ContrastiveTrainer(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        margin=args.margin,
        resume=args.resume,
    )
    trainer.train(train_df, val_df)


if __name__ == "__main__":
    main()
