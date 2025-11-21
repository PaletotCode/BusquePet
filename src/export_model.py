"""
Utilities for exporting BusquePet embedding models to TorchScript and ONNX.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn

from train_contrastive import EmbeddingModel

try:
    import onnx
    from onnx import checker as onnx_checker
except Exception:  # pragma: no cover - optional dependency
    onnx = None
    onnx_checker = None


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

__all__ = ["ModelExporter", "ModelExporterError"]


class ModelExporterError(RuntimeError):
    """Raised when any stage of the export pipeline fails."""


class ModelExporter:
    """Handle exporting trained embedding models for inference use."""

    def __init__(
        self,
        model_path: str,
        base_model_name: str = "ISxOdin/vit-base-oxford-iiit-pets",
        embedding_dim: int = 768,
        export_dir: str = "models/exported",
        image_size: int = 224,
        opset: int = 17,
        device: Optional[str] = None,
        seed: int = 42,
    ) -> None:
        self.model_path = Path(model_path)
        self.base_model_name = base_model_name
        self.embedding_dim = embedding_dim
        self.image_size = image_size
        self.opset = opset
        self.seed = seed
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self.torchscript_path = self.export_dir / "embedding_model.ts"
        self.onnx_path = self.export_dir / "embedding_model.onnx"
        self.metadata_path = self.export_dir / "export_metadata.json"
        try:
            resolved_device = torch.device(device) if device else torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        except (RuntimeError, TypeError) as exc:
            raise ModelExporterError(f"Invalid device specification: {device}") from exc
        self.device = resolved_device

    def export(self) -> Tuple[Path, Path]:
        """Export the trained embedding network to TorchScript and ONNX."""
        model = self._prepare_model()
        dummy_input = self._create_dummy_input()
        torchscript_path = self._export_torchscript(model, dummy_input)
        onnx_path = self._export_onnx(model, dummy_input)
        self._write_metadata(torchscript_path, onnx_path)
        LOGGER.info("Export finished. TorchScript: %s | ONNX: %s", torchscript_path, onnx_path)
        return torchscript_path, onnx_path

    def _prepare_model(self) -> nn.Module:
        model = EmbeddingModel(
            base_model_name=self.base_model_name,
            embedding_dim=self.embedding_dim,
            lora_path=None,
        )
        model = model.to(self.device)
        self._load_trained_weights(model)
        model.eval()
        return model

    def _load_trained_weights(self, model: nn.Module) -> None:
        if not self.model_path.exists():
            raise ModelExporterError(f"Checkpoint not found: {self.model_path}")
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
        except Exception as exc:  # pragma: no cover - torch raises RuntimeError/EOFError/etc.
            raise ModelExporterError(f"Unable to load checkpoint: {self.model_path}") from exc
        state_dict = self._extract_state_dict(checkpoint)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            LOGGER.warning("Missing keys while loading checkpoint: %s", missing)
        if unexpected:
            LOGGER.warning("Unexpected keys while loading checkpoint: %s", unexpected)
        LOGGER.info("Checkpoint successfully loaded from %s", self.model_path)

    @staticmethod
    def _extract_state_dict(checkpoint: object) -> dict:
        if isinstance(checkpoint, nn.Module):
            return checkpoint.state_dict()
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint and isinstance(checkpoint["model_state_dict"], dict):
                return checkpoint["model_state_dict"]
            if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
                return checkpoint["state_dict"]
            if all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
                return checkpoint
        raise ModelExporterError("Checkpoint does not contain a valid state_dict structure.")

    def _create_dummy_input(self) -> torch.Tensor:
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.seed)
        dummy = torch.randn(
            1,
            3,
            self.image_size,
            self.image_size,
            generator=generator,
            device=self.device,
            dtype=torch.float32,
        )
        return dummy

    def _export_torchscript(self, model: nn.Module, example_input: torch.Tensor) -> Path:
        LOGGER.info("Exporting TorchScript model to %s", self.torchscript_path)
        try:
            traced_module = torch.jit.trace(model, example_input)
            traced_module.save(self.torchscript_path.as_posix())
            self._validate_torchscript(traced_module, example_input)
        except Exception as exc:
            raise ModelExporterError("TorchScript export failed.") from exc
        return self.torchscript_path

    @staticmethod
    def _validate_torchscript(traced_module: torch.jit.ScriptModule, example_input: torch.Tensor) -> None:
        try:
            _ = traced_module(example_input)
        except Exception as exc:
            raise ModelExporterError("TorchScript validation failed.") from exc

    def _export_onnx(self, model: nn.Module, example_input: torch.Tensor) -> Path:
        LOGGER.info("Exporting ONNX model to %s", self.onnx_path)
        original_device = next(model.parameters()).device
        cpu_model = model.to(torch.device("cpu"))
        cpu_input = example_input.to(torch.device("cpu"))
        try:
            torch.onnx.export(
                cpu_model,
                cpu_input,
                self.onnx_path.as_posix(),
                input_names=["pixel_values"],
                output_names=["embeddings"],
                dynamic_axes={"pixel_values": {0: "batch"}, "embeddings": {0: "batch"}},
                opset_version=self.opset,
                do_constant_folding=True,
            )
            self._validate_onnx(self.onnx_path)
        except Exception as exc:
            raise ModelExporterError("ONNX export failed.") from exc
        finally:
            cpu_model.to(original_device)
        return self.onnx_path

    def _validate_onnx(self, onnx_path: Path) -> None:
        if onnx is None or onnx_checker is None:
            LOGGER.warning("onnx package is not installed; skipping ONNX validation.")
            return
        try:
            model_proto = onnx.load(onnx_path.as_posix())
            onnx_checker.check_model(model_proto)
        except Exception as exc:  # pragma: no cover - depends on external lib
            raise ModelExporterError("Exported ONNX model failed validation.") from exc

    def _write_metadata(self, torchscript_path: Path, onnx_path: Path) -> Path:
        metadata = {
            "model_path": str(self.model_path.resolve()),
            "torchscript_path": str(torchscript_path.resolve()),
            "onnx_path": str(onnx_path.resolve()),
            "base_model_name": self.base_model_name,
            "embedding_dim": self.embedding_dim,
            "image_size": self.image_size,
            "opset_version": self.opset,
            "device_used": str(self.device),
            "torch_version": torch.__version__,
            "onnx_version": getattr(onnx, "__version__", None),
        }
        with self.metadata_path.open("w", encoding="utf-8") as metadata_file:
            json.dump(metadata, metadata_file, indent=2)
        LOGGER.info("Export metadata stored at %s", self.metadata_path)
        return self.metadata_path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export BusquePet embedding model.")
    parser.add_argument("--model-path", required=True, help="Path to the trained checkpoint.")
    parser.add_argument(
        "--base-model-name",
        default="ISxOdin/vit-base-oxford-iiit-pets",
        help="Name of the base ViT model used during training.",
    )
    parser.add_argument("--embedding-dim", type=int, default=768, help="Embedding dimension size.")
    parser.add_argument("--export-dir", default="models/exported", help="Directory to store exported files.")
    parser.add_argument("--image-size", type=int, default=224, help="Input spatial size expected by the model.")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    parser.add_argument("--device", default=None, help="Device identifier (cpu, cuda, cuda:1, ...).")
    parser.add_argument("--seed", type=int, default=42, help="Seed for dummy input generation.")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    try:
        exporter = ModelExporter(
            model_path=args.model_path,
            base_model_name=args.base_model_name,
            embedding_dim=args.embedding_dim,
            export_dir=args.export_dir,
            image_size=args.image_size,
            opset=args.opset,
            device=args.device,
            seed=args.seed,
        )
        exporter.export()
    except ModelExporterError as exc:
        LOGGER.error("Model export failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
