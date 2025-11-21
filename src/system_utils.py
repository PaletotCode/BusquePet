"""System-level helpers for BusquePet pipelines."""

from __future__ import annotations

import json
import logging
import os
import platform
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import psutil
import torch

LOGGER = logging.getLogger(__name__)


def configure_logging(logger_name: str, log_file: str = "outputs/logs/train.log") -> logging.Logger:
    """Configure and return a logger that logs to stdout and to *log_file*."""
    logger = logging.getLogger(logger_name)
    if getattr(logger, "_busquepet_configured", False):
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger._busquepet_configured = True  # type: ignore[attr-defined]
    return logger


@dataclass
class ExecutionConfig:
    """Holds runtime parameters computed by :func:`auto_optimize_for_macos`."""

    device: torch.device
    batch_size: int
    num_workers: int
    use_mps: bool
    precision: torch.dtype
    use_mixed_precision: bool
    use_compile: bool
    num_threads: int
    image_size: int


def _estimate_sample_memory_mb(image_size: int) -> float:
    channels = 3
    bytes_per_channel = 4  # float32
    return (channels * image_size * image_size * bytes_per_channel) / (1024 ** 2)


def auto_optimize_for_macos(
    batch_size: int,
    num_workers: int,
    image_size: int = 224,
    prefer_mps: bool = True,
) -> ExecutionConfig:
    """Return a conservative execution plan tuned for macOS laptops (prefers MPS when available)."""

    system = platform.system().lower()
    is_macos = system == "darwin"
    available_mem = psutil.virtual_memory().available / (1024 ** 2)  # MB
    sample_mem = max(_estimate_sample_memory_mb(image_size), 1.0)
    safety_factor = 0.4
    max_batch_by_memory = max(1, int((available_mem * safety_factor) / sample_mem))
    adjusted_batch = max(1, min(batch_size, max_batch_by_memory))

    cpu_count = os.cpu_count() or 4
    adjusted_workers = max(1, min(num_workers, cpu_count // 2 or 1))
    if is_macos:
        adjusted_workers = min(adjusted_workers, 2)

    torch_threads = max(1, min(int(cpu_count * 0.75), 16))
    torch.set_num_threads(torch_threads)

    if is_macos and prefer_mps and torch.backends.mps.is_available():
        device = torch.device("mps")
        use_mps = True
    elif torch.cuda.is_available() and not is_macos:
        device = torch.device("cuda")
        use_mps = False
    else:
        device = torch.device("cpu")
        use_mps = False

    precision = torch.float32
    use_mixed_precision = False
    if device.type in {"cuda", "mps"}:
        precision = torch.float16 if device.type == "cuda" else torch.float16
        use_mixed_precision = True
    elif hasattr(torch, "bfloat16"):
        precision = torch.bfloat16
        use_mixed_precision = True

    use_compile = bool(getattr(torch, "compile", None)) and device.type != "mps"

    return ExecutionConfig(
        device=device,
        batch_size=adjusted_batch,
        num_workers=adjusted_workers,
        use_mps=use_mps,
        precision=precision,
        use_mixed_precision=use_mixed_precision,
        use_compile=use_compile,
        num_threads=torch_threads,
        image_size=image_size,
    )


def maybe_compile_model(model: torch.nn.Module, mode: str = "reduce-overhead") -> torch.nn.Module:
    """Compile the model with ``torch.compile`` when available."""
    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:
        return model
    try:
        return compile_fn(model, mode=mode)
    except Exception as exc:  # pragma: no cover - fallback path
        LOGGER.warning("torch.compile failed, continuing in eager mode: %s", exc)
        return model


@contextmanager
def mixed_precision_context(device: torch.device, precision: torch.dtype, enabled: bool):
    """Context manager that activates :func:`torch.autocast` when supported."""
    if not enabled or not hasattr(torch, "autocast"):
        yield
        return

    device_type = "cuda" if device.type == "cuda" else device.type
    try:
        with torch.autocast(device_type=device_type, dtype=precision):
            yield
    except RuntimeError:
        # Some CPU builds do not allow autocast; silently fallback.
        yield


class EarlyStopping:
    """Standard patience-based early stopping helper."""

    def __init__(self, patience: int = 5, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best: Optional[float] = None
        self.bad_epochs = 0

    def step(self, metric: float) -> bool:
        if self.best is None or metric < self.best - self.min_delta:
            self.best = metric
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


class CheckpointManager:
    """Persist checkpoints while keeping the latest *keep* files."""

    def __init__(self, directory: Path, keep: int = 3) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.keep = keep
        self.saved_paths: List[Path] = []

    def save(self, tag: str, state: Dict) -> Path:
        path = self.directory / f"{tag}.pt"
        torch.save(state, path)
        self.saved_paths.append(path)
        if len(self.saved_paths) > self.keep:
            old = self.saved_paths.pop(0)
            if old.exists():
                old.unlink(missing_ok=True)
        return path


class TrainingMetricsRecorder:
    """Append metrics to disk so dashboards can track training progress."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history_path = self.output_dir / "training_metrics.json"
        self.history: List[Dict] = []

    def record(self, metrics: Dict) -> None:
        self.history.append(metrics)
        with self.history_path.open("w", encoding="utf-8") as fp:
            json.dump(self.history, fp, indent=2)


def ensure_paths_exist(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required paths: {', '.join(missing)}")


def dump_execution_plan(config: ExecutionConfig, output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(asdict(config), fp, indent=2, default=str)
