"""BusquePet Diagnostics Report."""

from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path
from typing import Dict, List

import psutil

from system_utils import configure_logging

LOGGER = configure_logging(__name__)


DEFAULT_PATHS = {
    "model": Path("models/contrastive/best_model.pt"),
    "faiss": Path("models/faiss_index/faiss_index.bin"),
    "metadata": Path("data/embeddings/embedding_metadata.csv"),
    "data_dir": Path("data"),
}


def check_python_environment() -> Dict[str, str]:
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "executable": sys.executable,
    }


def check_installed_packages() -> Dict[str, str]:
    packages = ["torch", "transformers", "faiss", "albumentations", "psutil"]
    results: Dict[str, str] = {}
    for package in packages:
        try:
            module = __import__(package)
            version = getattr(module, "__version__", "unknown")
            results[package] = str(version)
        except Exception as err:
            results[package] = f"missing ({err})"
    return results


def validate_paths(paths: Dict[str, Path]) -> Dict[str, bool]:
    return {name: path.exists() for name, path in paths.items()}


def check_permissions(paths: List[Path]) -> Dict[str, bool]:
    return {str(path): os.access(path, os.W_OK) for path in paths}


def available_memory() -> Dict[str, float]:
    vm = psutil.virtual_memory()
    return {"total_gb": round(vm.total / (1024 ** 3), 2), "available_gb": round(vm.available / (1024 ** 3), 2)}


def run_inference_smoke_test() -> str:
    try:
        from inference import PetMatchingPipeline

        pipeline = PetMatchingPipeline(
            model_path=str(DEFAULT_PATHS["model"]),
            faiss_index_path=str(DEFAULT_PATHS["faiss"]),
            metadata_path=str(DEFAULT_PATHS["metadata"]),
        )
        sample = DEFAULT_PATHS["data_dir"] / "processed" / "example_query.jpg"
        if not sample.exists():
            return "SKIPPED (sample query not found)"
        pipeline.search(str(sample), k=1, use_hybrid=True)
        return "OK"
    except Exception as err:
        return f"FAIL ({err})"


def main() -> None:
    report = {
        "python": check_python_environment(),
        "packages": check_installed_packages(),
        "paths": validate_paths(DEFAULT_PATHS),
        "permissions": check_permissions(list(DEFAULT_PATHS.values())),
        "memory": available_memory(),
        "inference_smoke_test": run_inference_smoke_test(),
    }
    status = "OK"
    if not all(report["paths"].values()) or "FAIL" in report["inference_smoke_test"]:
        status = "FAIL"
    print("BusquePet Diagnostics Report")
    print(f"Status: {status}")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
