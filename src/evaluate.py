"""Evaluation utilities for the BusquePet retrieval pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from inference import PetMatchingPipeline
from system_utils import configure_logging

LOGGER = configure_logging(__name__)


class Evaluator:
    """Compute recall, threshold curves, and confusion matrices."""

    def __init__(self, pipeline: PetMatchingPipeline, test_df: pd.DataFrame, output_dir: str = "outputs/metrics") -> None:
        self.pipeline = pipeline
        self.test_df = test_df.reset_index(drop=True)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def compute_recall_at_k(self, k_values: List[int] | None = None) -> Dict[int, float]:
        k_values = k_values or [1, 5, 10, 20]
        recall_scores = {k: 0 for k in k_values}
        max_k = max(k_values)
        for _, row in tqdm(self.test_df.iterrows(), total=len(self.test_df), desc="Recall@K"):
            results = self.pipeline.search(row["processed_path"], k=max_k, use_hybrid=True)
            breeds = [result["breed"] for result in results]
            for k in k_values:
                if row["breed"] in breeds[:k]:
                    recall_scores[k] += 1
        recall_scores = {k: score / len(self.test_df) for k, score in recall_scores.items()}
        with (self.output_dir / "recall_at_k.json").open("w", encoding="utf-8") as fp:
            json.dump(recall_scores, fp, indent=2)
        self._plot_recall_at_k(recall_scores)
        return recall_scores

    def _plot_recall_at_k(self, recall_scores: Dict[int, float]) -> None:
        plt.figure(figsize=(8, 5))
        ks = sorted(recall_scores.keys())
        values = [recall_scores[k] for k in ks]
        plt.plot(ks, values, marker="o")
        plt.xlabel("K")
        plt.ylabel("Recall@K")
        plt.ylim(0, 1.05)
        plt.grid(True, alpha=0.3)
        plt.savefig(self.output_dir / "recall_at_k.png", dpi=150)
        plt.close()

    def threshold_curve(self, n_samples: int = 200, thresholds: np.ndarray | None = None) -> Tuple[float, pd.DataFrame]:
        thresholds = thresholds or np.linspace(0.1, 0.95, 20)
        records: List[Dict] = []
        sample_df = self.test_df.sample(n=min(n_samples, len(self.test_df)), random_state=42)
        for _, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Thresholds"):
            results = self.pipeline.search(row["processed_path"], k=10, use_hybrid=True)
            top_score = results[0]["score"] if results else 0.0
            label = 1 if results and results[0]["breed"] == row["breed"] else 0
            for threshold in thresholds:
                prediction = 1 if top_score >= threshold else 0
                records.append({"threshold": float(threshold), "prediction": prediction, "label": label})
        df = pd.DataFrame(records)
        threshold_stats = []
        best_threshold = thresholds[0]
        best_f1 = -1.0
        for threshold, group in df.groupby("threshold"):
            tp = ((group["prediction"] == 1) & (group["label"] == 1)).sum()
            fp = ((group["prediction"] == 1) & (group["label"] == 0)).sum()
            fn = ((group["prediction"] == 0) & (group["label"] == 1)).sum()
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-8)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
            threshold_stats.append(
                {
                    "threshold": threshold,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }
            )
        threshold_df = pd.DataFrame(threshold_stats)
        threshold_df.to_csv(self.output_dir / "threshold_curve.csv", index=False)
        self._plot_thresholds(threshold_df)
        return float(best_threshold), threshold_df

    def _plot_thresholds(self, threshold_df: pd.DataFrame) -> None:
        plt.figure(figsize=(8, 5))
        for metric in ["precision", "recall", "f1"]:
            plt.plot(threshold_df["threshold"], threshold_df[metric], label=metric)
        plt.xlabel("Threshold")
        plt.ylabel("Metric value")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(self.output_dir / "threshold_curves.png", dpi=150)
        plt.close()

    def compute_confusion_matrix(self, n_samples: int = 200) -> Tuple[np.ndarray, Dict]:
        sample_df = self.test_df.sample(n=min(n_samples, len(self.test_df)), random_state=42)
        y_true: List[str] = []
        y_pred: List[str] = []
        for _, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Confusion"):
            results = self.pipeline.search(row["processed_path"], k=1, use_hybrid=True)
            predicted = results[0]["breed"] if results else "unknown"
            y_true.append(row["breed"])
            y_pred.append(predicted)
        labels = sorted(set(y_true + y_pred))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(self.output_dir / "confusion_matrix.png", dpi=150)
        plt.close()
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)
        cm_df.to_csv(self.output_dir / "confusion_matrix.csv")
        report = classification_report(y_true, y_pred, labels=labels, output_dict=True)
        with (self.output_dir / "classification_report.json").open("w", encoding="utf-8") as fp:
            json.dump(report, fp, indent=2)
        return cm, report

    def run_full_evaluation(self) -> Dict:
        results: Dict[str, object] = {}
        results["recall_at_k"] = self.compute_recall_at_k([1, 5, 10, 20, 50])
        best_threshold, threshold_df = self.threshold_curve()
        results["best_threshold"] = best_threshold
        results["threshold_metrics"] = threshold_df.to_dict("records")
        _, classification_rep = self.compute_confusion_matrix()
        results["classification_report"] = classification_rep
        with (self.output_dir / "evaluation_results.json").open("w", encoding="utf-8") as fp:
            json.dump(results, fp, indent=2)
        LOGGER.info("Evaluation finished. Metrics stored in %s", self.output_dir)
        return results


def main() -> None:
    test_csv = Path("data/val_metadata.csv")
    if not test_csv.exists():
        raise FileNotFoundError(test_csv)
    pipeline = PetMatchingPipeline(
        model_path="models/contrastive/best_model.pt",
        faiss_index_path="models/faiss_index",
        metadata_path="data/embeddings/embedding_metadata.csv",
    )
    evaluator = Evaluator(pipeline, pd.read_csv(test_csv))
    evaluator.run_full_evaluation()


if __name__ == "__main__":
    main()
