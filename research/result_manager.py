# research/result_manager.py
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional


class ResultsManager:
    """Centralized management for predictions, metrics, and experiment metadata"""

    def __init__(self, base_dir: str = "./experiments"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.predictions_dir = self.base_dir / "predictions"
        self.metrics_dir = self.base_dir / "metrics"
        self.models_dir = self.base_dir / "models"
        self.logs_dir = self.base_dir / "logs"

        for d in [self.predictions_dir, self.metrics_dir, self.models_dir, self.logs_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Summary dataframes
        self.all_metrics = []
        self.all_predictions = {}

    def get_experiment_key(
            self,
            dataset: str,
            model: str,
            embedding: str,
            preprocessing: str = "standard"
    ) -> str:
        """Generate unique key for experiment"""
        return f"{dataset}_{model}_{embedding}_{preprocessing}"

    def save_predictions(
            self,
            y_test: np.ndarray,
            y_pred: np.ndarray,
            y_probs: np.ndarray,
            dataset: str,
            model: str,
            embedding: str,
            preprocessing: str = "standard",
            run_id: str = None
    ) -> str:
        """Save predictions to structured directory"""
        exp_key = self.get_experiment_key(dataset, model, embedding, preprocessing)

        # Create dataset-specific subdirectory
        dataset_pred_dir = self.predictions_dir / dataset
        dataset_pred_dir.mkdir(parents=True, exist_ok=True)

        # Filename with run_id for uniqueness
        filename = f"{exp_key}_{run_id}.csv" if run_id else f"{exp_key}.csv"
        filepath = dataset_pred_dir / filename

        df = pd.DataFrame({
            "text_id": range(len(y_test)),
            "true_label": y_test,
            "predicted_label": y_pred,
            "probability": y_probs,
            "correct": y_test == y_pred
        })

        df.to_csv(filepath, index=False)
        return str(filepath)

    def save_metrics(
            self,
            dataset: str,
            model: str,
            embedding: str,
            preprocessing: str,
            metrics: Dict[str, float],
            train_time: float,
            inference_time: float,
            best_trial: Optional[int] = None,
            run_id: str = None,
            hyperparams: Optional[Dict] = None
    ) -> dict:
        """Save metrics with full experiment metadata"""
        exp_key = self.get_experiment_key(dataset, model, embedding, preprocessing)

        row = {
            "experiment_key": exp_key,
            "dataset": dataset,
            "model": model,
            "embedding": embedding,
            "preprocessing": preprocessing,
            "accuracy": round(metrics.get("accuracy", 0), 4),
            "precision": round(metrics.get("precision", 0), 4),
            "recall": round(metrics.get("recall", 0), 4),
            "f1_score": round(metrics.get("f1_score", 0), 4),
            "train_time_sec": round(train_time, 2),
            "inference_time_sec": round(inference_time, 4),
            "best_trial": best_trial,
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
        }

        # Add confusion matrix if available
        if "confusion_matrix" in metrics:
            row["confusion_matrix"] = str(metrics["confusion_matrix"])

        self.all_metrics.append(row)

        return row

    def save_hyperparams(
            self,
            dataset: str,
            model: str,
            embedding: str,
            preprocessing: str,
            hyperparams: Dict[str, Any],
            run_id: str = None
    ) -> str:
        """Save hyperparameters to JSON"""
        exp_key = self.get_experiment_key(dataset, model, embedding, preprocessing)

        dataset_hp_dir = self.metrics_dir / dataset / "hyperparams"
        dataset_hp_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{exp_key}_{run_id}.json" if run_id else f"{exp_key}.json"
        filepath = dataset_hp_dir / filename

        with open(filepath, 'w') as f:
            json.dump(hyperparams, f, indent=2)

        return str(filepath)

    def save_trial_history(
            self,
            dataset: str,
            model: str,
            embedding: str,
            preprocessing: str,
            study: Any,  # optuna.Study
            run_id: str = None
    ) -> str:
        """Save Optuna trial history"""
        exp_key = self.get_experiment_key(dataset, model, embedding, preprocessing)

        dataset_trials_dir = self.metrics_dir / dataset / "trial_history"
        dataset_trials_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{exp_key}_{run_id}.csv" if run_id else f"{exp_key}.csv"
        filepath = dataset_trials_dir / filename

        trials_df = study.trials_dataframe()
        trials_df.to_csv(filepath, index=False)

        return str(filepath)

    def save_summary_metrics_csv(self) -> str:
        """Save all metrics to single CSV for easy analysis"""
        if not self.all_metrics:
            return None

        summary_path = self.metrics_dir / "all_metrics_summary.csv"
        summary_df = pd.DataFrame(self.all_metrics)
        summary_df.to_csv(summary_path, index=False)

        print(f"\n[Summary Saved] {summary_path}")
        return str(summary_path)

    def get_experiment_report(
            self,
            dataset: Optional[str] = None,
            model: Optional[str] = None
    ) -> pd.DataFrame:
        """Generate report filtered by dataset/model"""
        df = pd.DataFrame(self.all_metrics)

        if dataset:
            df = df[df['dataset'] == dataset]
        if model:
            df = df[df['model'] == model]

        return df.sort_values('f1_score', ascending=False)

    def print_experiment_summary(self) -> None:
        """Print comprehensive summary"""
        if not self.all_metrics:
            print("No experiments run yet")
            return

        df = pd.DataFrame(self.all_metrics)

        print("\n" + "=" * 100)
        print("EXPERIMENT SUMMARY")
        print("=" * 100)

        # Summary by dataset
        print("\n[BY DATASET]")
        dataset_summary = df.groupby('dataset').agg({
            'f1_score': ['mean', 'max', 'min', 'count'],
            'accuracy': 'mean',
            'train_time_sec': 'sum'
        }).round(4)
        print(dataset_summary)

        # Summary by model
        print("\n[BY MODEL]")
        model_summary = df.groupby('model').agg({
            'f1_score': ['mean', 'max', 'min'],
            'accuracy': 'mean'
        }).round(4)
        print(model_summary)

        # Summary by embedding
        print("\n[BY EMBEDDING]")
        embedding_summary = df.groupby('embedding').agg({
            'f1_score': ['mean', 'max', 'min'],
            'accuracy': 'mean'
        }).round(4)
        print(embedding_summary)

        # Top 10 experiments
        print("\n[TOP 10 EXPERIMENTS]")
        top_10 = df.nlargest(10, 'f1_score')[
            ['experiment_key', 'dataset', 'model', 'embedding',
             'f1_score', 'accuracy', 'train_time_sec']
        ]
        print(top_10.to_string(index=False))

        # Worst 5 experiments
        print("\n[WORST 5 EXPERIMENTS]")
        worst_5 = df.nsmallest(5, 'f1_score')[
            ['experiment_key', 'dataset', 'model', 'embedding', 'f1_score']
        ]
        print(worst_5.to_string(index=False))

        print("\n" + "=" * 100 + "\n")


class PredictionsAnalyzer:
    """Analyze predictions across experiments"""

    def __init__(self, predictions_dir: str = "./experiments/predictions"):
        self.predictions_dir = Path(predictions_dir)

    def load_predictions(self, experiment_key: str) -> Optional[pd.DataFrame]:
        """Load predictions for an experiment"""
        csv_files = list(self.predictions_dir.glob(f"**/*{experiment_key}*.csv"))

        if not csv_files:
            return None

        # Load latest file if multiple exist
        csv_file = sorted(csv_files)[-1]
        return pd.read_csv(csv_file)

    def compare_predictions(
            self,
            exp_key_1: str,
            exp_key_2: str,
            dataset: str
    ) -> Dict[str, Any]:
        """Compare predictions between two experiments"""
        dataset_pred_dir = self.predictions_dir / dataset

        df1 = pd.read_csv(sorted(dataset_pred_dir.glob(f"*{exp_key_1}*.csv"))[-1])
        df2 = pd.read_csv(sorted(dataset_pred_dir.glob(f"*{exp_key_2}*.csv"))[-1])

        agreement = (df1['predicted_label'] == df2['predicted_label']).sum() / len(df1)
        both_correct = ((df1['correct']) & (df2['correct'])).sum()
        one_correct = ((df1['correct']) ^ (df2['correct'])).sum()

        return {
            'agreement': agreement,
            'both_correct': both_correct,
            'disagreement_with_split': one_correct,
            'both_incorrect': len(df1) - both_correct - one_correct
        }

    def get_hard_examples(
            self,
            dataset: str,
            model: str,
            embedding: str,
            top_n: int = 10
    ) -> pd.DataFrame:
        """Get most uncertain predictions"""
        exp_key = f"{dataset}_{model}_{embedding}"
        dataset_pred_dir = self.predictions_dir / dataset

        df = pd.read_csv(sorted(dataset_pred_dir.glob(f"*{exp_key}*.csv"))[-1])

        # Sort by probability (confidence)
        df['confidence'] = np.abs(df['probability'] - 0.5)
        return df.nsmallest(top_n, 'confidence')

    def generate_confusion_matrices_report(self, dataset: str) -> str:
        """Generate report with confusion matrices"""
        report_path = self.predictions_dir / dataset / f"{dataset}_confusion_report.txt"

        with open(report_path, 'w') as f:
            f.write(f"Confusion Matrices Report - {dataset}\n")
            f.write("=" * 80 + "\n\n")

            # Load all predictions for dataset
            for pred_file in sorted(self.predictions_dir.glob(f"{dataset}/*.csv")):
                exp_name = pred_file.stem
                df = pd.read_csv(pred_file)

                from sklearn.metrics import confusion_matrix
                cm = confusion_matrix(df['true_label'], df['predicted_label'])

                f.write(f"\n{exp_name}\n")
                f.write(f"{cm}\n")

        return str(report_path)