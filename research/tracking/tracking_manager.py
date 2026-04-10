import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import mlflow
import pandas as pd


class HybridTrackingManager:
    """
    Manages both MLflow logging and structured local storage.

    Benefits:
    - MLflow UI for browsing and comparing experiments
    - Local CSV/JSON storage for programmatic analysis
    - Organized by dataset for easy navigation
    - Summary file for quick analysis
    """

    def __init__(self, mlflow_uri: str = "sqlite:///mlflow.db", local_storage: str = "./experiments"):
        self.mlflow_uri = mlflow_uri
        self.local_storage = Path(local_storage)

        # Setup directory structure
        self.predictions_dir = self.local_storage / "predictions"
        self.metrics_dir = self.local_storage / "metrics"
        self.hyperparams_dir = self.local_storage / "hyperparams"
        self.trial_history_dir = self.local_storage / "trial_history"
        self.plots_dir = self.local_storage / "plots"
        self.mlruns_dir = self.local_storage / "mlruns"

        for d in [
            self.predictions_dir,
            self.metrics_dir,
            self.hyperparams_dir,
            self.trial_history_dir,
            self.plots_dir,
            self.mlruns_dir
        ]:
            d.mkdir(parents=True, exist_ok=True)

        mlflow.set_tracking_uri(mlflow_uri)
        self.all_metrics = []

    def log_experiment(
            self,
            dataset: str,
            model: str,
            embedding: str,
            preprocessing: str,
            metrics: Dict[str, float],
            hyperparams: Dict[str, Any],
            predictions: Dict[str, Any],
            trial_history: pd.DataFrame,
            train_time: float,
            inference_time: float,
            best_trial: int,
            run_id: str
    ):
        """
        Log everything to both MLflow and local storage
        """
        exp_key = f"{dataset}_{model}_{embedding}"

        # Save predictions
        pred_df = pd.DataFrame(predictions)
        pred_dir = self.predictions_dir / dataset
        pred_dir.mkdir(parents=True, exist_ok=True)

        pred_path = pred_dir / f"{exp_key}_{run_id}.csv"
        pred_df.to_csv(pred_path, index=False)
        mlflow.log_artifact(str(pred_path), "predictions")

        # Save hyperparameters
        hp_dir = self.hyperparams_dir / dataset
        hp_dir.mkdir(parents=True, exist_ok=True)

        hp_path = hp_dir / f"{exp_key}_{run_id}.json"
        with open(hp_path, 'w') as f:
            json.dump(hyperparams, f, indent=2)
        mlflow.log_artifact(str(hp_path), "hyperparams")

        # Save trial history
        trial_dir = self.trial_history_dir / dataset
        trial_dir.mkdir(parents=True, exist_ok=True)

        trial_path = trial_dir / f"{exp_key}_{run_id}.csv"
        trial_history.to_csv(trial_path, index=False)
        mlflow.log_artifact(str(trial_path), "trial_history")

        # Store metrics in structured format for analysis
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

        # Save to CSV immediately after each experiment (crash-safe)
        self.save_summary_csv()

    def save_summary_csv(self):
        """Save all metrics to single CSV for analysis, with upsert logic:
        overwrite existing rows for the same experiment_key, append new ones."""
        if not self.all_metrics:
            return None

        summary_path = self.metrics_dir / "training_results.csv"
        df_new = pd.DataFrame(self.all_metrics)

        # Load existing data if it exists and merge
        if summary_path.exists():
            df_existing = pd.read_csv(summary_path)
            df = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df = df_new

        # Keep only the latest row for each experiment_key (upsert)
        df = df.drop_duplicates(subset='experiment_key', keep='last')

        # Sort by F1 score
        df = df.sort_values('f1_score', ascending=False)
        df.to_csv(summary_path, index=False)

        mlflow.log_artifact(str(summary_path), "summary")
        return str(summary_path)

    def get_best_experiments(self, top_n: int = 10):
        """Get top N experiments by F1 score"""
        df = pd.DataFrame(self.all_metrics)
        return df.nlargest(top_n, 'f1_score')

    def get_experiments_by_dataset(self, dataset: str):
        """Get all experiments for a specific dataset"""
        df = pd.DataFrame(self.all_metrics)
        return df[df['dataset'] == dataset].sort_values('f1_score', ascending=False)

    def get_experiments_by_model(self, model: str):
        """Get all experiments for a specific model"""
        df = pd.DataFrame(self.all_metrics)
        return df[df['model'] == model].sort_values('f1_score', ascending=False)

    def print_summary_stats(self):
        """Print comprehensive statistics"""
        if not self.all_metrics:
            print("No experiments run yet")
            return

        df = pd.DataFrame(self.all_metrics)

        print("\n" + "=" * 100)
        print("EXPERIMENT SUMMARY STATISTICS")
        print("=" * 100)

        # Overall statistics
        print("\n[OVERALL]")
        print(f"  Total experiments: {len(df)}")
        print(f"  Average F1 Score:  {df['f1_score'].mean():.4f}")
        print(f"  Average Accuracy:  {df['accuracy'].mean():.4f}")
        print(f"  Median F1 Score:   {df['f1_score'].median():.4f}")
        print(f"  Std F1 Score:      {df['f1_score'].std():.4f}")
        print(f"  Min F1 Score:      {df['f1_score'].min():.4f}")
        print(f"  Max F1 Score:      {df['f1_score'].max():.4f}")

        # By Dataset
        print("\n[BY DATASET]")
        dataset_stats = df.groupby('dataset').agg({
            'f1_score': ['count', 'mean', 'max', 'std'],
            'accuracy': ['mean', 'max'],
            'train_time_sec': 'sum'
        }).round(4)
        print(dataset_stats)

        # By Model
        print("\n[BY MODEL]")
        model_stats = df.groupby('model').agg({
            'f1_score': ['count', 'mean', 'max', 'std'],
            'accuracy': ['mean', 'max']
        }).round(4)
        print(model_stats.sort_values(('f1_score', 'mean'), ascending=False))

        # By Embedding
        print("\n[BY EMBEDDING]")
        emb_stats = df.groupby('embedding').agg({
            'f1_score': ['count', 'mean', 'max', 'std'],
            'accuracy': ['mean', 'max']
        }).round(4)
        print(emb_stats.sort_values(('f1_score', 'mean'), ascending=False))

        # Top experiments
        print("\n[TOP 10 EXPERIMENTS]")
        top_10 = df.nlargest(10, 'f1_score')[[
            'experiment_key', 'dataset', 'model', 'embedding',
            'f1_score', 'accuracy', 'train_time_sec'
        ]]
        print(top_10.to_string(index=False))

        # Worst experiments
        print("\n[WORST 5 EXPERIMENTS]")
        worst_5 = df.nsmallest(5, 'f1_score')[[
            'experiment_key', 'dataset', 'model', 'embedding', 'f1_score'
        ]]
        print(worst_5.to_string(index=False))

        # Model-Embedding combinations
        print("\n[BEST MODEL-EMBEDDING COMBINATION BY DATASET]")
        for dataset in df['dataset'].unique():
            dataset_df = df[df['dataset'] == dataset]
            best_combo = dataset_df.loc[dataset_df['f1_score'].idxmax()]
            print(
                f"  {dataset}: "
                f"{best_combo['model']} + {best_combo['embedding']} "
                f"(F1: {best_combo['f1_score']:.4f})"
            )

        print("\n" + "=" * 100 + "\n")

    def load_predictions(self, dataset: str, model: str, embedding: str):
        """Load predictions for a specific experiment"""
        pred_dir = self.predictions_dir / dataset
        csv_files = list(pred_dir.glob(f"{dataset}_{model}_{embedding}_*.csv"))

        if not csv_files:
            return None

        # Load latest file
        latest_file = sorted(csv_files)[-1]
        return pd.read_csv(latest_file)

    def load_hyperparams(self, dataset: str, model: str, embedding: str):
        """Load hyperparameters for a specific experiment"""
        hp_dir = self.hyperparams_dir / dataset
        json_files = list(hp_dir.glob(f"{dataset}_{model}_{embedding}_*.json"))

        if not json_files:
            return None

        latest_file = sorted(json_files)[-1]
        with open(latest_file, 'r') as f:
            return json.load(f)

    def compare_experiments(
            self,
            dataset: str,
            exp_keys: List[str]
    ):
        """Compare multiple experiments"""
        df = pd.DataFrame(self.all_metrics)
        dataset_df = df[df['dataset'] == dataset]

        comparison = dataset_df[dataset_df['experiment_key'].isin(exp_keys)]
        return comparison.sort_values('f1_score', ascending=False)

    def export_report(self, output_path: str = "experiment_report.md"):
        """Export summary report to markdown"""
        if not self.all_metrics:
            return

        df = pd.DataFrame(self.all_metrics)

        with open(output_path, 'w') as f:
            f.write("# Experiment Report\n\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")

            f.write("## Overall Statistics\n\n")
            f.write(f"- Total experiments: {len(df)}\n")
            f.write(f"- Average F1 Score: {df['f1_score'].mean():.4f}\n")
            f.write(f"- Best F1 Score: {df['f1_score'].max():.4f}\n\n")

            f.write("## Top 10 Experiments\n\n")
            f.write(df.nlargest(10, 'f1_score')[[
                'experiment_key', 'f1_score', 'accuracy'
            ]].to_markdown(index=False))

            f.write("\n\n## By Dataset\n\n")
            for dataset in df['dataset'].unique():
                dataset_df = df[df['dataset'] == dataset]
                f.write(f"### {dataset}\n\n")
                f.write(f"- Experiments: {len(dataset_df)}\n")
                f.write(f"- Average F1: {dataset_df['f1_score'].mean():.4f}\n")
                f.write(f"- Best F1: {dataset_df['f1_score'].max():.4f}\n\n")

        print(f"Report saved to {output_path}")
