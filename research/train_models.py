import os
import random
import traceback
from datetime import datetime
from typing import Dict, List

import hydra
import mlflow
import numpy as np
import optuna
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from optuna.samplers import TPESampler
import copy

from research.configs.embeddings.embedding_factory import EmbedderFactory
from research.configs.models.model_factory import ModelFactory
from research.configs.preprocessing.preprocessor import TextPreprocessor
from research.tracking.tracking_manager import HybridTrackingManager


def get_completed_experiments(results_dir: str) -> set:
    """Load experiment keys that already finished successfully.

    Reads experiments/metrics/training_results.csv (persisted on Google
    Drive via colab_setup symlinks) and returns a set of experiment_key
    strings like ``ISOT_bert_bert-base-uncased``.
    """
    csv_path = os.path.join(os.path.abspath(results_dir), "metrics", "training_results.csv")
    if not os.path.exists(csv_path):
        return set()
    try:
        df = pd.read_csv(csv_path)
        keys = set(df["experiment_key"].dropna().unique())
        print(f"[Resume] Found {len(keys)} completed experiment(s) in {csv_path}")
        return keys
    except Exception as e:
        print(f"[Resume] Could not read results CSV: {e}")
        return set()


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_data(base_dir: str, dataset_name: str, split_type: str):
    base_dir = os.path.abspath(base_dir)
    file_path = os.path.join(base_dir, dataset_name, f"{split_type}.csv")

    if not os.path.exists(file_path):
        print(f"[Warning] File not found: {file_path}")
        return [], []

    df = pd.read_csv(file_path).dropna(subset=['text', 'label'])
    return df['text'].astype(str).tolist(), df['label'].tolist()


def preprocess_data(texts: List[str], preprocessing_name: str):
    """Preprocess text data"""
    preprocessor = TextPreprocessor(mode=preprocessing_name)
    return preprocessor.transform(texts)


def embed_data(texts: List[str], embedding_name: str, fit: bool = False, embedder=None):
    """Embed text data"""
    if fit:
        embedder = EmbedderFactory.create(embedding_name)
        embeddings = embedder.fit_transform(texts)
        return embeddings, embedder
    else:
        embeddings = embedder.transform(texts)
        return embeddings, embedder


def suggest_model_params(trial: optuna.Trial, optuna_config: Dict):
    """Suggest hyperparameters based on config"""
    params = {}

    for param_name, param_config in optuna_config.items():
        param_type = param_config.get('type', 'float')

        if param_type == 'float':
            low = param_config['low']
            high = param_config['high']
            log = param_config.get('log', False)
            params[param_name] = trial.suggest_float(param_name, low, high, log=log)

        elif param_type == 'int':
            low = param_config['low']
            high = param_config['high']
            params[param_name] = trial.suggest_int(param_name, low, high)

        elif param_type == 'categorical':
            choices = param_config['choices']
            params[param_name] = trial.suggest_categorical(param_name, choices)

    return params


def validate_compatibility(model_name: str, embedding_name: str, preprocessing_name: str = None):
    """Validate model-embedding-preprocessing compatibility"""
    if embedding_name not in ModelFactory.get_valid_embeddings_for_model(model_name):
        raise ValueError(
            f"Invalid embeddings '{embedding_name}' for model '{model_name}'"
        )

    expected_preprocessing = ModelFactory.get_preprocessing_for_model(model_name)
    if preprocessing_name and preprocessing_name != expected_preprocessing:
        print(
            f"[Warning] Preprocessing override: config says '{preprocessing_name}', "
            f"but PREPROCESSING_MAP requires '{expected_preprocessing}' for model '{model_name}'. "
            f"Using '{expected_preprocessing}'."
        )

    return True


def get_predictions_and_probs(model, X_test_vec: np.ndarray):
    """Get predictions and probabilities from model"""
    if model.scaler:
        X_test_scaled = model.scaler.transform(X_test_vec)
    else:
        X_test_scaled = X_test_vec

    y_pred = model.classifier.predict(X_test_scaled)
    y_probs = model.predict_proba_on_vectors(X_test_vec)

    return y_pred, y_probs


def get_predictions_and_probs_transformer(model, X_test_texts: List[str]):
    """Get predictions and probabilities from a transformer model (operates on raw text)."""
    y_pred, y_probs = model._predict_batch(X_test_texts)
    return y_pred, y_probs


def run_single_trial(
    trial: optuna.Trial,
    cfg: DictConfig,
    X_train_vec: np.ndarray,
    X_test_vec: np.ndarray,
    y_train: List[int],
    y_test: List[int],
    model_name: str,
    embedding_name: str,
    dataset_name: str,
    preprocessing_name: str,
    embedding_object
):
    """Run a single Optuna trial"""
    try:
        # Get model config and optuna search space
        model_config = OmegaConf.to_container(cfg.models.get(model_name, {}))
        optuna_config = model_config.pop('optuna', {})

        # Suggest hyperparameters
        suggested_params = suggest_model_params(trial, optuna_config)

        # Merge with base model params
        model_params = {**model_config, **suggested_params}
        model_params.pop('optuna', None)

        trial_run_name = (
            f"{dataset_name}_{model_name}_{embedding_name}_{preprocessing_name}_trial{trial.number}"
        )

        # Start nested MLflow run for this trial
        with mlflow.start_run(run_name=trial_run_name, nested=True):
            mlflow.log_params({
                "trial_number": trial.number,
                **suggested_params
            })

            # Create and train model
            model = ModelFactory.get_model(
                dataset_name=dataset_name,
                model_type=model_name,
                embedding_type=embedding_name,
                **model_params
            )

            model.embedder = copy.deepcopy(embedding_object)

            # Train
            model.train_on_vectors(X_train_vec, y_train)

            # Evaluate
            metrics = model.evaluate_on_vectors(X_test_vec, y_test)

            # Log metrics
            mlflow.log_metrics({
                "trial_accuracy": metrics["accuracy"],
                "trial_f1_score": metrics["f1_score"],
                "trial_precision": metrics["precision"],
                "trial_recall": metrics["recall"],
            })

            trial_f1 = metrics["f1_score"]
            return trial_f1

    except Exception as e:
        print(f"Trial {trial.number} FAILED: {str(e)}")
        return 0.0


def run_single_trial_transformer(
    trial: optuna.Trial,
    cfg: DictConfig,
    X_train_texts: List[str],
    X_test_texts: List[str],
    y_train: List[int],
    y_test: List[int],
    model_name: str,
    embedding_name: str,
    dataset_name: str,
    preprocessing_name: str,
):
    """Run a single Optuna trial for a transformer model (operates on raw text)."""
    try:
        # Get model config and optuna search space
        model_config = OmegaConf.to_container(cfg.models.get(model_name, {}))
        optuna_config = model_config.pop('optuna', {})

        # Suggest hyperparameters
        suggested_params = suggest_model_params(trial, optuna_config)

        # Merge with base model params
        model_params = {**model_config, **suggested_params}
        model_params.pop('optuna', None)

        trial_run_name = (
            f"{dataset_name}_{model_name}_{embedding_name}_{preprocessing_name}_trial{trial.number}"
        )

        # Start nested MLflow run for this trial
        with mlflow.start_run(run_name=trial_run_name, nested=True):
            mlflow.log_params({
                "trial_number": trial.number,
                **suggested_params
            })

            # Create transformer model
            model = ModelFactory.get_model(
                dataset_name=dataset_name,
                model_type=model_name,
                embedding_type=embedding_name,
                **model_params
            )

            # Train on raw text
            model.train(X_train_texts, y_train)

            # Evaluate on raw text
            metrics = model.evaluate(X_test_texts, y_test)

            # Log metrics
            mlflow.log_metrics({
                "trial_accuracy": metrics["accuracy"],
                "trial_f1_score": metrics["f1_score"],
                "trial_precision": metrics["precision"],
                "trial_recall": metrics["recall"],
            })

            trial_f1 = metrics["f1_score"]
            return trial_f1

    except Exception as e:
        print(f"Trial {trial.number} FAILED: {str(e)}")
        traceback.print_exc()
        return 0.0


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """
    Main training loop with Optuna hyperparameter optimization
    Runs across multiple datasets, models, and embeddings
    """
    set_seed(cfg.seed)

    # Initialize tracking manager
    tracking_mgr = HybridTrackingManager(
        mlflow_uri=cfg.mlflow.tracking_uri,
        local_storage=cfg.get('results_dir', './experiments')
    )

    # Configuration
    datasets = cfg.get('datasets_list', [cfg.datasets.name])
    models_to_optimize = cfg.get('models_to_optimize', [cfg.models.name])
    embeddings_to_use = cfg.get('embeddings_to_use', [cfg.embeddings.name])
    cfg_preprocessing_name = cfg.preprocessing.name  # fallback from config

    # Print header
    print_header(datasets, models_to_optimize, embeddings_to_use, cfg)

    # Setup MLflow
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    try:
        mlflow.set_experiment(cfg.mlflow.experiment_name)
    except Exception as e:
        # Handle incompatible MLflow DB schema (version mismatch)
        mlflow_uri = cfg.mlflow.tracking_uri
        print(f"[MLflow] DB schema error: {e}")
        recovered = False
        if mlflow_uri and mlflow_uri.startswith("sqlite:///"):
            db_path = mlflow_uri.replace("sqlite:///", "")
            if os.path.exists(db_path):
                try:
                    os.remove(db_path)
                    print(f"[MLflow] Removed incompatible DB: {db_path}")
                    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
                    mlflow.set_experiment(cfg.mlflow.experiment_name)
                    recovered = True
                except OSError:
                    print(f"[MLflow] DB file locked, falling back to file-based tracking")
        if not recovered:
            fallback_uri = "./experiments/mlruns"
            print(f"[MLflow] Using fallback tracking URI: {fallback_uri}")
            mlflow.set_tracking_uri(fallback_uri)
            mlflow.set_experiment(cfg.mlflow.experiment_name)

    experiment_count = 0
    total_combinations = (
        len(datasets) * len(models_to_optimize) * len(embeddings_to_use)
    )
    failed_experiments = []

    # Resume support: skip experiments that already completed
    resume_enabled = cfg.get("resume", True)
    completed_keys: set = set()
    if resume_enabled:
        completed_keys = get_completed_experiments(
            cfg.get("results_dir", "./experiments")
        )

    # Main loop: Datasets -> Models -> Embeddings
    for dataset_idx, dataset_name in enumerate(datasets, 1):
        print(f"\n{'#'*100}")
        print(f"[{dataset_idx}/{len(datasets)}] DATASET: {dataset_name}")
        print(f"{'#'*100}\n")

        # Load data once per dataset
        print("[Loading Data]")
        X_train_raw, y_train = load_data(cfg.base_data_dir, dataset_name, "train")
        X_test_raw, y_test = load_data(cfg.base_data_dir, dataset_name, "test")

        if not X_train_raw or not X_test_raw:
            print(f"[SKIP] Missing data for {dataset_name}")
            failed_experiments.append((dataset_name, "N/A", "N/A", "Missing data"))
            continue

        print(f"Train samples: {len(X_train_raw)}")
        print(f"Test samples: {len(X_test_raw)}")

        # Cache preprocessed data per preprocessing mode (avoid re-processing)
        preprocessing_cache_train = {}
        preprocessing_cache_test = {}

        # Loop over models and embeddings
        for model_name in models_to_optimize:
            # Resolve preprocessing mode for this model from PREPROCESSING_MAP
            preprocessing_name = ModelFactory.get_preprocessing_for_model(
                model_name, fallback=cfg_preprocessing_name
            )

            # Preprocess (with caching per mode)
            if preprocessing_name not in preprocessing_cache_train:
                print(f"\n[Preprocessing] {preprocessing_name.upper()} (for model '{model_name}')")
                try:
                    t_prep = datetime.now()
                    preprocessing_cache_train[preprocessing_name] = preprocess_data(
                        X_train_raw, preprocessing_name
                    )
                    preprocessing_cache_test[preprocessing_name] = preprocess_data(
                        X_test_raw, preprocessing_name
                    )
                    prep_time = (datetime.now() - t_prep).total_seconds()
                    print(f"Preprocessing done ({prep_time:.2f}s)")
                except Exception as e:
                    print(f"[ERROR] Preprocessing failed: {str(e)}")
                    failed_experiments.append(
                        (dataset_name, model_name, "N/A", f"Preprocessing: {str(e)}")
                    )
                    continue
            else:
                print(f"\n[Preprocessing] {preprocessing_name.upper()} (cached)")

            X_train_preprocessed = preprocessing_cache_train[preprocessing_name]
            X_test_preprocessed = preprocessing_cache_test[preprocessing_name]

            for embedding_name in embeddings_to_use:
                experiment_count += 1

                exp_id = f"[{experiment_count}/{total_combinations}]"
                print(f"\n{exp_id} {dataset_name.upper()} | {model_name.upper()} | {embedding_name.upper()}")
                print("-" * 100)

                # Skip already-completed experiments (resume after Colab disconnect)
                experiment_key = f"{dataset_name}_{model_name}_{embedding_name}"
                if experiment_key in completed_keys:
                    print(f"  ⏭️ SKIP (already completed) — {experiment_key}")
                    continue

                is_transformer = ModelFactory.is_transformer_model(model_name)

                try:
                    # Validate compatibility
                    validate_compatibility(model_name, embedding_name, preprocessing_name)

                    # Embedding step (skipped for transformers) ──
                    embedder = None
                    X_train_vec = None
                    X_test_vec = None

                    if not is_transformer:
                        print(f"[Embedding] {embedding_name}")
                        t_emb_start = datetime.now()
                        X_train_vec, embedder = embed_data(
                            X_train_preprocessed, embedding_name, fit=True
                        )
                        X_test_vec, _ = embed_data(
                            X_test_preprocessed, embedding_name, fit=False, embedder=embedder
                        )
                        emb_time = (datetime.now() - t_emb_start).total_seconds()
                        print(f"Vectorized: {X_train_vec.shape} ({emb_time:.2f}s)")
                    else:
                        print(f"[Transformer] Tokenization handled internally by {model_name}")

                    run_name = (
                        f"{dataset_name}_{model_name}_{embedding_name}_{preprocessing_name}"
                    )

                    with mlflow.start_run(run_name=run_name, nested=True):
                        mlflow.set_tag("group", run_name)
                        mlflow.set_tag("model", model_name)
                        mlflow.set_tag("embedding", embedding_name)
                        mlflow.set_tag("dataset", dataset_name)
                        mlflow.set_tag("preprocessing", preprocessing_name)

                        run_id = mlflow.active_run().info.run_id

                        mlflow.log_params({
                            "dataset": dataset_name,
                            "model": model_name,
                            "embedding": embedding_name,
                            "preprocessing": preprocessing_name,
                            "n_trials": cfg.optuna.n_trials,
                        })

                        # Optuna optimization
                        print(f"[Optuna] Running {cfg.optuna.n_trials} trials...")

                        sampler = TPESampler(seed=cfg.seed)
                        try:
                            study = optuna.create_study(
                                sampler=sampler,
                                direction='maximize'
                            )
                        except Exception as e:
                            # DB is locked by the failed connection on Windows;
                            # fall back to in-memory storage instead of deleting
                            print(f"[Optuna] Storage error: {e}")
                            print(f"[Optuna] Falling back to in-memory storage "
                                  f"(results still saved via MLflow/CSV)")
                            study = optuna.create_study(
                                sampler=sampler,
                                direction='maximize',
                                study_name=run_name,
                            )

                        # Define objective (transformer vs classic) ──
                        if is_transformer:
                            def objective(trial):
                                return run_single_trial_transformer(
                                    trial=trial,
                                    cfg=cfg,
                                    X_train_texts=X_train_preprocessed,
                                    X_test_texts=X_test_preprocessed,
                                    y_train=y_train,
                                    y_test=y_test,
                                    model_name=model_name,
                                    embedding_name=embedding_name,
                                    dataset_name=dataset_name,
                                    preprocessing_name=preprocessing_name,
                                )
                        else:
                            def objective(trial):
                                return run_single_trial(
                                    trial=trial,
                                    cfg=cfg,
                                    X_train_vec=X_train_vec,
                                    X_test_vec=X_test_vec,
                                    y_train=y_train,
                                    y_test=y_test,
                                    model_name=model_name,
                                    embedding_name=embedding_name,
                                    dataset_name=dataset_name,
                                    preprocessing_name=preprocessing_name,
                                    embedding_object=embedder
                                )

                        study.optimize(
                            objective,
                            n_trials=cfg.optuna.n_trials,
                            timeout=cfg.optuna.get('timeout', None),
                            show_progress_bar=False
                        )

                        best_trial = study.best_trial
                        print(f"[Best Trial] #{best_trial.number} | F1: {best_trial.value:.4f}")

                        print(f"[Final Model] Training with best hyperparameters...")

                        model_config = OmegaConf.to_container(
                            cfg.models.get(model_name, {})
                        )
                        model_config.pop('optuna', None)
                        final_model_params = {**model_config, **best_trial.params}

                        final_model = ModelFactory.get_model(
                            dataset_name=dataset_name,
                            model_type=model_name,
                            embedding_type=embedding_name,
                            **final_model_params
                        )

                        # Train & evaluate final model ──
                        if is_transformer:
                            # Transformer: train and evaluate on raw text
                            t_train = datetime.now()
                            final_model.train(X_train_preprocessed, y_train)
                            train_time = (datetime.now() - t_train).total_seconds()

                            t_infer = datetime.now()
                            metrics = final_model.evaluate(X_test_preprocessed, y_test)
                            inference_time = (datetime.now() - t_infer).total_seconds()

                            # Save model
                            final_model.save()

                            # Get predictions
                            y_pred, y_probs = get_predictions_and_probs_transformer(
                                final_model, X_test_preprocessed
                            )
                        else:
                            # Classic ML: train and evaluate on vectors
                            final_model.embedder = copy.deepcopy(embedder)

                            t_train = datetime.now()
                            final_model.train_on_vectors(X_train_vec, y_train)
                            train_time = (datetime.now() - t_train).total_seconds()

                            t_infer = datetime.now()
                            metrics = final_model.evaluate_on_vectors(X_test_vec, y_test)
                            inference_time = (datetime.now() - t_infer).total_seconds()

                            # Save model
                            final_model.save()

                            # Get predictions
                            y_pred, y_probs = get_predictions_and_probs(
                                final_model, X_test_vec
                            )

                        # Log with hybrid tracking manager (CSV + MLflow)
                        tracking_mgr.log_experiment(
                            dataset=dataset_name,
                            model=model_name,
                            embedding=embedding_name,
                            preprocessing=preprocessing_name,
                            metrics=metrics,
                            hyperparams=best_trial.params,
                            predictions={
                                "text_id": range(len(y_test)),
                                "true_label": y_test,
                                "predicted_label": y_pred,
                                "probability": y_probs,
                            },
                            trial_history=study.trials_dataframe(),
                            train_time=train_time,
                            inference_time=inference_time,
                            best_trial=best_trial.number,
                            run_id=run_id
                        )

                        # Log to MLflow
                        mlflow.log_params(best_trial.params)
                        mlflow.log_metrics({
                            "final_accuracy": round(metrics["accuracy"], 4),
                            "final_precision": round(metrics["precision"], 4),
                            "final_recall": round(metrics["recall"], 4),
                            "final_f1_score": round(metrics["f1_score"], 4),
                            "train_time_sec": round(train_time, 2),
                            "inference_time_sec": round(inference_time, 4),
                        })

                        # Print results
                        print(f"[Results]")
                        print(f"F1 Score: {metrics['f1_score']:.4f}")
                        print(f"Accuracy: {metrics['accuracy']:.4f}")
                        print(f"Precision: {metrics['precision']:.4f}")
                        print(f"Recall: {metrics['recall']:.4f}")
                        print(f"Train time: {train_time:.2f}s | Inference: {inference_time:.4f}s")

                except Exception as e:
                    error_msg = str(e)
                    print(f"[ERROR]: {error_msg}")
                    traceback.print_exc()
                    failed_experiments.append((dataset_name, model_name, embedding_name, error_msg))
                    mlflow.set_tag("status", "FAILED")

    # Final summary of all experiments
    print_final_summary(tracking_mgr, failed_experiments, total_combinations)


def print_header(datasets, models, embeddings, cfg):
    """Print experiment header"""
    total = len(datasets) * len(models) * len(embeddings)
    print(f"\n{'='*100}")
    print(f"LARGE-SCALE EXPERIMENT RUNNER")
    print(f"{'='*100}")
    print(f"Datasets: {len(datasets)} -> {datasets}")
    print(f"Models: {len(models)} -> {models}")
    print(f"Embeddings: {len(embeddings)} -> {embeddings}")
    print(f"Preprocessing: {cfg.preprocessing.name}")
    print(f"Total combinations: {total}")
    print(f"Optuna trials per model: {cfg.optuna.n_trials}")
    print(f"MLflow experiment: {cfg.mlflow.experiment_name}")
    print(f"Results directory: {cfg.get('results_dir', './experiments')}")
    print(f"{'='*100}\n")


def print_final_summary(tracking_mgr, failed_experiments, total_combinations):
    """Print final comprehensive summary"""
    print(f"\n{'='*100}")
    print(f"FINAL SUMMARY")
    print(f"{'='*100}\n")

    # Save summary
    tracking_mgr.save_summary_csv()

    # Print statistics
    tracking_mgr.print_summary_stats()

    # Print failed experiments
    if failed_experiments:
        print(f"\n[FAILED EXPERIMENTS] ({len(failed_experiments)}/{total_combinations})")
        for dataset, model, embedding, error in failed_experiments:
            print(f" {dataset} | {model} | {embedding}: {error}")
    else:
        print(f"\nAll {total_combinations} experiments completed successfully!")

    print(f"\n[Results Location]")
    print(f"MLflow UI: mlflow ui --backend-store-uri '{tracking_mgr.local_storage}/mlruns'")
    print(f"Summary CSV: {tracking_mgr.metrics_dir / 'training_results.csv'}")
    print(f"All files: {tracking_mgr.local_storage}")
    print(f"\n{'='*100}\n")


if __name__ == "__main__":
    main()
