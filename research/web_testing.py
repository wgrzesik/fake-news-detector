import os
from datetime import datetime
from typing import Dict, List

import pandas as pd

from research.configs.models.model_factory import ModelFactory
from research.configs.preprocessing.preprocessor import TextPreprocessor


class WebTestEvaluator:
    """
    Evaluate models trained on standard datasets against web-scraped data
    """

    def __init__(
        self,
        preprocessor_name: str = "classic",
        results_dir: str = "./experiments/web_test_results"
    ):
        self.preprocessor_name = preprocessor_name
        self.results_dir = os.path.abspath(results_dir)
        os.makedirs(self.results_dir, exist_ok=True)

        self.preprocessing = TextPreprocessor(mode=preprocessor_name)
        self.results = []

    def evaluate_model_on_web_data(
        self,
        model,
        X_web: List[str],
        y_web: List[int],
        dataset_name: str,
        model_name: str,
        embedding_name: str,
        run_id: str = None
    ):
        """
        Evaluate a trained model on web-scraped data
        """
        print(f"\n[Web Testing] {model_name} + {embedding_name} on {dataset_name}")

        try:
            # Preprocess
            X_web_proc = self.preprocessing.transform(X_web)

            # Embed using model's embedder
            X_web_vec = model.embedder.transform(X_web_proc)

            # Evaluate
            metrics = model.evaluate_on_vectors(X_web_vec, y_web)

            # Get predictions
            if model.scaler:
                X_web_scaled = model.scaler.transform(X_web_vec)
            else:
                X_web_scaled = X_web_vec

            y_pred = model.classifier.predict(X_web_scaled)
            y_probs = model.predict_proba_on_vectors(X_web_vec)

            results = {
                'dataset': dataset_name,
                'model': model_name,
                'embedding': embedding_name,
                'test_type': 'web_scraped',
                'num_samples': len(y_web),
                'accuracy': round(metrics['accuracy'], 4),
                'precision': round(metrics['precision'], 4),
                'recall': round(metrics['recall'], 4),
                'f1_score': round(metrics['f1_score'], 4),
                'run_id': run_id,
                'timestamp': datetime.now().isoformat(),
            }

            return results

        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
            return None

    def evaluate_multiple_models(
        self,
        models_config: List[Dict],
        X_web: List[str],
        y_web: List[int],
        dataset_name: str,
        embedding_name: str,
        run_id: str = None
    ):
        """
        Evaluate multiple model configurations on web data
        """
        results = []

        for config in models_config:
            model_name = config['model_name']

            try:
                # Load model
                model = ModelFactory.get_model(
                    dataset_name=dataset_name,
                    model_type=model_name,
                    embedding_type=embedding_name
                )
                model.load()

                # Evaluate
                result = self.evaluate_model_on_web_data(
                    model=model,
                    X_web=X_web,
                    y_web=y_web,
                    dataset_name=dataset_name,
                    model_name=model_name,
                    embedding_name=embedding_name,
                    run_id=run_id
                )

                if result:
                    results.append(result)

            except Exception as e:
                print(f"Error evaluating {model_name} on web data: {str(e)}")

        return pd.DataFrame(results)

    def save_web_test_results(self, results_df: pd.DataFrame, filename: str = "web_test_results.csv"):
        """Save web test results"""
        filepath = os.path.join(self.results_dir, filename)
        file_exists = os.path.exists(filepath)
        results_df.to_csv(
            filepath,
            mode='a',
            header=not file_exists,
            index=False,
        )
        print(f"Saved to {filepath}")
        return filepath

    @staticmethod
    def compare_train_vs_web(train_results: pd.DataFrame, web_results: pd.DataFrame):
        """
        Compare model performance on training data vs web data
        """
        print("\n[Comparing Train vs Web Performance]")

        comparison = []

        for model_name in train_results['model'].unique():
            for embedding_name in train_results['embedding'].unique():
                train_row = train_results[
                    (train_results['model'] == model_name) &
                    (train_results['embedding'] == embedding_name)
                ]

                web_row = web_results[
                    (web_results['model'] == model_name) &
                    (web_results['embedding'] == embedding_name)
                ]

                if len(train_row) > 0 and len(web_row) > 0:
                    train_f1 = train_row['f1_score'].values[0]
                    web_f1 = web_row['f1_score'].values[0]
                    f1_drop = train_f1 - web_f1

                    comparison.append({
                        'model': model_name,
                        'embedding': embedding_name,
                        'train_f1': train_f1,
                        'web_f1': web_f1,
                        'f1_drop': round(f1_drop, 4),
                        'drop_percentage': round(f1_drop / train_f1 * 100, 2) if train_f1 > 0 else 0
                    })

        comparison_df = pd.DataFrame(comparison)
        comparison_df = comparison_df.sort_values('f1_drop', ascending=False)

        print("\n" + "="*100)
        print("TRAIN vs WEB DATA COMPARISON")
        print("="*100)
        print(comparison_df.to_string(index=False))
        print("="*100 + "\n")

        return comparison_df

    def generate_web_test_report(self, train_results: pd.DataFrame, web_results: pd.DataFrame, output_path: str = "web_test_report.txt"):
        """Generate comprehensive web testing report"""
        comparison = self.compare_train_vs_web(train_results, web_results)

        with open(output_path, 'w') as f:
            f.write("="*100 + "\n")
            f.write("WEB DATA GENERALIZATION REPORT\n")
            f.write("="*100 + "\n\n")

            f.write("SUMMARY\n")
            f.write("-"*100 + "\n")
            f.write(f"Train models tested on web-scraped data\n")
            f.write(f"Web samples: {len(web_results)}\n")
            f.write(f"Average F1 drop: {comparison['f1_drop'].mean():.4f}\n\n")

            f.write("DETAILED COMPARISON\n")
            f.write("-"*100 + "\n")
            f.write(comparison.to_string(index=False))
            f.write("\n\n")

            f.write("MODELS WITH WORST GENERALIZATION\n")
            f.write("-"*100 + "\n")
            worst = comparison.nlargest(5, 'f1_drop')
            f.write(worst.to_string(index=False))
            f.write("\n\n")

            f.write("MODELS WITH BEST GENERALIZATION\n")
            f.write("-"*100 + "\n")
            best = comparison.nsmallest(5, 'f1_drop')
            f.write(best.to_string(index=False))

        print(f"Report saved to {output_path}")