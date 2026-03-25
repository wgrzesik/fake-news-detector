import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd
from omegaconf import OmegaConf

from research.configs.models.model_factory import ModelFactory
from research.web_scrapper import create_simulated_web_data, WebNewsCollector
from research.web_testing import WebTestEvaluator


class WebTestRunner:
    """Run comprehensive web testing"""

    def __init__(
            self,
            saved_models_dir: str = None,
            dataset_name: str = "ISOT",
            results_dir: str = "./experiments/web_test_results"
    ):
        # If no directory specified, find it relative to research directory
        if saved_models_dir is None:
            script_dir = Path(__file__).parent
            saved_models_dir = script_dir.parent / "saved_models"
        else:
            saved_models_dir = Path(saved_models_dir)

        self.saved_models_root = saved_models_dir.resolve()
        self.dataset_name = dataset_name
        self.saved_models_dir = self.saved_models_root / dataset_name
        self.results_dir = Path(results_dir).resolve()
        self.results_dir.mkdir(parents=True, exist_ok=True)

        if not self.saved_models_dir.exists():
            print(f"Dataset directory not found: {self.saved_models_dir}")
            raise FileNotFoundError(f"Directory not found: {self.saved_models_dir}")

    def find_saved_models(self, models: Optional[List[str]] = None, embeddings: Optional[List[str]] = None):
        """
        Find saved models with optional filtering

        Directory structure:
        saved_models/
        └── ISOT/
            ├── rf_tfidf/
            │   ├── classifier.joblib
            │   └── embedder.pkl
            ├── svm_tfidf/
            └── xgb_bow/
        """
        found_models = []

        if not self.saved_models_dir.exists():
            print(f"Directory not found: {self.saved_models_dir}")
            return found_models

        print(f"\nSearching for models in: {self.saved_models_dir}")

        for model_dir in self.saved_models_dir.iterdir():
            if not model_dir.is_dir():
                continue

            # Check if it's a complete model directory
            if not (model_dir / "classifier.joblib").exists():
                print(f"Skipping {model_dir.name} - missing classifier.joblib")
                continue

            # Extract model name and embedding from directory name
            dir_name = model_dir.name
            model_info = self._parse_model_dir_name(dir_name)

            if not model_info:
                print(f"Could not parse model directory: {dir_name}")
                continue

            model_name = model_info['model']
            embedding_name = model_info['embedding']

            # Apply filters
            if models and model_name not in models:
                continue

            if embeddings and embedding_name not in embeddings:
                continue

            found_models.append({
                'dataset': self.dataset_name,
                'model': model_name,
                'embedding': embedding_name,
                'dir_name': dir_name,
                'path': str(model_dir)
            })

        return found_models

    @staticmethod
    def _parse_model_dir_name(dir_name: str):
        """
        Parse model directory name to extract model and embedding
        Format: {model}_{embedding}
        Examples: rf_tfidf, svm_bow, xgb_word2vec
        """
        parts = dir_name.rsplit('_', 1)

        if len(parts) != 2:
            return None

        model_name = parts[0]
        embedding_name = parts[1]

        # Validate
        valid_models = [
            'svm', 'lr', 'nb', 'mnb', 'knn', 'rf', 'dt', 'xgb', 'sgd', 'ridge'
        ]
        valid_embeddings = ['tfidf', 'bow', 'word2vec', 'glove']

        if model_name not in valid_models:
            return None

        if embedding_name not in valid_embeddings:
            return None

        return {
            'model': model_name,
            'embedding': embedding_name
        }

    def run_web_tests(
            self,
            web_data: pd.DataFrame,
            models: Optional[List[str]] = None,
            embeddings: Optional[List[str]] = None,
            num_models: Optional[int] = None
    ):
        """Run tests on web data with filtering"""
        found_models = self.find_saved_models(
            models=models,
            embeddings=embeddings
        )

        if len(found_models) == 0:
            print("No models found matching criteria")
            return pd.DataFrame()

        if num_models:
            found_models = found_models[:num_models]

        print(f"\n{'=' * 100}")
        print(f"[Web Testing] Found {len(found_models)} models to test")
        print(f"{'=' * 100}")

        print("\nModels to test:")
        for i, m in enumerate(found_models, 1):
            print(
                f"{i:3d}. {m['model']:8s} + {m['embedding']:10s} "
                f"({m['dir_name']})"
            )

        web_tester = WebTestEvaluator(results_dir=str(self.results_dir))
        X_web = web_data['text'].tolist()
        y_web = web_data['label'].tolist()

        results = []
        failed = []

        for idx, model_info in enumerate(found_models, 1):
            exp_key = f"{model_info['model']}_{model_info['embedding']}"
            print(
                f"\n[{idx}/{len(found_models)}] Testing {exp_key}..."
            )

            try:
                # Load model from saved directory
                model = ModelFactory.get_model(
                    dataset_name=self.dataset_name,
                    model_type=model_info['model'],
                    embedding_type=model_info['embedding']
                )
                model.load()

                # Evaluate on web data
                result = web_tester.evaluate_model_on_web_data(
                    model=model,
                    X_web=X_web,
                    y_web=y_web,
                    dataset_name=self.dataset_name,
                    model_name=model_info['model'],
                    embedding_name=model_info['embedding']
                )

                if result:
                    results.append(result)
                    print(
                        f"F1: {result['f1_score']:.4f} | "
                        f"Acc: {result['accuracy']:.4f} | "
                        f"Precision: {result['precision']:.4f} | "
                        f"Recall: {result['recall']:.4f}"
                    )

            except Exception as e:
                print(f"Failed: {str(e)}")
                failed.append(exp_key)
                import traceback
                traceback.print_exc()

        if failed:
            print(f"\n{len(failed)} models failed: {', '.join(failed)}")

        return pd.DataFrame(results)

    def generate_report(
            self,
            web_results: pd.DataFrame,
            output_path: str = "web_test_report.txt"
    ):
        """Generate comprehensive report"""
        if web_results.empty:
            print("No results to report")
            return

        with open(output_path, 'w') as f:
            f.write("=" * 100 + "\n")
            f.write("WEB DATA TESTING REPORT\n")
            f.write(f"Dataset: {self.dataset_name}\n")
            f.write("=" * 100 + "\n\n")

            f.write("SUMMARY\n")
            f.write("-" * 100 + "\n")
            f.write(f"Total models tested: {len(web_results)}\n")
            f.write(f"Test samples: {web_results['num_samples'].iloc[0] if len(web_results) > 0 else 0}\n")
            f.write(f"Average F1 Score: {web_results['f1_score'].mean():.4f}\n")
            f.write(f"Std F1 Score: {web_results['f1_score'].std():.4f}\n")
            f.write(f"Best F1 Score: {web_results['f1_score'].max():.4f}\n")
            f.write(f"Worst F1 Score: {web_results['f1_score'].min():.4f}\n\n")

            f.write("WEB TEST RESULTS (sorted by F1 score)\n")
            f.write("-" * 100 + "\n")
            sorted_results = web_results.sort_values('f1_score', ascending=False)
            f.write(sorted_results[[
                'model', 'embedding', 'f1_score', 'accuracy', 'precision', 'recall'
            ]].to_string(index=False))

            f.write("\n\n")
            f.write("BEST PERFORMING MODELS\n")
            f.write("-" * 100 + "\n")
            best = web_results.nlargest(10, 'f1_score')
            f.write(best[[
                'model', 'embedding', 'f1_score', 'accuracy'
            ]].to_string(index=False))

            f.write("\n\n")
            f.write("WORST PERFORMING MODELS\n")
            f.write("-" * 100 + "\n")
            worst = web_results.nsmallest(10, 'f1_score')
            f.write(worst[[
                'model', 'embedding', 'f1_score', 'accuracy'
            ]].to_string(index=False))

            f.write("\n\n")
            f.write("BY MODEL\n")
            f.write("-" * 100 + "\n")
            by_model = web_results.groupby('model').agg({
                'f1_score': ['count', 'mean', 'max', 'min', 'std'],
                'accuracy': 'mean'
            }).round(4)
            f.write(str(by_model))

            f.write("\n\n")
            f.write("BY EMBEDDING\n")
            f.write("-" * 100 + "\n")
            by_embedding = web_results.groupby('embedding').agg({
                'f1_score': ['count', 'mean', 'max', 'min', 'std'],
                'accuracy': 'mean'
            }).round(4)
            f.write(str(by_embedding))

        print(f"Report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Test trained models on web-scraped data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Test all models for default dataset (ISOT)
  python -m research.test_on_web_data

  # Test specific models only
  python -m research.test_on_web_data --models svm xgb rf

  # Test specific embeddings
  python -m research.test_on_web_data --embeddings tfidf

  # Test specific model + embedding combinations
  python -m research.test_on_web_data --models svm xgb --embeddings tfidf

  # Limit to first 5 models
  python -m research.test_on_web_data --num-models 5

  # Custom dataset
  python -m research.test_on_web_data --dataset ISOT
        """
    )

    parser.add_argument(
        '--models-dir',
        default=None,
        help='Saved models root directory (default: auto-detect)'
    )

    parser.add_argument(
        '--results-dir',
        default='./experiments/web_test_results',
        help='Results directory (default: ./experiments/web_test_results)'
    )

    parser.add_argument(
        '--dataset',
        default='ISOT',
        help='Dataset name (default: ISOT)'
    )

    parser.add_argument(
        '--models',
        nargs='+',
        help='Filter by model names (e.g., svm xgb rf dt)'
    )

    parser.add_argument(
        '--embeddings',
        nargs='+',
        help='Filter by embedding names (e.g., tfidf bow word2vec)'
    )

    parser.add_argument(
        '--num-models',
        type=int,
        default=None,
        help='Limit number of models to test'
    )

    parser.add_argument(
        '--report',
        default='web_test_report.txt',
        help='Report output path (default: web_test_report.txt)'
    )

    args = parser.parse_args()

    print("\n" + "=" * 100)
    print("WEB DATA TESTING")
    print("=" * 100)

    # Run tests
    try:
        runner = WebTestRunner(
            saved_models_dir=args.models_dir,
            dataset_name=args.dataset,
            results_dir=args.results_dir
        )
    except FileNotFoundError as e:
        print(f"{e}")
        return

    cfg = OmegaConf.load("research/configs/config.yaml")
    web_data = pd.read_csv(cfg.web_scraping.output_dir + "/web_scraped_news.csv")
    web_data = web_data.dropna(subset=['text', 'label'])
    # web_data['text'] = web_data['text'].astype(str)
    # web_data['label'] = web_data['label'].astype(int)
    print(f"Loaded {len(web_data)} valid samples for testing.")

    print("[Finding and testing models]\n")
    web_results = runner.run_web_tests(
        web_data,
        models=args.models,
        embeddings=args.embeddings,
        num_models=args.num_models
    )

    if web_results.empty:
        print("No results to save")
        return

    # Save results
    output_path = Path(args.results_dir) / "web_test_results.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    web_results.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

    # Print summary
    print("\n" + "=" * 100)
    print("RESULTS SUMMARY")
    print("=" * 100)
    print(f"\nTotal models tested: {len(web_results)}")
    print(f"Average F1 Score: {web_results['f1_score'].mean():.4f}")
    print(f"Best F1 Score: {web_results['f1_score'].max():.4f}")
    print(f"Worst F1 Score: {web_results['f1_score'].min():.4f}\n")

    print("Top 10 Models:")
    print(web_results.nlargest(10, 'f1_score')[[
        'model', 'embedding', 'f1_score', 'accuracy'
    ]].to_string(index=False))

    print("\nBottom 5 Models:")
    print(web_results.nsmallest(5, 'f1_score')[[
        'model', 'embedding', 'f1_score', 'accuracy'
    ]].to_string(index=False))

    # Generate report
    print(f"\n[Generating report]")
    runner.generate_report(web_results, output_path=args.report)

    print("\n" + "=" * 100 + "\n")


if __name__ == "__main__":
    main()
