from pathlib import Path
from typing import List, Optional

import hydra
import pandas as pd
from omegaconf import DictConfig

from research.configs.models.model_factory import ModelFactory
from research.web.evaluator import WebTestEvaluator


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
            # ML models save classifier.joblib; transformers save a transformer/ subfolder
            has_classifier = (model_dir / "classifier.joblib").exists()
            has_transformer = (model_dir / "transformer").is_dir()
            if not (has_classifier or has_transformer):
                print(f"Skipping {model_dir.name} - missing classifier.joblib or transformer/")
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

        # Validate against ModelFactory registries
        valid_models = list(ModelFactory.COMPATIBILITY_MAP.keys())
        valid_embeddings = set()
        for embs in ModelFactory.COMPATIBILITY_MAP.values():
            valid_embeddings.update(embs)

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
            preprocessing_name: str = "classic",
            num_models: Optional[int] = None
    ):
        """Run tests on web data with filtering.
        preprocessing_name is used only as a fallback – the actual mode
        is resolved per model via ModelFactory.PREPROCESSING_MAP."""
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
            prep_mode = ModelFactory.get_preprocessing_for_model(
                m['model'], fallback=preprocessing_name
            )
            print(
                f"{i:3d}. {m['model']:8s} + {m['embedding']:10s} "
                f"(preprocessing: {prep_mode}) ({m['dir_name']})"
            )

        web_tester = WebTestEvaluator(
            preprocessor_name=preprocessing_name,
            results_dir=str(self.results_dir)
        )
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

                # Evaluate on web data (preprocessing resolved automatically per model)
                result = web_tester.evaluate_model_on_web_data(
                    model=model,
                    X_web=X_web,
                    y_web=y_web,
                    dataset_name=self.dataset_name,
                    model_name=model_info['model'],
                    embedding_name=model_info['embedding'],
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

        # Clear preprocessing cache after a full run
        web_tester.clear_cache()

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


def _run_evaluation_for_web_variant(
        cfg, datasets, models_list, embeddings_list, preprocessing_name,
        results_dir, web_data_path, text_type: str
):
    """Run evaluation on a single web-data variant (short or full_article).

    Returns a list of result DataFrames (one per dataset).
    """
    if not Path(web_data_path).exists():
        print(f"[Warning] Web data not found at {web_data_path}")
        print(f"Skipping web testing for text_type='{text_type}'")
        return []

    web_data = pd.read_csv(web_data_path)
    web_data = web_data.dropna(subset=['text', 'label'])
    print(f"Loaded {len(web_data)} valid samples for testing (text_type='{text_type}').\n")

    all_results = []
    for dataset_name in datasets:
        print("\n" + "=" * 100)
        print(f"TESTING DATASET: {dataset_name.upper()} | text_type: {text_type}")
        print("=" * 100)

        try:
            runner = WebTestRunner(
                saved_models_dir=None,
                dataset_name=dataset_name,
                results_dir=results_dir
            )
        except FileNotFoundError as e:
            print(f"{e}")
            continue

        print("[Finding and testing models]\n")
        web_results = runner.run_web_tests(
            web_data,
            models=models_list,
            embeddings=embeddings_list,
            preprocessing_name=preprocessing_name,
            num_models=None
        )

        if not web_results.empty:
            # Tag every result with the text type so short vs full can be distinguished
            web_results['text_type'] = text_type
            # Make experiment_key unique per text_type
            web_results['experiment_key'] = (
                web_results['experiment_key'] + f"_{text_type}"
            )
            all_results.append(web_results)

            # Print dataset summary
            print("\n" + "-" * 100)
            print(f"DATASET SUMMARY: {dataset_name.upper()} ({text_type})")
            print("-" * 100)
            print(f"Total models tested: {len(web_results)}")
            print(f"Average F1 Score: {web_results['f1_score'].mean():.4f}")
            print(f"Best F1 Score: {web_results['f1_score'].max():.4f}")
            print(f"Worst F1 Score: {web_results['f1_score'].min():.4f}\n")

            print("Top 5 Models:")
            print(web_results.nlargest(5, 'f1_score')[[
                'model', 'embedding', 'f1_score', 'accuracy'
            ]].to_string(index=False))

    return all_results


def _save_and_print_summary(all_results, results_dir):
    """Merge, upsert, save combined results and print summary."""
    if not all_results:
        return

    combined_results = pd.concat(all_results, ignore_index=True)
    combined_results = combined_results.sort_values('f1_score', ascending=False)

    # Keep only the required columns
    columns = [
        'experiment_key', 'dataset', 'model', 'embedding', 'preprocessing',
        'test_type', 'text_type', 'num_samples',
        'accuracy', 'precision', 'recall', 'f1_score',
        'timestamp', 'confusion_matrix',
    ]
    combined_results = combined_results[[c for c in columns if c in combined_results.columns]]

    # Upsert: overwrite existing rows for the same experiment_key, append new ones
    output_path = Path(results_dir) / "results.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        df_existing = pd.read_csv(output_path)
        df = pd.concat([df_existing, combined_results], ignore_index=True)
    else:
        df = combined_results

    df = df.drop_duplicates(subset='experiment_key', keep='last')
    df = df.sort_values('f1_score', ascending=False)
    df.to_csv(output_path, index=False)

    print(f"\n\nResults saved to {output_path}")

    # Print overall summary
    print("\n" + "=" * 100)
    print("OVERALL RESULTS SUMMARY")
    print("=" * 100)
    print(f"\nTotal models tested: {len(combined_results)}")
    print(f"Average F1 Score: {combined_results['f1_score'].mean():.4f}")
    print(f"Best F1 Score: {combined_results['f1_score'].max():.4f}")
    print(f"Worst F1 Score: {combined_results['f1_score'].min():.4f}\n")

    print("Top 10 Models (across all datasets):")
    print(combined_results.nlargest(10, 'f1_score')[[
        'dataset', 'model', 'embedding', 'text_type', 'f1_score', 'accuracy'
    ]].to_string(index=False))

    print("\nBottom 5 Models:")
    print(combined_results.nsmallest(5, 'f1_score')[[
        'dataset', 'model', 'embedding', 'text_type', 'f1_score', 'accuracy'
    ]].to_string(index=False))

    # Summary by dataset
    print("\n" + "-" * 100)
    print("SUMMARY BY DATASET")
    print("-" * 100)
    by_dataset = combined_results.groupby('dataset').agg({
        'f1_score': ['count', 'mean', 'max', 'min', 'std'],
        'accuracy': 'mean'
    }).round(4)
    print(by_dataset)

    # Summary by text_type (short vs full_article)
    if 'text_type' in combined_results.columns:
        print("\n" + "-" * 100)
        print("SUMMARY BY TEXT TYPE (short vs full_article)")
        print("-" * 100)
        by_text_type = combined_results.groupby('text_type').agg({
            'f1_score': ['count', 'mean', 'max', 'min', 'std'],
            'accuracy': 'mean'
        }).round(4)
        print(by_text_type)


def main_with_hydra(cfg: DictConfig):
    """Main function with Hydra configuration support"""
    print("\n" + "=" * 100)
    print("WEB DATA TESTING")
    print("=" * 100)

    # Get configuration values
    datasets = cfg.get('datasets_list', [cfg.datasets.name])
    models_list = cfg.get('models_to_optimize', [cfg.models.name])
    embeddings_list = cfg.get('embeddings_to_use', [cfg.embeddings.name])
    preprocessing_name = cfg.get('preprocessing', {}).get('name', 'classic')
    results_dir = cfg.get('results_dir', './experiments/web_test_results')

    full_article_filename = cfg.web_scraping.get(
        'full_article_filename', 'web_scraped_news_full.csv'
    )
    scrape_full_articles = cfg.web_scraping.get('scrape_full_articles', False)

    print(f"\nConfiguration loaded:")
    print(f"  Datasets: {datasets}")
    print(f"  Models: {models_list}")
    print(f"  Embeddings: {embeddings_list}")
    print(f"  Preprocessing: {preprocessing_name}")
    print(f"  Results dir: {results_dir}")
    print(f"  Full-article evaluation: {scrape_full_articles}\n")

    all_results = []

    # Evaluate on SHORT text (RSS descriptions)
    short_path = cfg.web_scraping.output_dir + "/web_scraped_news.csv"
    print("\n" + "#" * 100)
    print("  PHASE 1: SHORT TEXT (RSS descriptions)")
    print("#" * 100)
    short_results = _run_evaluation_for_web_variant(
        cfg, datasets, models_list, embeddings_list,
        preprocessing_name, results_dir, short_path,
        text_type="short"
    )
    all_results.extend(short_results)

    # Evaluate on FULL ARTICLES (if available)
    if scrape_full_articles:
        full_path = cfg.web_scraping.output_dir + "/" + full_article_filename
        print("\n" + "#" * 100)
        print("  PHASE 2: FULL ARTICLES")
        print("#" * 100)
        full_results = _run_evaluation_for_web_variant(
            cfg, datasets, models_list, embeddings_list,
            preprocessing_name, results_dir, full_path,
            text_type="full_article"
        )
        all_results.extend(full_results)

    # Save combined results
    _save_and_print_summary(all_results, results_dir)

    print("\n" + "=" * 100 + "\n")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """Main entry point with Hydra configuration"""
    main_with_hydra(cfg)


if __name__ == "__main__":
    main()
