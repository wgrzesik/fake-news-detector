# Fake News Detector & Research Framework

A comprehensive machine learning project for detecting fake news using various NLP techniques. The repository contains a **Research Framework** for benchmarking and optimising models across multiple datasets and a **Chrome Extension** powered by a FastAPI backend that detects fake news in real time.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Project Structure](#project-structure)
4. [Prerequisites](#prerequisites)
5. [Installation & Setup](#installation--setup)
6. [Research Pipeline — Quick Start](#research-pipeline--quick-start)
7. [Experiment Outputs](#experiment-outputs)
8. [MLflow Tracking](#mlflow-tracking)
9. [Chrome Extension](#chrome-extension)
10. [Models & Embeddings Reference](#models--embeddings-reference)
11. [Smart Routing Logic](#smart-routing-logic)
12. [Datasets](#datasets)
13. [Contributing](#contributing)

---

## Project Overview

The project has two main components:

| Component | Description |
|---|---|
| **Research framework** | Two training pipelines (simple benchmark + Hydra/Optuna), web scraping, web testing, and result tracking |
| **Chrome extension** | Browser popup that sends selected text to the FastAPI backend and displays a real/fake verdict |

---

## Architecture

```
Web scraping (collect_web_data.py)
        │
        ▼
  Raw text data
        │
        ▼
Preprocessing (preprocess.py / TextPreprocessor)
        │
        ▼
Embedding (TF-IDF / BoW / Word2Vec / GloVe)
        │
        ▼
Model training & Optuna optimisation (train_all_models_final.py)
        │
        ▼
Evaluation & result tracking (result_manager.py / tracking_manager.py)
        │
        ▼
 saved_models/   ←→   FastAPI backend (api.py)
                              │
                              ▼
                    Chrome Extension (extension/)
```

---

## Project Structure

```text
fake-news-detector/
├── backend/
│   └── api.py                        # FastAPI server with smart model routing
├── extension/                        # Chrome Extension
│   ├── manifest.json
│   ├── popup.html
│   ├── popup.js
│   └── styles.css
├── research/
│   ├── configs/
│   │   ├── config.yaml               # Hydra config (datasets, models, embeddings, Optuna, MLflow)
│   │   ├── embeddings/               # Embedding config classes
│   │   ├── models/                   # Model config classes & factory
│   │   └── preprocessing/            # Preprocessor config
│   ├── embeddings/                   # Embedding implementations (TF-IDF, BoW, Word2Vec, GloVe)
│   ├── models/                       # Model wrappers (SVM, LR, RF, XGB, …)
│   ├── analyze_experiments.py        # Post-run analysis utilities
│   ├── base.py                       # Abstract base classes
│   ├── collect_web_data.py           # Web scraper for real news articles
│   ├── factory.py                    # Simple model factory (used by main.py)
│   ├── main.py                       # Simple benchmark pipeline
│   ├── preprocess.py                 # Text preprocessing utilities
│   ├── result_manager.py             # Predictions & metrics storage
│   ├── test_on_web_data.py           # CLI: test trained models on web-scraped data
│   ├── tracking_manager.py           # Hybrid CSV + MLflow tracking manager
│   ├── train_all_models_final.py     # Full Hydra/Optuna training pipeline
│   ├── web_scrapper.py               # News collection helpers
│   └── web_testing.py                # WebTestEvaluator class
├── experiments/                      # Generated output directory (gitignored data, skeleton tracked)
│   ├── results/                      # benchmark_*.csv from main.py
│   ├── metrics/                      # SUMMARY.csv from full pipeline
│   ├── predictions/                  # Per-experiment prediction CSVs
│   ├── web_test_results/             # Web testing CSVs and reports
│   ├── hyperparams/                  # Best Optuna hyperparameters (JSON)
│   ├── trial_history/                # Optuna trial history CSVs
│   └── plots/                        # Visualisation outputs
├── saved_models/                     # Serialised models (.joblib, .pkl) — gitignored
├── requirements.txt
└── README.md
```

---

## Prerequisites

- Python **3.9+**
- `pip`
- Google Chrome (for the extension)

---

## Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/wgrzesik/fake-news-detector.git
   cd fake-news-detector
   ```

2. **(Recommended) Create a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare datasets**

   Datasets are not included in the repository due to their size. Download them and place the processed splits under `research/data/processed/`:

   | Dataset | Expected path |
   |---|---|
   | ISOT | `research/data/processed/ISOT/train.csv`, `test.csv` |
   | LIAR | `research/data/processed/LIAR/train.csv`, `test.csv` |
   | WELFake | `research/data/processed/WELFake/train.csv`, `test.csv` |

   Each CSV must have at least two columns: `text` and `label` (0 = real, 1 = fake).

   Download links:
   - **ISOT**: [https://www.uvic.ca/engineering/ece/isot/datasets/](https://www.uvic.ca/engineering/ece/isot/datasets/)
   - **LIAR**: [https://www.cs.ucsb.edu/~william/data/liar_dataset.zip](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip)
   - **WELFake**: [https://zenodo.org/record/4561253](https://zenodo.org/record/4561253)

---

## Research Pipeline — Quick Start

### Simple benchmark (`research/main.py`)

Trains every model × embedding combination without hyperparameter optimisation:

```bash
python -m research.main
```

Results are saved to `experiments/results/benchmark_{dataset}.csv`.

### Full Hydra/Optuna pipeline (`research/train_all_models_final.py`)

Runs Optuna hyperparameter search for every dataset × model × embedding combination and tracks results with MLflow:

```bash
python -m research.train_all_models_final
```

Key configuration options are in `research/configs/config.yaml`:

| Key | Default | Description |
|---|---|---|
| `datasets_list` | `[ISOT, liar, welfake]` | Datasets to iterate over |
| `models_to_optimize` | all 8 | Models to run |
| `embeddings_to_use` | all 4 | Embeddings to run |
| `optuna.n_trials` | `10` | Optuna trials per experiment |
| `preprocessing.name` | `classic` | Preprocessing strategy |
| `mlflow.experiment_name` | `fake_news_detection` | MLflow experiment name |

Results are saved to `experiments/metrics/SUMMARY.csv`.

### Collect web data (`research/collect_web_data.py`)

Scrapes real news articles from the web:

```bash
python -m research.collect_web_data
```

Scraped data is saved to the directory configured in `config.yaml` under `web_scraping.output_dir`.

### Test on web data (`research/test_on_web_data.py`)

Tests all saved models against web-scraped data:

```bash
# Test all models for the default dataset (ISOT)
python -m research.test_on_web_data

# Test specific models only
python -m research.test_on_web_data --models svm xgb rf

# Test specific embeddings
python -m research.test_on_web_data --embeddings tfidf

# Limit to the first 5 models found
python -m research.test_on_web_data --num-models 5

# Custom dataset
python -m research.test_on_web_data --dataset LIAR

# Custom results directory
python -m research.test_on_web_data --results-dir ./my_results
```

| Flag | Default | Description |
|---|---|---|
| `--dataset` | `ISOT` | Dataset whose trained models to load |
| `--models` | all | Filter by model name(s) |
| `--embeddings` | all | Filter by embedding name(s) |
| `--num-models` | unlimited | Limit number of models tested |
| `--models-dir` | auto-detect | Override saved models root directory |
| `--results-dir` | `./experiments/web_test_results` | Where to write result CSVs |
| `--report` | `web_test_report.txt` | Report output path |

---

## Experiment Outputs

### Unified result schema

Both pipelines write results using the same column schema:

| Column | Description |
|---|---|
| `dataset` | Dataset name (ISOT, LIAR, WELFake) |
| `model` | Model key (svm, lr, rf, …) |
| `embedding` | Embedding key (tfidf, bow, word2vec, glove) |
| `preprocessing` | Preprocessing strategy (`none` for `main.py`, `classic` for full pipeline) |
| `accuracy` | Test accuracy |
| `precision` | Weighted precision |
| `recall` | Weighted recall |
| `f1_score` | Weighted F1 score |
| `train_time_sec` | Model training time in seconds |
| `inference_time_sec` | Evaluation/inference time in seconds |
| `timestamp` | ISO-8601 timestamp |

The full pipeline (`train_all_models_final.py`) additionally includes: `experiment_key`, `best_trial`, `run_id`, `confusion_matrix`.

Web test results (`test_on_web_data.py`) additionally include: `test_type`, `num_samples`, `run_id`.

### Output file locations

| File | Created by |
|---|---|
| `experiments/results/benchmark_{dataset}.csv` | `research/main.py` |
| `experiments/metrics/SUMMARY.csv` | `research/train_all_models_final.py` |
| `experiments/predictions/{dataset}/*.csv` | Full pipeline (per-experiment predictions) |
| `experiments/web_test_results/web_test_results.csv` | `research/test_on_web_data.py` |
| `experiments/hyperparams/{dataset}/*.json` | Full pipeline (best Optuna params) |
| `experiments/trial_history/{dataset}/*.csv` | Full pipeline (Optuna trial history) |

---

## MLflow Tracking

The full pipeline logs all runs to MLflow. To view the UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open [http://localhost:5000](http://localhost:5000) in your browser.

The UI shows per-run metrics, hyperparameters, tags, and a comparison view across all experiments.

---

## Chrome Extension

### 1. Start the backend API

```bash
python backend/api.py
```

The server starts on `http://127.0.0.1:8000`. It loads three pre-trained models on startup.

### 2. Load the extension in Chrome

1. Open Chrome and go to `chrome://extensions/`
2. Enable **Developer mode** (top-right toggle)
3. Click **Load unpacked**
4. Select the `extension/` folder from this project

### 3. Usage

1. Navigate to any news website
2. Select (highlight) a paragraph or headline
3. Click the **Fake News Detector** icon in the Chrome toolbar
4. Click **ANALYZE SELECTION**

---

## Models & Embeddings Reference

### Supported models

| Key | Algorithm |
|---|---|
| `svm` | Support Vector Machine |
| `lr` | Logistic Regression |
| `rf` | Random Forest |
| `dt` | Decision Tree |
| `xgb` | XGBoost |
| `nb` | Naive Bayes (Gaussian) |
| `mnb` | Multinomial Naive Bayes |
| `knn` | K-Nearest Neighbours |

### Supported embeddings

| Key | Method |
|---|---|
| `tfidf` | TF-IDF |
| `bow` | Bag of Words |
| `word2vec` | Word2Vec (average pooling) |
| `glove` | GloVe (average pooling) |

Not all model × embedding combinations are valid. The factory validates compatibility automatically.

---

## Smart Routing Logic

The API selects the best available model based on the word count of the input text:

| Word count | Model used |
|---|---|
| < 30 words | `short_text` model (LR on LIAR — optimised for short claims) |
| > 100 words | `long_article` model (SVM on ISOT — optimised for full articles) |
| 30–100 words | `general` model (XGBoost on WELFake — general purpose) |

If the preferred model is not loaded, the API falls back to `general` or the first available model.

---

## Datasets

| Dataset | Description | Download |
|---|---|---|
| **ISOT** | ~44 000 real and fake news articles from Reuters (real) and various unreliable sites (fake) | [UVic ISOT](https://www.uvic.ca/engineering/ece/isot/datasets/) |
| **LIAR** | ~12 800 short political statements labelled by PolitiFact into 6 truthfulness categories (binarised for this project) | [UCSB LIAR](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip) |
| **WELFake** | ~72 000 news articles combining four existing datasets for improved diversity | [Zenodo WELFake](https://zenodo.org/record/4561253) |

---

## Contributing

1. Fork the repository and create a feature branch from `develop`
2. Follow the existing code style (factory pattern, typed hints, consistent CSV schema)
3. Add or update tests if applicable
4. Open a pull request targeting the `develop` branch
