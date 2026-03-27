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
13. [Docker](#docker)
14. [Contributing](#contributing)

---

## Project Overview

The project has two main components:

| Component | Description |
|---|---|
| **Research framework** | Hydra/Optuna training pipeline, web scraping, web testing, and result tracking |
| **Chrome extension** | Browser popup that sends selected text to the FastAPI backend and displays a real/fake verdict |

---

## Architecture

```
Web scraping (collect_web.py)
        │
        ▼
  Raw text data
        │
        ▼
Preprocessing (TextPreprocessor)
        │
        ▼
Embedding (TF-IDF / BoW / Word2Vec / GloVe)
        │
        ▼
Model training & Optuna optimisation (train_models.py)
        │
        ▼
Evaluation & result tracking (tracking_manager.py)
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
│   │   ├── embeddings/               # Embedding implementations (TF-IDF, BoW, Word2Vec, GloVe)
│   │   ├── models/                   # Model wrappers & factory (SVM, LR, RF, XGB, …)
│   │   └── preprocessing/            # Text preprocessing (TextPreprocessor)
│   ├── tracking/
│   │   └── tracking_manager.py       # Hybrid CSV + MLflow tracking manager
│   ├── web/
│   │   ├── collect_web.py            # CLI: collect/simulate web-scraped articles
│   │   ├── web_scraper.py            # News source definitions & RSS collector
│   │   └── evaluator.py              # WebTestEvaluator class
│   ├── train_models.py               # Hydra/Optuna training pipeline (main entry point)
│   ├── evaluate_web.py               # CLI: test trained models on web-scraped data
│   └── analyze_results.py            # Analysis & visualisation of training vs web results
├── experiments/                      # Generated output directory (gitignored data, skeleton tracked)
│   ├── metrics/                      # training_results.csv from training pipeline
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

   Datasets are not included in the repository due to their size. Download them and place the processed splits under `research/configs/datasets/processed/`:

   | Dataset | Expected path |
   |---|---|
   | ISOT | `research/configs/datasets/processed/isot/train.csv`, `test.csv` |
   | LIAR | `research/configs/datasets/processed/liar/train.csv`, `test.csv` |
   | WELFake | `research/configs/datasets/processed/welfake/train.csv`, `test.csv` |

   Each CSV must have at least two columns: `text` and `label` (0 = fake, 1 = real).

   Download links:
   - **ISOT**: [https://www.uvic.ca/engineering/ece/isot/datasets/](https://www.uvic.ca/engineering/ece/isot/datasets/)
   - **LIAR**: [https://www.cs.ucsb.edu/~william/data/liar_dataset.zip](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip)
   - **WELFake**: [https://zenodo.org/record/4561253](https://zenodo.org/record/4561253)

---

## Research Pipeline — Quick Start

### Hydra/Optuna pipeline (`research/train_models.py`)

Runs Optuna hyperparameter search for every dataset × model × embedding combination and tracks results with MLflow:

```bash
python -m research.train_models
```

Key configuration options are in `research/configs/config.yaml`:

| Key | Default | Description |
|---|---|---|
| `datasets_list` | `[isot, liar, welfake]` | Datasets to iterate over |
| `models_to_optimize` | all 8 | Models to run |
| `embeddings_to_use` | all 4 | Embeddings to run |
| `optuna.n_trials` | `1` | Optuna trials per experiment |
| `preprocessing.name` | `classic` | Preprocessing strategy |
| `mlflow.experiment_name` | `fake_news_detection` | MLflow experiment name |

Results are saved to `experiments/metrics/training_results.csv`.

### Collect web data (`research/web/collect_web.py`)

Scrapes real news articles from the web:

```bash
python -m research.web.collect_web
```

Scraped data is saved to the directory configured in `config.yaml` under `web_scraping.output_dir`.

### Evaluate on web data (`research/evaluate_web.py`)

Tests all saved models against web-scraped data:

```bash
# Test all models using default config
python -m research.evaluate_web

# Test specific models only (Hydra override)
python -m research.evaluate_web models_to_optimize=[svm,xgb,rf]

# Test specific embeddings
python -m research.evaluate_web embeddings_to_use=[tfidf]

# Custom dataset list
python -m research.evaluate_web datasets_list=[liar]
```

Configuration is managed via Hydra overrides on `research/configs/config.yaml`:

| Override | Default | Description |
|---|---|---|
| `datasets_list` | `[isot, liar, welfake]` | Datasets whose trained models to load |
| `models_to_optimize` | all 8 | Filter by model name(s) |
| `embeddings_to_use` | all 4 | Filter by embedding name(s) |
| `web_testing.results_dir` | `./experiments/web_test_results` | Where to write result CSVs |

### Analyse results (`research/analyze_results.py`)

Compares training results with web evaluation results, generates per-dataset and global rankings, delta metrics and visualisations:

```bash
python -m research.analyze_results
```

| Output | Location |
|---|---|
| Per-dataset ranking & charts | `experiments/results/{dataset}/` |
| Global ranking & charts | `experiments/results/all/` |
| Cross-comparison summary | `experiments/results/{dataset}/summary.csv` |

---

## Experiment Outputs

### Unified result schema

The training pipeline writes results using the following column schema:

| Column | Description |
|---|---|
| `dataset` | Dataset name (ISOT, LIAR, WELFake) |
| `model` | Model key (svm, lr, rf, …) |
| `embedding` | Embedding key (tfidf, bow, word2vec, glove) |
| `preprocessing` | Preprocessing strategy (`classic` for the training pipeline) |
| `accuracy` | Test accuracy |
| `precision` | Weighted precision |
| `recall` | Weighted recall |
| `f1_score` | Weighted F1 score |
| `train_time_sec` | Model training time in seconds |
| `inference_time_sec` | Evaluation/inference time in seconds |
| `timestamp` | ISO-8601 timestamp |

The pipeline additionally includes: `experiment_key`, `best_trial`, `run_id`, `confusion_matrix`.

Web test results (`evaluate_web.py`) additionally include: `test_type`, `num_samples`.

### Output file locations

| File | Created by |
|---|---|
| `experiments/metrics/training_results.csv` | `research/train_models.py` |
| `experiments/predictions/{dataset}/*.csv` | Training pipeline (per-experiment predictions) |
| `experiments/web_test_results/results.csv` | `research/evaluate_web.py` |
| `experiments/results/{dataset}/` | `research/analyze_results.py` |
| `experiments/results/all/` | `research/analyze_results.py` |
| `experiments/hyperparams/{dataset}/*.json` | Training pipeline (best Optuna params) |
| `experiments/trial_history/{dataset}/*.csv` | Training pipeline (Optuna trial history) |

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

Not all model × embedding combinations are valid. The `ModelFactory` validates compatibility automatically.

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

## Docker

The project includes Docker support for reproducible deployment and experimentation.

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running

### Files overview

| File | Purpose |
|---|---|
| `Dockerfile` | FastAPI prediction API (includes only the 3 models the API uses) |
| `Dockerfile.research` | Training pipeline, MLflow UI, web evaluation |
| `docker-compose.yml` | Orchestrates all services |
| `.dockerignore` | Keeps images small by excluding data, models, and caches |

### Quick start — API only

```bash
# Build and start the prediction API
docker-compose up api

# API is now available at http://localhost:8000
# Swagger docs at      http://localhost:8000/docs
```

### Start API + MLflow UI

```bash
docker-compose up api mlflow

# API    → http://localhost:8000
# MLflow → http://localhost:5000
```

### Run training pipeline

```bash
# Make sure datasets are in research/configs/datasets/processed/
docker-compose --profile training run train
```

Training results, models, and databases are written back to your host machine via Docker volumes.

### Rebuild after code changes

```bash
docker-compose build api          # rebuild API image only
docker-compose build              # rebuild all images
docker-compose up --build api     # rebuild + start in one step
```

### Stop everything

```bash
docker-compose down
```

---

## Contributing

1. Fork the repository and create a feature branch from `develop`
2. Follow the existing code style (typed hints, consistent CSV schema)
3. Add or update tests if applicable
4. Open a pull request targeting the `develop` branch
