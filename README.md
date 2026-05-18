# Fake News Detector & Research Framework

A comprehensive machine learning project for detecting fake news using various NLP techniques. The repository contains a **Research Framework** for benchmarking and optimising models across multiple datasets and a **Chrome Extension** powered by a FastAPI backend that detects fake news in real time.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [How it Works](#how-it-works)
3. [Architecture](#architecture)
4. [Project Structure](#project-structure)
5. [Prerequisites](#prerequisites)
6. [Installation & Setup](#installation--setup)
7. [Dataset Preprocessing](#dataset-preprocessing)
8. [Research Pipeline — Quick Start](#research-pipeline--quick-start)
9. [Google Colab](#google-colab)
10. [Experiment Outputs](#experiment-outputs)
11. [MLflow Tracking](#mlflow-tracking)
12. [Chrome Extension](#chrome-extension)
13. [Models & Embeddings Reference](#models--embeddings-reference)
14. [Smart Routing Logic](#smart-routing-logic)
15. [Datasets](#datasets)
16. [Docker](#docker)
17. [Contributing](#contributing)

---

## Project Overview

The project has two main components:

| Component | Description |
|---|---|
| **Research framework** | Hydra/Optuna training pipeline, web scraping, web testing, and result tracking |
| **Chrome extension** | Browser popup that sends selected text to the FastAPI backend and displays a real/fake verdict |

---

## How it Works

### Research pipeline

1. **Dataset preprocessing** — Raw datasets (ISOT, LIAR, WELFake) are processed into train/test/val splits using `research/preprocess_datasets.py`. This script:
   - Loads raw dataset files (CSV/TSV format)
   - Binarises labels according to dataset-specific schemes (e.g., LIAR's 6-point scale → 0/1)
   - Shuffles and stratifies splits into 80% train / 10% test / 10% validation
   - Saves clean CSV files with `text` and `label` columns to `research/configs/datasets/processed/`
   - Optionally cleans ISOT to remove Reuters leakage (via `clean_isot.py`)

2. **Dataset loading** — Three labelled news datasets (ISOT, LIAR, WELFake) are read from pre-split `train.csv` / `test.csv` / `val.csv` files. Each row contains a `text` column and a binary `label` (0 = fake, 1 = real).

3. **Preprocessing** — `TextPreprocessor` applies one of two strategies selected automatically from `ModelFactory.PREPROCESSING_MAP`:
   - `classic` — lowercase, strip URLs/HTML, remove punctuation and digits, collapse whitespace. Used for all classical ML and deep learning models.
   - `bert` — whitespace normalisation only. Used for HuggingFace transformer models to preserve casing and subword boundaries.

4. **Embedding** — For classical ML and deep learning models the preprocessed text is converted to dense vectors:
   - `tfidf` / `bow` — sparse count-based representations fit on the training split.
   - `word2vec` / `glove` — each document is represented as the average of its token embeddings.
   - Transformer and DL models tokenise text internally; no external embedder step is needed.

5. **Hyperparameter optimisation** — [Optuna](https://optuna.org/) runs `n_trials` experiments per model × embedding × dataset combination. A `TPESampler` proposes candidates and a `MedianPruner` kills unpromising transformer trials early. The best trial's hyperparameters are used to retrain the final model.

6. **Tracking** — `HybridTrackingManager` writes results simultaneously to:
   - A local `experiments/metrics/training_results.csv` (durable, Drive-backed in Colab).
   - An MLflow SQLite database for UI-based comparison across runs.
   - Per-experiment JSON (best hyperparameters), CSV (trial history), and prediction files.

7. **Serialisation** — The final model (`classifier.joblib`), embedder (`embedder.pkl`), and optional scaler (`scaler.joblib`) are saved to `saved_models/{dataset}/{model_name}/`.

### API prediction

1. The FastAPI server (`backend/api.py`) loads three specialised models at startup via the `lifespan` context manager.
2. `POST /predict` receives a JSON body `{"text": "..."}`.
3. A word-count heuristic selects the best model (see [Smart Routing Logic](#smart-routing-logic)).
4. The selected model's `predict()` method preprocesses the text, vectorises it, runs inference, and returns a `label`, `score`, and metadata dict.

### Chrome extension

1. The user highlights text on any webpage and clicks the extension icon.
2. `popup.js` reads the selection via `window.getSelection()` and POSTs it to `http://127.0.0.1:8000/predict`.
3. The verdict (`REAL` / `FAKE`) and confidence score are displayed with colour-coded styling.

---

## Architecture

```
Raw Datasets (ISOT, LIAR, WELFake)
        │
        ▼
Dataset Preprocessing (preprocess_datasets.py)
 (split into train/test/val)
        │
        ▼
Dataset loading (train.csv / test.csv / val.csv)
        │
        ▼
Text Preprocessing (TextPreprocessor)
        │
        ▼
Embedding (TF-IDF / BoW / Word2Vec / GloVe / Transformers)
        │
        ▼
Model training & Optuna optimisation (train_models.py)
        │
        ▼
Evaluation & result tracking (tracking_manager.py, evaluate_web.py)
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
│   ├── preprocess_datasets.py        # Dataset preprocessing (train/test/val splits)
│   ├── train_models.py               # Hydra/Optuna training pipeline (main entry point)
│   ├── evaluate_web.py               # CLI: test trained models on web-scraped data
│   ├── analyze_results.py            # Analysis & visualisation of training vs web results
│   ├── clean_isot.py                 # ISOT Reuters leakage removal
│   ├── configs/
│   │   ├── config.yaml               # Hydra config (datasets, models, embeddings, Optuna, MLflow)
│   │   ├── embeddings/               # Embedding implementations (TF-IDF, BoW, Word2Vec, GloVe)
│   │   ├── models/                   # Model wrappers & factory (SVM, LR, RF, XGB, …)
│   │   ├── preprocessing/            # Text preprocessing (TextPreprocessor)
│   │   └── datasets/processed/       # Preprocessed datasets (train/test/val splits)
│   ├── tracking/
│   │   └── tracking_manager.py       # Hybrid CSV + MLflow tracking manager
│   ├── web/
│   │   ├── collect_web.py            # CLI: collect/simulate web-scraped articles
│   │   ├── web_scraper.py            # News source definitions & RSS collector
│   │   └── evaluator.py              # WebTestEvaluator class
│   ├── colab/                        # Google Colab integration
│   │   ├── colab_config.py           # Colab-specific configuration
│   │   └── colab_secrets.json        # Colab secrets (GitHub token, paths)
│   └── data/                         # Raw datasets (gitignored)
├── experiments/                      # Generated output directory (gitignored data, skeleton tracked)
│   ├── metrics/                      # training_results.csv from training pipeline
│   ├── predictions/                  # Per-experiment prediction CSVs
│   ├── web_test_results/             # Web testing CSVs and reports
│   ├── hyperparams/                  # Best Optuna hyperparameters (JSON)
│   ├── trial_history/                # Optuna trial history CSVs
│   └── results/                      # Visualisation outputs
├── saved_models/                     # Serialised models (.joblib, .pkl) — gitignored
├── requirements.txt
├── README.md
├── docker-compose.yml
├── Dockerfile                        # API container
├── Dockerfile.research               # Training/research container
└── fake_news_detector.ipynb          # Google Colab notebook
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

4. **Download raw datasets** (optional)

   If you plan to preprocess datasets yourself, download them and place them in `research/` directory:

   | Dataset | Download link | Expected location |
   |---|---|---|
   | ISOT | [https://www.uvic.ca/engineering/ece/isot/datasets/](https://www.uvic.ca/engineering/ece/isot/datasets/) | `research/ISOT/` |
   | LIAR | [https://www.cs.ucsb.edu/~william/data/liar_dataset.zip](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip) | `research/LIAR/` |
   | WELFake | [https://zenodo.org/record/4561253](https://zenodo.org/record/4561253) | `research/WELFake/` |

5. **Preprocess datasets** (see [Dataset Preprocessing](#dataset-preprocessing) section below)

   Or manually place pre-split datasets:

   | Dataset | Expected paths |
   |---|---|
   | ISOT | `research/configs/datasets/processed/isot/train.csv`, `test.csv`, `val.csv` |
   | LIAR | `research/configs/datasets/processed/liar/train.csv`, `test.csv`, `val.csv` |
   | WELFake | `research/configs/datasets/processed/welfake/train.csv`, `test.csv`, `val.csv` |

   Each CSV must have at least two columns: `text` and `label` (0 = fake, 1 = real).


---

## Dataset Preprocessing

The `research/preprocess_datasets.py` script automates the preparation of raw datasets into train/test/val splits suitable for the training pipeline.

### Overview

This script:
- **Normalises labels** according to dataset-specific schemes (e.g., LIAR's 6-point Likert scale → binary 0/1)
- **Splits data** into 80% training, 10% test, 10% validation using stratified sampling (preserves class balance)
- **Validates quality** by removing rows with missing text fields
- **Cleans ISOT** to remove Reuters leakage (articles present in both real and fake subsets)
- **Saves normalised CSVs** with consistent `text` and `label` columns

### Prerequisites

All raw datasets must be placed in the `research/` directory before running preprocessing:

```
research/
├── ISOT/
│   ├── True.csv
│   └── Fake.csv
├── LIAR/
│   ├── train.tsv
│   ├── test.tsv
│   └── valid.tsv
└── WELFake/
    └── data.csv
```
### Running the preprocessor

```bash
python -m research.preprocess_datasets
```

### Output

Processed datasets are saved to `research/configs/datasets/processed/{dataset}/`:

```
research/configs/datasets/processed/
├── isot/
│   ├── train.csv
│   ├── test.csv
│   └── val.csv
├── liar/
│   ├── train.csv
│   ├── test.csv
│   └── val.csv
└── welfake/
    ├── train.csv
    ├── test.csv
    └── val.csv
```

Each file contains:
- `text` — News article or claim text
- `label` — Binary label (1 = fake, 0 = real)

### Dataset-specific processing

#### ISOT
- Combines `True.csv` and `Fake.csv` (True → label 0, Fake → label 1)
- Stratified 80/10/10 split
- Additional cleaning via `clean_isot.py` removes Reuters articles that appear in both subsets

#### LIAR
- TSV format with predefined train/test/valid splits (original split structure is preserved)
- Label mapping: `{pants-fire, false, barely-true}` → 1 (fake), `{half-true, mostly-true, true}` → 0 (real)
- Removes statements not in the target label set

#### WELFake
- Combines four external datasets for diversity
- Stratified 80/10/10 split
- Already contains `text` and `label` columns; no additional mapping needed

### Example output

```
ISOT
  train.csv: 28800 rows
  test.csv: 3600 rows
  val.csv: 3600 rows
LIAR
  train.csv: 8555 rows
  test.csv: 2703 rows
  val.csv: 1255 rows
WELFake
  train.csv: 57600 rows
  test.csv: 7200 rows
  val.csv: 7200 rows
All done!
```

---

## Research Pipeline — Quick Start

### Step 0: Preprocess datasets (first time only)

If you haven't already, preprocess your raw datasets into train/test/val splits:

```bash
python research/preprocess_datasets.py
```

See [Dataset Preprocessing](#dataset-preprocessing) for detailed instructions.

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

## Google Colab

The repository includes `fake_news_detector.ipynb` for running the full training pipeline on Google Colab with automatic Google Drive persistence. This is the recommended approach for training transformer and deep learning models, which benefit from GPU acceleration.

### First-time setup

1. Upload your datasets and pre-trained word vectors to `My Drive/fake-news-results/`:
   ```
   fake-news-results/
   └── datasets/
       ├── processed/      ← ISOT/, LIAR/, WELFake/ (train.csv, test.csv)
       └── embeddings/     ← glove.6B.100d.txt, GoogleNews-vectors-negative300.bin
   ```
2. Copy `colab_secrets.json.template` to `colab_secrets.json` and fill in your GitHub personal access token (already in `.gitignore`).

### Notebook cells

| Cell | Description |
|---|---|
| 0 — Configuration | Loads `colab_secrets.json` for repo URL and Drive paths |
| 1 — Mount Drive | `drive.mount("/content/drive")` |
| 2 — Clone / pull repo | Clones on first run; pulls latest changes on subsequent runs |
| 3 — Symlink setup | Links Drive folders into the project via `colab_setup.py` |
| 4 — Install deps | `pip install -r requirements.txt` |
| 5 — Preprocess datasets | `python research/preprocess_datasets.py` (first time only) |
| 6 — Train models | `python -m research.train_models` (all Hydra overrides apply) |
| 7 — Collect web data | `python -m research.web.collect_web` |
| 8 — Evaluate on web | `python -m research.evaluate_web` |
| 9 — Analyse results | `python -m research.analyze_results` |

All outputs (models, experiments, MLflow DB) are written back to Drive, so interrupted sessions resume from where they left off.

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

### Classical ML models

| Key | Algorithm |
|---|---|
| `svm` | Support Vector Machine (LinearSVC + calibrated probabilities) |
| `lr` | Logistic Regression |
| `rf` | Random Forest |
| `dt` | Decision Tree |
| `xgb` | XGBoost |
| `nb` | Naive Bayes (Gaussian) |
| `mnb` | Multinomial Naive Bayes |
| `knn` | K-Nearest Neighbours |

### Deep learning models

| Key | Architecture |
|---|---|
| `lstm` | LSTM (unidirectional) |
| `gru` | GRU |
| `bilstm` | Bidirectional LSTM |
| `cnn` | 1-D Convolutional Neural Network |

Deep learning models receive raw (pre-cleaned) text; word vectors are loaded internally.

### Transformer models

| Key | Pretrained checkpoint |
|---|---|
| `bert` | `bert-base-uncased` |
| `roberta` | `roberta-base` |
| `distilbert` | `distilbert-base-uncased` |
| `fakebert` | `bert-base-uncased` (domain-tuned variant) |

Transformer models use light preprocessing (whitespace normalisation only) and tokenise text internally via HuggingFace Tokenizers.

### Supported embeddings

| Key | Method | Used with |
|---|---|---|
| `tfidf` | TF-IDF | Classical ML |
| `bow` | Bag of Words | Classical ML |
| `word2vec` | Word2Vec (average pooling) | Classical ML, DL |
| `glove` | GloVe (average pooling) | Classical ML, DL |
| `bert-base-uncased` | BERT tokeniser | `bert`, `fakebert` |
| `roberta-base` | RoBERTa tokeniser | `roberta` |
| `distilbert-base-uncased` | DistilBERT tokeniser | `distilbert` |

Not all model × embedding combinations are valid. The `ModelFactory` validates compatibility automatically and raises an error on mismatches.

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
