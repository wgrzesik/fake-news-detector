# Architecture

This document describes the main components of the Fake News Detector project and the flow of data from research datasets to the Chrome extension demo.

## System Overview

The project has three main layers:

1. Research pipeline: prepares datasets, trains models, evaluates them, and stores metrics and model artifacts.
2. FastAPI backend: loads selected trained models and exposes prediction endpoints.
3. Chrome extension: reads selected webpage text and displays the prediction returned by the backend.

```text
Raw Datasets (ISOT, LIAR, WELFake)
        │
        ▼
Dataset Preprocessing (research/preprocess_datasets.py)
        │
        ▼
Processed train/test/validation splits
        │
        ▼
Text Preprocessing (TextPreprocessor)
        │
        ▼
Embedding or tokenizer stage
        │
        ▼
Model training and Optuna optimization (research/train_models.py)
        │
        ▼
Evaluation and tracking (tracking_manager.py, evaluate_web.py)
        │
        ▼
saved_models/
        │
        ▼
FastAPI backend (backend/api.py)
        │
        ▼
Chrome extension (extension/)
```

## Research Pipeline

### 1. Dataset Preprocessing

Raw datasets are processed into train, test, and validation splits using `research/preprocess_datasets.py`.

The preprocessing script:

- loads raw dataset files in CSV or TSV format,
- normalizes labels according to dataset-specific rules,
- validates rows and removes entries with missing text,
- creates clean files with consistent `text` and `label` columns,
- saves processed files under `research/configs/datasets/processed/`,

Expected raw dataset layout:

```text
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

Run preprocessing from the repository root:

```bash
python -m research.preprocess_datasets
```

Processed output layout:

```text
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

Each processed file contains:

- `text`: news article or claim text,
- `label`: binary class label, where `0` means fake and `1` means real.

### 2. Dataset-Specific Processing

#### ISOT

- Combines `True.csv` and `Fake.csv`.
- Maps true articles to label `1` and fake articles to label `0`.
- Uses a stratified train/test/validation split.
- Includes additional cleaning for Reuters leakage when enabled.

#### LIAR

- Uses TSV files with predefined train, test, and validation splits.
- Maps `{pants-fire, false, barely-true}` to fake.
- Maps `{half-true, mostly-true, true}` to real.
- Removes statements outside the supported target label set.

#### WELFake

- Uses a combined fake-news dataset built from multiple sources.
- Already provides text and label columns.
- Uses a stratified train/test/validation split.

### 3. Dataset Loading

The training pipeline reads ISOT, LIAR, and WELFake from the processed split directories. Each dataset is loaded from `train.csv`, `test.csv`, and `val.csv`.

### 4. Text Preprocessing

`TextPreprocessor` applies preprocessing strategies selected through the model configuration.

Main strategies:

- `classic`: lowercasing, URL/HTML removal, punctuation and digit removal, whitespace normalization. Used for classical ML and deep learning models.
- `bert`: light whitespace normalization. Used for transformer models to preserve text structure needed by tokenizers.

### 5. Embedding and Tokenization

Classical ML and selected deep learning models use vector representations:

- `tfidf` and `bow`: sparse count-based representations fitted on the training split,
- `word2vec` and `glove`: document vectors based on averaged token embeddings.

Transformer models tokenize text internally through HuggingFace tokenizers. Some deep learning models also operate on token sequences internally, so they do not always need an external embedder in the same way as classical ML models.

### 6. Model Training and Hyperparameter Optimization

`research/train_models.py` runs the main training workflow. It uses Hydra configuration and Optuna optimization over dataset, model, and embedding combinations.

Optuna:

- runs `n_trials` experiments per configured combination,
- uses a `TPESampler` to propose hyperparameters,
- can prune weak transformer trials early through a `MedianPruner`,
- stores the best hyperparameters for final model training.

Important configuration is stored in `research/configs/config.yaml`.

### 7. Tracking and Experiment Outputs

`HybridTrackingManager` writes results to several places:

- `experiments/metrics/training_results.csv`: main training metrics table,
- MLflow SQLite tracking database: UI-based run comparison,
- per-experiment JSON files: best hyperparameters,
- trial history CSV files,
- prediction files and evaluation artifacts.

Web evaluation is handled by `research/evaluate_web.py`, and result analysis is handled by `research/analyze_results.py`.

### 8. Model Serialization

Trained artifacts are saved under `saved_models/{dataset}/{model_name}/`.

Typical files include:

- `classifier.joblib`: trained classifier,
- `embedder.pkl`: fitted embedder,
- `scaler.joblib`: optional scaler, when required by the model.

## Web-Scraped Data and Result Analysis

The project includes a separate web evaluation path for checking how trained models behave on news collected outside the original benchmark datasets. This is important because a model can perform well on held-out dataset splits but generalize poorly to current web text.

### Web Data Collection

Web data collection starts in `research/web/collect_web.py`.

The collector:

- reads scraping configuration from `research/configs/config.yaml`,
- uses source definitions from `research/web/sources_config.py`,
- collects real and fake news samples through `WebNewsCollector`,
- removes duplicate URLs,
- shuffles the final dataset,
- optionally fetches full article text,
- saves one unified CSV through `collector.save_scraped_data()`.

When full-article scraping is enabled, the collected dataset keeps multiple text variants:

- `text`: the full article text,
- `short_text`: the shorter RSS summary or original feed text.
- `title`: the article title.

Run collection with:

```bash
python -m research.web.collect_web
```

The default output location is controlled by the `web_scraping.output_dir` set in `research/configs/config.yaml`.

### Web Evaluation

`research/evaluate_web.py` loads saved models from `saved_models/` and evaluates them on web-scraped data.

The web evaluation supports multiple text variants:

- `title`: article title,
- `text`: full article body,
- `short_text`: RSS summary or shorter article description.

This allows the project to compare whether models work better on short claims, summaries, or full articles.

Typical output files are written to `experiments/web_test_results/`:

```text
experiments/web_test_results/
├── results_title.csv
├── results_text.csv
└── results_short_text.csv
```

Each result file uses the same core metrics as the training pipeline: accuracy, precision, recall, F1 score, inference time, and confusion matrix where available.

### Analysis with `analyze_results.py`

`research/analyze_results.py` compares training metrics against web evaluation metrics.

Input files:

- `experiments/metrics/training_results.csv`,
- `experiments/web_test_results/results_title.csv`,
- `experiments/web_test_results/results_text.csv`,
- `experiments/web_test_results/results_short_text.csv`.

The script performs five main steps:

1. Loads and cleans training results.
2. Loads web results for each supported text type.
3. Creates per-dataset and global comparisons between training and web metrics.
4. Creates a training-only ranking.
5. Creates a cross-text-type comparison when at least two web result files are available.

The comparison logic merges results by:

- `model`,
- `embedding`,
- `dataset`.

It then computes metric deltas, for example `delta_f1_score = f1_score_train - f1_score_web`. Smaller drops indicate better generalization from dataset evaluation to web data.

Main outputs are saved under `experiments/results/`:

```text
experiments/results/
├── title/
│   ├── isot/
│   ├── liar/
│   ├── welfake/
│   └── all/
├── text/
│   ├── isot/
│   ├── liar/
│   ├── welfake/
│   └── all/
├── short_text/
│   ├── isot/
│   ├── liar/
│   ├── welfake/
│   └── all/
├── training/
└── cross_text_type/
```

Generated artifacts include:

- ranking CSV and TXT files,
- train-vs-web comparison plots,
- generalization scatter plots,
- confusion matrix heatmaps,
- summary files,
- F1 comparison files,
- cross-text-type comparison charts and summaries.

Run analysis with:

```bash
python -m research.analyze_results
```

## API Prediction Layer

The API is implemented in `backend/api.py` with FastAPI.

At startup, the FastAPI lifespan handler loads the configured models into `app.state.models`. The current demo configuration loads three routes when matching model artifacts are available:

| Route key | Dataset | Model | Embedding |
|---|---|---|---|
| `short_text` | WELFake | XGBoost | TF-IDF |
| `general` | ISOT | Random Forest | Bag of Words |
| `long_article` | LIAR | BiLSTM | GloVe |

Main endpoints:

- `GET /health`: reports whether models are loaded and lists loaded route keys.
- `POST /predict`: accepts `{"text": "..."}` and returns a prediction.

Prediction flow:

1. The request body is validated by `TextRequest`.
2. Empty or very short text is rejected or returned as `NEUTRAL`.
3. `select_best_model()` chooses a model using word-count routing.
4. The selected model runs `predict()`.
5. The response includes the model output plus metadata.

Response metadata:

```json
{
  "meta": {
    "used_dataset": "ISOT",
    "used_model": "rf",
    "word_count": 42
  }
}
```

## Smart Routing Logic

The API selects the best available model based on the number of words in the input text.

| Word count | Route key | Dataset | Model | Embedding |
|---|---|---|---|---|
| Less than 30 words | `short_text` | WELFake | XGBoost | TF-IDF |
| 30 to 100 words | `general` | ISOT | Random Forest | Bag of Words |
| More than 100 words | `long_article` | LIAR | BiLSTM | GloVe |

If the preferred route is unavailable, the API falls back to `general` or to the first loaded model.

## Chrome Extension Layer

The extension is located in `extension/`.

Main files:

- `manifest.json`: Chrome extension manifest,
- `popup.html`: popup structure,
- `styles.css`: popup styling,
- `config.js`: API URL and UI message constants,
- `api.js`: backend request helper,
- `ui.js`: DOM update and result rendering helpers,
- `popup.js`: selection reading and popup interaction flow.

User flow:

1. The user highlights text on a webpage.
2. The user opens the extension popup and clicks `Analyze Selection`.
3. `popup.js` reads the selected text using `window.getSelection()`.
4. `api.js` sends the text to `http://127.0.0.1:8000/predict`.
5. `ui.js` renders the returned `REAL`, `FAKE`, or `NEUTRAL` label, confidence score, and model metadata.

## Research Workflow Commands

Run the full research workflow after preprocessing:

```bash
python -m research.train_models
python -m research.web.collect_web
python -m research.evaluate_web
python -m research.analyze_results
```

Common outputs:

| Output | Location |
|---|---|
| Training metrics | `experiments/metrics/training_results.csv` |
| Web evaluation results | `experiments/web_test_results/results_title.csv`, `results_text.csv`, `results_short_text.csv` |
| Analysis plots and summaries | `experiments/results/{text_type}/{dataset}/`, `experiments/results/{text_type}/all/` |
| Cross-text-type analysis | `experiments/results/cross_text_type/` |
| Best hyperparameters | `experiments/hyperparams/` |
| Trial histories | `experiments/trial_history/` |
| Serialized models | `saved_models/` |

## MLflow Tracking

The training pipeline can log runs to MLflow. Start the UI with:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open `http://localhost:5000`.

## Google Colab Workflow

The repository includes `fake_news_detector.ipynb` for running the training workflow on Google Colab with Google Drive persistence. This is useful for long-running or GPU-backed experiments.

Typical notebook flow:

1. Mount Google Drive.
2. Clone or update the repository.
3. Link Drive folders for datasets, models, and experiment outputs.
4. Install dependencies.
5. Run preprocessing, training, web evaluation, and result analysis.

Outputs are written back to Drive so interrupted Colab sessions can be resumed.

## Docker Architecture

The Docker setup contains:

| File | Purpose |
|---|---|
| `Dockerfile` | FastAPI prediction backend |
| `Dockerfile.research` | Research, training, MLflow, and evaluation environment |
| `docker-compose.yml` | Service orchestration for API, MLflow, and training |

Main Docker services:

- `api`: prediction backend on port `8000`,
- `mlflow`: tracking UI on port `5000`,
- `train`: on-demand training service enabled by the `training` profile.
