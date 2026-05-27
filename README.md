# Fake News Detector

## Project Goal

Fake News Detector is an NLP and machine learning project for classifying news text as `REAL` or `FAKE`.

This repository contains a fake-news detection project with two main parts:
- A research pipeline for training and evaluating NLP/ML models on ISOT, LIAR, and WELFake datasets.
- A Chrome extension connected to a FastAPI backend for real-time prediction on selected webpage text.

The main deliverables are:
- trained models (`saved_models/`),
- benchmark and analysis outputs (`experiments/`, `results_*.csv`),
- a working demo path: backend API + browser extension.

## Requirements

### Software
- Python 3.10+ (recommended 3.11)
- pip
- Node.js 18+ and npm (for extension tests)
- Google Chrome (for extension demo)
- Optional: Docker Desktop (for containerized run)

### Hardware
- Demo/API only: typical laptop, at least 4 GB RAM
- Full training: higher RAM and significantly more time/storage
- Optional GPU recommended for deep learning/transformer experiments

## Repository Structure

```text
fake-news-detector/
├── backend/                          # FastAPI server with smart model routing
│   └── api.py                        
├── extension/                        # Chrome Extension
│   ├── manifest.json
│   ├── popup.html
│   ├── popup.js
│   ├── api.js
│   ├── ui.js
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
├── tests/                            
│   ├── backend/                      # Tests for backend
│   ├── extension/                    # Jest tests for extension modules
├── saved_models/                     # Serialised models (.joblib, .pkl) — gitignored
├── requirements.txt
├── README.md
├── docker-compose.yml
├── Dockerfile                        # API container
├── Dockerfile.research               # Training/research container
└── fake_news_detector.ipynb          # Google Colab notebook
```

## Installation and Configuration

Clone the repository:

```bash
git clone https://github.com/wgrzesik/fake-news-detector.git
cd fake-news-detector
```

Create and activate a Python virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Install JavaScript test dependencies:

```bash
cd tests/extension
npm install
cd ../..
```

## Running the Demonstration

The recommended evaluation scenario is the local backend plus Chrome extension.

Start the backend from the repository root:

```bash
python backend/api.py
```

The API loads two active routing targets and one fallback model when matching artifacts are available:

- `short_text`: ISOT + Random Forest + Bag of Words, used for texts up to and including 100 words
- `long_article`: LIAR + BiLSTM + GloVe, used for texts over 100 words

Then load the extension in Chrome:

1. Open `chrome://extensions/`.
2. Enable `Developer mode`.
3. Click `Load unpacked`.
4. Select the `extension/` directory.
5. Open a webpage, highlight at least 10 characters, open the extension, and click `Analyze Selection`.


## Expected Result

When the demo is running correctly:

- the API is available at `http://127.0.0.1:8000`,
- Swagger documentation is available at `http://127.0.0.1:8000/docs`,
- `GET /health` returns loaded model keys or `degraded` if no model artifacts were loaded,
- the extension shows a prediction label, confidence score, and model metadata,
- too-short selections return a neutral message instead of calling a model.

Example API request:

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"text\":\"This is a sample news text long enough for prediction.\"}"
```

Expected response shape:

```json
{
  "label": "REAL",
  "score": 0.88,
  "meta": {
    "used_dataset": "ISOT",
    "used_model": "rf",
    "word_count": 11
  }
}
```

## Data

The full raw datasets are not bundled in the repository because of size and licensing constraints.

Datasets used by the project:

- ISOT Fake News Dataset: `https://www.kaggle.com/datasets/rmisra/news-category-dataset`
- LIAR dataset: `https://www.kaggle.com/datasets/doanquanvietnamca/liar-dataset`
- WELFake dataset: `https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification`

Place raw datasets in:

```text
research/ISOT/
research/LIAR/
research/WELFake/
```

Processed datasets are expected in:

```text
research/configs/datasets/processed/isot/
research/configs/datasets/processed/liar/
research/configs/datasets/processed/welfake/
```

Run preprocessing from the repository root:

```bash
python -m research.preprocess_datasets
```

## Reproduction and Verification of Results

To reproduce the main research workflow, run:

```bash
python -m research.train_models
python -m research.web.collect_web
python -m research.evaluate_web
python -m research.analyze_results
```

Generated artifacts include:

- training metrics in `experiments/metrics/training_results.csv`,
- web evaluation results in `experiments/web_test_results/results.csv`,
- plots, summaries, and comparison outputs in `experiments/results/`,
- trained models in `saved_models/`,
- MLflow data in `mlruns/` and `mlflow.db` when tracking is enabled.

Full reproduction can take a long time and may require significant compute resources. For evaluation, the shorter verification path is to inspect saved artifacts and run the backend plus extension demo.

MLflow UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open `http://localhost:5000` to inspect recorded runs.

## Google Colab

The recommended way to execute the full research pipeline is through `fake_news_detector.ipynb` in Google Colab. This avoids depending on a local GPU or long-running local machine, and uses Google Drive to persist datasets, processed splits, trained models, MLflow data, and experiment outputs between Colab sessions.

The notebook walks through the complete research workflow:

1. Configuration
2. Mount Google Drive
3. Clone or update repository
4. Install dependencies
5. Create Drive folders
6. Upload raw datasets and embeddings
7. Validate and preprocess datasets
8. Configure project symlinks (Drive ↔ project)
9. Train models
10. Collect web data
11. Clean web data
12. Evaluate on web data
13. Analyze results

## Tests

Backend tests:

```bash
python -m pytest tests/backend
```

Extension tests:

```bash
cd tests/extension
npm test
```

Current automated test coverage checks:

- backend health, prediction routing, validation, and error paths,
- extension API request handling,
- extension popup flow,
- extension result rendering and error display.

## Docker

Run the API:

```bash
docker compose up api
```

Run the API and MLflow UI:

```bash
docker compose up api mlflow
```

Run the training service:

```bash
docker compose --profile training run train
```

## Documentation

Project documentation is split into:

- User documentation: this `README.md`,
- Architecture  documentation: `docs/architecture.md`.
- Maintenance documentation: `docs/maintenance.md`.

## Limitations

- Full model training is resource-intensive and may require GPU acceleration for deep learning and transformer models.
- Raw datasets are external and require manual download.
- The Chrome extension expects the backend at `http://127.0.0.1:8000`.
- Prediction quality depends on the available trained model artifacts.
- If model artifacts are missing, the API health endpoint reports a degraded state and prediction requests return `503`.

## Demo Video

A short screen recording of the Chrome extension working.

![Demo video](demo.gif)
