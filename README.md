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
â”śâ”€â”€ backend/                          # FastAPI server with smart model routing
â”‚   â””â”€â”€ api.py                        
â”śâ”€â”€ extension/                        # Chrome Extension
â”‚   â”śâ”€â”€ manifest.json
â”‚   â”śâ”€â”€ popup.html
â”‚   â”śâ”€â”€ popup.js
â”‚   â”śâ”€â”€ api.js
â”‚   â”śâ”€â”€ ui.js
â”‚   â”śâ”€â”€ popup.js
â”‚   â””â”€â”€ styles.css
â”śâ”€â”€ research/
â”‚   â”śâ”€â”€ preprocess_datasets.py        # Dataset preprocessing, splits, and ISOT Reuters cleanup
â”‚   â”śâ”€â”€ train_models.py               # Hydra/Optuna training pipeline (main entry point)
â”‚   â”śâ”€â”€ evaluate_web.py               # CLI: test trained models on web-scraped data
â”‚   â”śâ”€â”€ analyze_results.py            # Analysis & visualisation of training vs web results
â”‚   â”śâ”€â”€ configs/
â”‚   â”‚   â”śâ”€â”€ config.yaml               # Hydra config (datasets, models, embeddings, Optuna, MLflow)
â”‚   â”‚   â”śâ”€â”€ embeddings/               # Embedding implementations (TF-IDF, BoW, Word2Vec, GloVe)
â”‚   â”‚   â”śâ”€â”€ models/                   # Model wrappers & factory (SVM, LR, RF, XGB, â€¦)
â”‚   â”‚   â”śâ”€â”€ preprocessing/            # Text preprocessing (TextPreprocessor)
â”‚   â”‚   â””â”€â”€ datasets/processed/       # Preprocessed datasets (train/test/val splits)
â”‚   â”śâ”€â”€ tracking/
â”‚   â”‚   â””â”€â”€ tracking_manager.py       # Hybrid CSV + MLflow tracking manager
â”‚   â”śâ”€â”€ web/
â”‚   â”‚   â”śâ”€â”€ collect_web.py            # CLI: collect/simulate web-scraped articles
â”‚   â”‚   â”śâ”€â”€ web_scraper.py            # News source definitions & RSS collector
â”‚   â”‚   â””â”€â”€ evaluator.py              # WebTestEvaluator class
â”‚   â”śâ”€â”€ colab/                        # Google Colab integration
â”‚   â”‚   â”śâ”€â”€ colab_setup.py            # Colab/Drive workspace linking and resume checks
â”‚   â”‚   â””â”€â”€ colab_secrets.json.template # Optional template for private Colab config
â”‚   â””â”€â”€ data/                         # Raw datasets (gitignored)
â”śâ”€â”€ experiments/                      # Generated output directory (gitignored data, skeleton tracked)
â”‚   â”śâ”€â”€ metrics/                      # training_results.csv from training pipeline
â”‚   â”śâ”€â”€ predictions/                  # Per-experiment prediction CSVs
â”‚   â”śâ”€â”€ web_test_results/             # Web testing CSVs and reports
â”‚   â”śâ”€â”€ hyperparams/                  # Best Optuna hyperparameters (JSON)
â”‚   â”śâ”€â”€ trial_history/                # Optuna trial history CSVs
â”‚   â””â”€â”€ results/                      # Visualisation outputs
â”śâ”€â”€ tests/                            
â”‚   â”śâ”€â”€ backend/                      # Tests for backend
â”‚   â”śâ”€â”€ extension/                    # Jest tests for extension modules
â”śâ”€â”€ saved_models/                     # Serialised models (.joblib, .pkl) â€” gitignored
â”śâ”€â”€ requirements.txt
â”śâ”€â”€ README.md
â”śâ”€â”€ docker-compose.yml
â”śâ”€â”€ Dockerfile                        # API container
â”śâ”€â”€ Dockerfile.research               # Training/research container
â””â”€â”€ fake_news_detector.ipynb          # Google Colab notebook
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

The API loads three model routes when matching artifacts are available:

- `short_text`: WELFake + XGBoost + TF-IDF
- `general`: ISOT + Random Forest + Bag of Words
- `long_article`: LIAR + BiLSTM + GloVe

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

1. Configure repository, branch, project path, and Drive path.
2. Mount Google Drive.
3. Clone or update the repository in the Colab runtime.
4. Install project dependencies.
5. Create the required Drive folders.
6. Upload raw datasets and the GloVe embedding file.
7. Validate inputs and preprocess datasets.
8. Configure project symlinks between Colab and Drive.
9. Train models.
10. Collect web data.
11. Evaluate trained models on web data.
12. Analyze and export results.

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
