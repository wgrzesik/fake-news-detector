# Maintenance

This document explains how to develop and maintain the Fake News Detector project after the initial submission.

## Updating Models

When replacing or adding trained models:

- keep model artifacts under `saved_models/{dataset}/{model_name}/`,
- make sure each model directory contains the files expected by the model class, such as `classifier.joblib`, `embedder.pkl`, `scaler.joblib`, or neural-network checkpoints,
- update `MODEL_CONFIGS` in `backend/api.py` if the API should load a different dataset, model, or embedding combination,
- run the API and check `GET /health` to confirm that the expected route keys are loaded,
- update `docs/architecture.md` if the model routing table changes.

## Updating API Routing

The prediction API currently uses word-count routing:

- texts up to and including 100 words use `short_text`,
- texts over 100 words use `long_text`.

The active model configuration contains only these two route keys. If both model artifacts are missing at startup, the API reports a degraded state and prediction requests return `503`.

If thresholds or selected models change, update:

- `backend/api.py`,
- `docs/architecture.md`,
- `README.md`,
- tests in `tests/backend`.

After changes, run:

```bash
python -m pytest tests/backend
```

## Updating the Chrome Extension

When changing backend URL, request behavior, or response rendering:

- update `extension/config.js` for API URL, timeout, and UI constants,
- keep the expected API response format compatible with `extension/ui.js`,
- update `extension/api.js` if request or error handling changes,
- update `extension/popup.js` if the selection or interaction flow changes,
- reload the unpacked extension in `chrome://extensions/`,
- run extension tests from `tests/extension`.

Test command:

```bash
cd tests/extension
npm test
```

## Updating Research Results

After retraining models or rerunning web evaluation:

- keep `experiments/metrics/training_results.csv` as the main training metrics file,
- keep web evaluation outputs in `experiments/web_test_results/`,
- rerun `python -m research.analyze_results` to regenerate plots and summaries,
- check that generated outputs are saved under `experiments/results/`,
- document major result changes in `README.md`, `docs/architecture.md`, or research summary files.

Recommended workflow:

```bash
python -m research.train_models
python -m research.web.collect_web
python -m research.evaluate_web
python -m research.analyze_results
```

## Updating Configuration

Main configuration locations:

- `research/configs/config.yaml`: research pipeline, datasets, models, embeddings, Optuna, web scraping, and MLflow settings,
- `backend/api.py`: API model routes and routing thresholds,
- `extension/config.js`: extension API URL, text limits, timeout, and UI messages,
- `docker-compose.yml`: Docker services and exposed ports.

When changing configuration, update documentation and tests that depend on it.

## Dependency Maintenance

Python dependencies are listed in `requirements.txt`. JavaScript test dependencies are listed in `tests/extension/package.json`.

When adding or updating dependencies:

- keep dependency files committed,
- avoid committing generated dependency directories such as `node_modules/`,
- rerun backend and extension tests,
- note any new system-level requirements in `README.md`.

## Files That Should Not Be Committed

Do not commit:

- private tokens or credentials,
- `.env` files,
- raw large datasets,
- `node_modules/`,
- Python cache folders,
- temporary notebook outputs,
- local virtual environments,
- generated artifacts unless they are intentionally included for evaluation.

Large datasets, private configuration, or oversized model artifacts should be listed in the submission PDF and passed through Teams or OneDrive when needed.

## Tests Before Submission

Before delivering the repository, run:

```bash
python -m pytest tests/backend
cd tests/extension
npm test
```

The backend tests check prediction routing and API behavior. The extension tests check API calls, popup flow, and result rendering.

## Documentation Maintenance

Keep documentation synchronized with code changes:

- update `README.md` when setup, demo, test commands, or expected outputs change,
- update `docs/architecture.md` when components, routing, technologies, communication, or deployment details change,
- update this file when maintenance workflow, dependencies, or release checks change.
