## Maintenance

### Updating Models

When replacing or adding trained models:

- keep model artifacts under `saved_models/{dataset}/{model_name}/`,
- make sure each model directory contains the files expected by the model class, such as `classifier.joblib`, `embedder.pkl`, or neural-network checkpoints,
- update `MODEL_CONFIGS` in `backend/api.py` if the API should load a different dataset/model/embedding combination,
- run `GET /health` after starting the API to confirm that the expected route keys are loaded.

### Updating API Routing

The prediction API uses word-count routing:

- texts under 30 words use `short_text`,
- texts from 30 to 100 words use `general`,
- texts over 100 words use `long_article`.

If these thresholds or selected models change, update both:

- `backend/api.py`,
- `docs/architecture.md`.

### Updating the Chrome Extension

When changing the backend URL or request behavior:

- update `extension/config.js`,
- keep the expected API response format compatible with `extension/ui.js`,
- reload the unpacked extension in `chrome://extensions/`,
- run extension tests from `tests/extension`.

### Updating Research Results

After retraining or rerunning evaluation:

- keep `experiments/metrics/training_results.csv` as the main training metrics file,
- keep web evaluation outputs in `experiments/web_test_results/`,
- rerun `python -m research.analyze_results` to regenerate plots and summaries,
- document any major result changes in the README or research summary files.

### Tests Before Submission

Before delivering the repository, run:

```bash
python -m pytest tests/backend
cd tests/extension
npm test