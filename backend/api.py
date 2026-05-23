import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from research.configs.models.model_factory import ModelFactory  # noqa: E402
from research.configs.models.base_model import BaseModel as ResearchBaseModel  # noqa: E402

logger = logging.getLogger("fake_news_api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

MODEL_CONFIGS: Dict[str, Dict[str, str]] = {
    "short_text":   {"dataset": "WELFake", "type": "xgb",    "emb": "tfidf"},
    "long_article": {"dataset": "LIAR",    "type": "bilstm", "emb": "glove"},
    "general":      {"dataset": "ISOT",    "type": "rf",     "emb": "bow"},
}

SHORT_TEXT_THRESHOLD = 30
LONG_ARTICLE_THRESHOLD = 100
MIN_TEXT_LENGTH = 10
MAX_TEXT_LENGTH = 50_000

ALLOWED_ORIGINS = [
    "http://127.0.0.1:8000",
    "http://localhost:8000",
    "chrome-extension://*",
]


def load_all_required_models(
    factory: type = ModelFactory,
    configs: Dict[str, Dict[str, str]] = MODEL_CONFIGS,
) -> Dict[str, ResearchBaseModel]:
    """Load startup models pre-trained on different datasets/text lengths."""
    loaded: Dict[str, ResearchBaseModel] = {}
    for key, cfg in configs.items():
        try:
            logger.info("Loading model '%s' (%s/%s/%s)", key, cfg["dataset"], cfg["type"], cfg["emb"])
            model = factory.get_model(cfg["dataset"], cfg["type"], cfg["emb"])
            model.load()
            loaded[key] = model
        except Exception as exc:
            logger.error("Failed to load model '%s': %s", key, exc)
    return loaded


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.models = load_all_required_models()
    yield
    app.state.models = {}


app = FastAPI(title="Smart Fake News Detector API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^(http://(127\.0\.0\.1|localhost):\d+|chrome-extension://[a-z]+)$",
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class TextRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_LENGTH)


def select_best_model(text: str, models: Dict[str, ResearchBaseModel]) -> Optional[ResearchBaseModel]:
    if not models:
        return None
    word_count = len(text.split())
    if word_count < SHORT_TEXT_THRESHOLD:
        return models.get("short_text") or models.get("general") or next(iter(models.values()))
    if word_count > LONG_ARTICLE_THRESHOLD:
        return models.get("long_article") or models.get("general") or next(iter(models.values()))
    return models.get("general") or next(iter(models.values()))


@app.get("/health")
def health(request: Request) -> Dict[str, Any]:
    models = getattr(request.app.state, "models", {}) or {}
    return {"status": "ok" if models else "degraded", "models_loaded": sorted(models.keys())}


@app.post("/predict")
def predict(request: Request, body: TextRequest) -> Dict[str, Any]:
    models = getattr(request.app.state, "models", {}) or {}
    if not models:
        raise HTTPException(status_code=503, detail="No models loaded.")

    raw_text = body.text.strip()
    if len(raw_text) < MIN_TEXT_LENGTH:
        return {"label": "NEUTRAL", "score": 0.0, "message": "Text too short for analysis"}

    model = select_best_model(raw_text, models)
    if model is None:
        raise HTTPException(status_code=500, detail="Failed to select a model.")

    try:
        result = model.predict(raw_text)
    except Exception as exc:
        logger.exception("Prediction error")
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}") from exc

    result["meta"] = {
        "used_dataset": model.dataset_name,
        "used_model": model.model_name,
        "word_count": len(raw_text.split()),
    }
    return result


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)