import os
import sys
from typing import Dict, Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(os.path.join(project_root))
os.chdir(project_root)

try:
    from research.configs.models.model_factory import ModelFactory
    from research.configs.models.base_model import BaseModel as ResearchBaseModel
except ImportError as e:
    print("IMPORT ERROR: Could not find files in the /research folder.")
    raise e

app = FastAPI(title="Smart Fake News Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS: Dict[str, Any] = {}

def load_all_required_models():
    """
    Loads a set of startup models.
    These models are pre-trained on different datasets and optimized for different text lengths.
    """
    configs = {
        "short_text": {"datasets": "LIAR", "type": "lr", "emb": "word2vec"},
       
        "long_article": {"datasets": "ISOT", "type": "svm", "emb": "tfidf"},
        
        "general": {"datasets": "WELFake", "type": "xgb", "emb": "tfidf"}
    }

    for key, cfg in configs.items():
        try:
            print(f"Loading model for category: {key} ({cfg['datasets']})...")
            model = ModelFactory.get_model(cfg['datasets'], cfg['type'], cfg['emb'])
            model.load()
            MODELS[key] = model
            print(f"Model {key} is ready.")
        except Exception as e:
            print(f"! Failed to load model {key}: {e}")

load_all_required_models()

class TextRequest(BaseModel):
    text: str

def select_best_model(text: str) -> ResearchBaseModel:
    """
    Decision logic: which model best handles this specific text?
    """
    word_count = len(text.split())
    
    if word_count < 30:
        return MODELS.get("short_text") or MODELS.get("general")
    
    if word_count > 100:
        return MODELS.get("long_article") or MODELS.get("general")
    
    return MODELS.get("general") or list(MODELS.values())[0]

@app.post("/predict")
def predict(request: TextRequest):
    if not MODELS:
        raise HTTPException(status_code=503, detail="No models loaded.")
    
    raw_text = request.text.strip()
    if len(raw_text) < 10:
         return {"label": "NEUTRAL", "score": 0.0, "message": "Text too short for analysis"}

    model = select_best_model(raw_text)
    
    if not model:
        raise HTTPException(status_code=500, detail="Failed to select a model.")

    try:
        result = model.predict(raw_text)
        result["meta"] = {
            "used_dataset": model.dataset_name,
            "used_model": model.model_name,
            "word_count": len(raw_text.split())
        }
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
