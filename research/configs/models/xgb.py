from xgboost import XGBClassifier
from .base_model import BaseModel
from research.configs.embeddings.embedding_factory import EmbedderFactory


class XGBoostModel(BaseModel):
    """XGBoost model for fake news detection."""

    def __init__(self, dataset_name: str, embedding_type: str, **kwargs):
        super().__init__(dataset_name, f"xgb_{embedding_type}")
        
        self.embedder = EmbedderFactory.create(embedding_type)
        self.classifier = XGBClassifier(**kwargs)
        self.scaler = None
