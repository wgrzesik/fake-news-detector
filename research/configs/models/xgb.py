from xgboost import XGBClassifier
from research.configs.models.base_model import BaseModel
from research.configs.embeddings.embedder_factory import EmbedderFactory


class XGBoostModel(BaseModel):
    def __init__(self, dataset_name: str, embedding_type: str, **kwargs):
        super().__init__(dataset_name, f"xgb_{embedding_type}")
        
        self.embedder = EmbedderFactory.create(embedding_type)
        self.classifier = XGBClassifier(**kwargs)
        self.scaler = None
