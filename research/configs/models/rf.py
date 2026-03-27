from sklearn.ensemble import RandomForestClassifier
from .base_model import BaseModel
from research.configs.embeddings.embedding_factory import EmbedderFactory


class RandomForestModel(BaseModel):
    """Random Forest model for fake news detection."""

    def __init__(self, dataset_name: str, embedding_type: str, **kwargs):
        super().__init__(dataset_name, f"rf_{embedding_type}")

        self.embedder = EmbedderFactory.create(embedding_type)
        self.classifier = RandomForestClassifier(**kwargs)
        self.scaler = None
