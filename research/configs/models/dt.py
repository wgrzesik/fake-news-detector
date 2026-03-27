from sklearn.tree import DecisionTreeClassifier
from .base_model import BaseModel
from research.configs.embeddings.embedding_factory import EmbedderFactory


class DecisionTreeModel(BaseModel):
    """Decision Tree model for fake news detection."""

    def __init__(self, dataset_name: str, embedding_type: str, **kwargs):
        super().__init__(dataset_name, f"dt_{embedding_type}")

        self.embedder = EmbedderFactory.create(embedding_type)
        self.classifier = DecisionTreeClassifier(**kwargs)
        self.scaler = None
