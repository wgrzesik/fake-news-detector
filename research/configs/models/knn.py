from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from .base_model import BaseModel
from research.configs.embeddings.embedding_factory import EmbedderFactory


class KNNModel(BaseModel):
    """K-Nearest Neighbours model for fake news detection."""

    def __init__(self, dataset_name: str, embedding_type: str, **kwargs):
        super().__init__(dataset_name, f"knn_{embedding_type}")

        self.embedder = EmbedderFactory.create(embedding_type)
        self.classifier = KNeighborsClassifier(**kwargs)
        self.scaler = StandardScaler(with_mean=False)
