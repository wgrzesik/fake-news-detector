from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from .base_model import BaseModel
from research.configs.embeddings.embedding_factory import EmbedderFactory


class NaiveBayesModel(BaseModel):
    """Gaussian Naive Bayes model for fake news detection."""

    def __init__(self, dataset_name: str, embedding_type: str, **kwargs):
        super().__init__(dataset_name, f"nb_{embedding_type}")

        self.embedder = EmbedderFactory.create(embedding_type)
        self.classifier = GaussianNB(**kwargs)
        self.scaler = StandardScaler()
