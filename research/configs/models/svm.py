from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from .base_model import BaseModel
from sklearn.calibration import CalibratedClassifierCV
from research.configs.embeddings.embedding_factory import EmbedderFactory


class SVMModel(BaseModel):
    """Support Vector Machine model for fake news detection."""

    def __init__(self, dataset_name: str, embedding_type: str, **kwargs):
        super().__init__(dataset_name, f"svm_{embedding_type}")

        self.embedder = EmbedderFactory.create(embedding_type)
        self.classifier = CalibratedClassifierCV(LinearSVC(**kwargs))
        self.scaler = StandardScaler(with_mean=False)
