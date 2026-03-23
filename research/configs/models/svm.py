from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from .base_model import BaseModel
from sklearn.calibration import CalibratedClassifierCV
from research.configs.embeddings.embedder_factory import EmbedderFactory


class SVMModel(BaseModel):
    def __init__(self, dataset_name: str, embedding_type: str, **kwargs):
        super().__init__(dataset_name, f"svm_{embedding_type}")

        self.embedder = EmbedderFactory.create(embedding_type)
        self.classifier = CalibratedClassifierCV(LinearSVC(**kwargs))
        self.scaler = StandardScaler(with_mean=False)
