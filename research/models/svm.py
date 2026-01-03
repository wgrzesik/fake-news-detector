from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from research.base import BaseFakeNewsModel
from sklearn.calibration import CalibratedClassifierCV
from research.embeddings.base_embedding import BaseEmbedder 

class SVMModel(BaseFakeNewsModel):
    def __init__(self, dataset_name: str, embedding_type: str, **kwargs):
        super().__init__(dataset_name, f"svm_{embedding_type}")

        self.embedder = BaseEmbedder.create(embedding_type, **kwargs)
        svm = LinearSVC(dual=False, random_state=42, max_iter=5000, tol=1e-3)
        self.classifier = CalibratedClassifierCV(svm)
        self.scaler = StandardScaler(with_mean=False)
