from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from research.configs.models.base_model import BaseModel
from research.configs.embeddings.embedder_factory import EmbedderFactory

class LogisticRegressionModel(BaseModel):
    def __init__(self, dataset_name: str, embedding_type: str, **kwargs):
        super().__init__(dataset_name, f"lr_{embedding_type}")

        self.embedder = EmbedderFactory.create(embedding_type)
        self.classifier = LogisticRegression(**kwargs)
        self.scaler = StandardScaler(with_mean=False)