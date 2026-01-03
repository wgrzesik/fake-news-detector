from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from research.base import BaseFakeNewsModel
from research.embeddings.base_embedding import BaseEmbedder 

class LogisticRegressionModel(BaseFakeNewsModel):
    def __init__(self, dataset_name: str, embedding_type: str):
        super().__init__(dataset_name, f"lr_{embedding_type}")

        self.embedder = BaseEmbedder.create(embedding_type)
        self.classifier = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42)
        self.scaler = StandardScaler(with_mean=False)