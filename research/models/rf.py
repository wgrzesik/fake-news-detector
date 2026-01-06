from sklearn.ensemble import RandomForestClassifier
from research.base import BaseFakeNewsModel
from research.embeddings.base_embedding import BaseEmbedder

class RandomForestModel(BaseFakeNewsModel):
    def __init__(self, dataset_name: str, embedding_type: str):
        super().__init__(dataset_name, f"rf_{embedding_type}")

        self.embedding_type = embedding_type
        self.embedder = BaseEmbedder.create(embedding_type)
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        self.scaler = None