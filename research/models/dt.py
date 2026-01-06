from sklearn.tree import DecisionTreeClassifier
from research.base import BaseFakeNewsModel
from research.embeddings.base_embedding import BaseEmbedder

class DecisionTreeModel(BaseFakeNewsModel):
    def __init__(self, dataset_name: str, embedding_type: str):
        super().__init__(dataset_name, f"dt_{embedding_type}")

        self.embedding_type = embedding_type
        self.embedder = BaseEmbedder.create(embedding_type)
        self.classifier = DecisionTreeClassifier(random_state=42)
        self.scaler = None