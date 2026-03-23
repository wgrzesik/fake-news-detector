from sklearn.tree import DecisionTreeClassifier
from research.configs.models.base_model import BaseModel
from research.configs.embeddings.embedder_factory import EmbedderFactory


class DecisionTreeModel(BaseModel):
    def __init__(self, dataset_name: str, embedding_type: str, **kwargs):
        super().__init__(dataset_name, f"dt_{embedding_type}")

        self.embedding_type = embedding_type
        self.embedder = EmbedderFactory.create(embedding_type)
        self.classifier = DecisionTreeClassifier(**kwargs)
        self.scaler = None
