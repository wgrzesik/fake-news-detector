from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from research.configs.models.base_model import BaseModel
from research.configs.embeddings.embedder_factory import EmbedderFactory


class KNNModel(BaseModel):
    def __init__(self, dataset_name: str, embedding_type: str, **kwargs):
        super().__init__(dataset_name, f"knn_{embedding_type}")
        
        self.embedding_type = embedding_type
        self.embedder = EmbedderFactory.create(embedding_type)
        self.classifier = KNeighborsClassifier(**kwargs)
        self.scaler = StandardScaler(with_mean=False)
