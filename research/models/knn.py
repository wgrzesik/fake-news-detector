from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from research.base import BaseFakeNewsModel
from research.embeddings.base_embedding import BaseEmbedder

class KNNModel(BaseFakeNewsModel):
    def __init__(self, dataset_name: str, embedding_type: str):
        super().__init__(dataset_name, f"knn_{embedding_type}")
        
        self.embedding_type = embedding_type
        self.embedder = BaseEmbedder.create(embedding_type)
        self.classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
        self.scaler = StandardScaler(with_mean=False)