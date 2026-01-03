from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from research.base import BaseFakeNewsModel
from research.embeddings.base_embedding import BaseEmbedder 

class NaiveBayesModel(BaseFakeNewsModel):
    def __init__(self, dataset_name: str, embedding_type: str):
        super().__init__(dataset_name, f"nb_{embedding_type}")

        self.embedder = BaseEmbedder.create(embedding_type)
        self.classifier = GaussianNB()
        self.scaler = StandardScaler()