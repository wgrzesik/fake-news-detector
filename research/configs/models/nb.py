from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from research.configs.models.base_model import BaseModel
from research.configs.embeddings.embedder_factory import EmbedderFactory


class NaiveBayesModel(BaseModel):
    def __init__(self, dataset_name: str, embedding_type: str, **kwargs):
        super().__init__(dataset_name, f"nb_{embedding_type}")

        self.embedder = EmbedderFactory.create(embedding_type)
        self.classifier = GaussianNB(**kwargs)
        self.scaler = StandardScaler()
