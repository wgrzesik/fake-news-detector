from sklearn.naive_bayes import MultinomialNB
from research.configs.models.base_model import BaseModel
from research.configs.embeddings.embedder_factory import EmbedderFactory


class MultinomialNaiveBayesModel(BaseModel):
    def __init__(self, dataset_name: str, embedding_type: str, **kwargs):
        super().__init__(dataset_name, f"mnb_{embedding_type}")

        self.embedder = EmbedderFactory.create(embedding_type)
        self.classifier = MultinomialNB(**kwargs)
        self.scaler = None
