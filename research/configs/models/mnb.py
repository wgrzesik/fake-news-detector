from sklearn.naive_bayes import MultinomialNB
from .base_model import BaseModel
from research.configs.embeddings.embedding_factory import EmbedderFactory


class MultinomialNaiveBayesModel(BaseModel):
    """Multinomial Naive Bayes model for fake news detection."""

    def __init__(self, dataset_name: str, embedding_type: str, **kwargs):
        super().__init__(dataset_name, f"mnb_{embedding_type}")

        self.embedder = EmbedderFactory.create(embedding_type)
        self.classifier = MultinomialNB(**kwargs)
        self.scaler = None
