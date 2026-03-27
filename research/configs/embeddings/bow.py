from omegaconf import ListConfig
from sklearn.feature_extraction.text import CountVectorizer
from .base_embedding import BaseEmbedder


class BowEmbedder(BaseEmbedder):
    """Bag of Words embedder using scikit-learn CountVectorizer."""

    def __init__(
        self,
        max_features=20000,
        ngram_range=(1, 1)
    ):

        if isinstance(ngram_range, (list, ListConfig)):
            ngram_range = tuple(ngram_range)

        self.vectorizer = CountVectorizer(
            max_features=max_features,
            ngram_range=ngram_range
        )

    def fit(self, texts):
        self.vectorizer.fit(texts)

    def transform(self, texts):
        return self.vectorizer.transform(texts)
