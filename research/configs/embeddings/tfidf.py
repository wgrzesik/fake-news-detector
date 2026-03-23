from omegaconf import ListConfig
from sklearn.feature_extraction.text import TfidfVectorizer
from .base_embedding import BaseEmbedder


class TfidfEmbedder(BaseEmbedder):
    def __init__(
        self,
        max_features=20000,
        ngram_range=(1, 2),
        min_df=2
    ):

        if isinstance(ngram_range, (list, ListConfig)):
            ngram_range = tuple(ngram_range)

        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df
        )

    def fit(self, texts):
        self.vectorizer.fit(texts)

    def transform(self, texts):
        return self.vectorizer.transform(texts)
