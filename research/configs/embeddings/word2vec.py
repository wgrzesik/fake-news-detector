from gensim.models import Word2Vec
import numpy as np
from .base_embedding import BaseEmbedder


class Word2VecEmbedder(BaseEmbedder):
    def __init__(
        self,
        vector_size=100,
        window=5,
        min_count=2,
        workers=4
    ):

        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers

        self.model = None

    def fit(self, texts):
        tokenized = [t.split() for t in texts]

        self.model = Word2Vec(
            sentences=tokenized,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers
        )

    def transform(self, texts):
        vectors = []

        for text in texts:
            tokens = text.split()

            word_vectors = [
                self.model.wv[w] for w in tokens
                if w in self.model.wv
            ]

            if word_vectors:
                vectors.append(np.mean(word_vectors, axis=0))
            else:
                vectors.append(np.zeros(self.vector_size))

        return np.array(vectors)
