import numpy as np
import os
from typing import List, Union
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from .base_embedding import BaseEmbedder

class Word2VecEmbedder(BaseEmbedder):
    def __init__(self, vector_size: int = 100, window: int = 5, min_count: int = 1):
        super().__init__("word2vec")
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.model = None

    def _tokenize(self, texts: List[str]) -> List[List[str]]:
        """
        First cleans the text using the BaseEmbedder method,
        then splits it into tokens.
        """
        clean_texts = self._preprocess_batch(texts)
        return [simple_preprocess(text) for text in clean_texts]

    def fit(self, texts: List[str]) -> None:
        print("   [Word2Vec] Cleaning and training...")
        tokenized_texts = self._tokenize(texts)
        
        self.model = Word2Vec(
            sentences=tokenized_texts,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=4
        )
        self.is_fitted = True

    def transform(self, texts: Union[str, List[str]]) -> np.ndarray:
        if not self.is_fitted or self.model is None:
            raise ValueError("Word2Vec model has not been fitted!")

        if isinstance(texts, str):
            texts = [texts]

        tokenized_texts = self._tokenize(texts)
        
        embeddings = []
        for tokens in tokenized_texts:
            if not tokens:
                embeddings.append(np.zeros(self.vector_size))
                continue
            
            word_vectors = [self.model.wv[word] for word in tokens if word in self.model.wv]
            
            if not word_vectors:
                embeddings.append(np.zeros(self.vector_size))
            else:
                embeddings.append(np.mean(word_vectors, axis=0))
                
        return np.array(embeddings)

    def save(self, path: str) -> None:
        self.model.save(path)

    def load(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file missing: {path}")
        self.model = Word2Vec.load(path)
        self.is_fitted = True