import numpy as np
import os
import pickle
from typing import List, Union
from .base_embedding import BaseEmbedder

class GloveEmbedder(BaseEmbedder):
    def __init__(self, vector_size: int = 100, glove_path: str = "research/data/embeddings/glove.6B.100d.txt"):
        super().__init__("glove")
        self.vector_size = vector_size
        self.embeddings_index = {}
        self.glove_path = glove_path

    def _load_pretrained_glove(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"GloVe source file not found at {path}")
        
        embeddings_index = {}
        with open(path, 'r', encoding='utf8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
        return embeddings_index

    def fit(self, texts: List[str]):
        print(f"[GloVe] Loading pre-trained vectors from {self.glove_path}...")
        self.embeddings_index = self._load_pretrained_glove(self.glove_path)
        self.is_fitted = True

    def transform(self, texts: Union[str, List[str]]) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("GloveEmbedder has not been fitted/loaded!")

        if isinstance(texts, str):
            texts = [texts]

        clean_texts = self._preprocess_batch(texts)
        embeddings = []

        for text in clean_texts:
            words = text.split()

            word_vectors = [self.embeddings_index[w] for w in words if w in self.embeddings_index]
            
            if not word_vectors:
                embeddings.append(np.zeros(self.vector_size))
            else:
                embeddings.append(np.mean(word_vectors, axis=0))
                
        return np.array(embeddings)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "glove_path": self.glove_path,
                    "vector_size": self.vector_size
                },
                f
            )

    def load(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.glove_path = data["glove_path"]
        self.vector_size = data["vector_size"]

        self.embeddings_index = self._load_pretrained_glove(self.glove_path)
        self.is_fitted = True
