import pickle
import os
from sklearn.feature_extraction.text import CountVectorizer
from .base_embedding import BaseEmbedder

class BowEmbedder(BaseEmbedder):
    def __init__(self, max_features: int = 5000):
        super().__init__("bow")
        self.vectorizer = CountVectorizer(
            max_features=max_features, 
            stop_words='english',
            ngram_range=(1, 2) 
        )

    def fit(self, texts):
        print(f"[BoW] Fitting on {len(texts)} documents")
        
        clean_texts = self._preprocess_batch(texts)
        
        self.vectorizer.fit(clean_texts)
        self.is_fitted = True

    def transform(self, texts):
        if not self.is_fitted:
            raise ValueError("BoW Vectorizer not fitted!")
            
        if isinstance(texts, str):
            texts = [texts]
            
        clean_texts = self._preprocess_batch(texts)
        
        return self.vectorizer.transform(clean_texts)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.vectorizer, f)

    def load(self, path: str):
        with open(path, 'rb') as f:
            self.vectorizer = pickle.load(f)
        self.is_fitted = True