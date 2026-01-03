import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from .base_embedding import BaseEmbedder

class TfidfEmbedder(BaseEmbedder):
    def __init__(self, max_features: int = 5000):
        super().__init__("tfidf")
        self.vectorizer = TfidfVectorizer(
            max_features=max_features, 
            stop_words='english',
            ngram_range=(1, 2)
        )

    def fit(self, texts: list):
        print(f"[TF-IDF] Fitting on {len(texts)} documents...")
        clean_texts = self._preprocess_batch(texts)
        self.vectorizer.fit(clean_texts)
        self.is_fitted = True

    def transform(self, texts: list):
        if not self.is_fitted:
            raise ValueError("TF-IDF Vectorizer has not been fitted yet!")
            
        if isinstance(texts, str):
            texts = [texts]
            
        clean_texts = self._preprocess_batch(texts)
        return self.vectorizer.transform(clean_texts)

    def save(self, path: str):
        """Saves the fitted vectorizer to disk"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        print(f"[Save] TF-IDF Vocab saved to {path}")

    def load(self, path: str):
        """Loads the vectorizer to reuse the same vocabulary"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"No TF-IDF model found at {path}")
        with open(path, 'rb') as f:
            self.vectorizer = pickle.load(f)
        self.is_fitted = True
        print(f"[Load] TF-IDF Vocab loaded from {path}")