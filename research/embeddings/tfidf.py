import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from .base_embedding import BaseEmbedder

class TfidfEmbedder(BaseEmbedder):
    def __init__(self, max_features: int = 5000):
        super().__init__("tfidf")
        self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')

    def fit(self, texts):
        print("   [TF-IDF] Cleaning data and training...")
        clean_texts = self._preprocess_batch(texts)
        self.vectorizer.fit(clean_texts)
        self.is_fitted = True

    def transform(self, text):
        if not self.is_fitted:
            raise ValueError("Vectorizer has not been fitted yet!")
            
        if isinstance(text, str):
            text = [text]
            
        clean_texts = self._preprocess_batch(text)
        return self.vectorizer.transform(clean_texts)

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self.vectorizer, f)

    def load(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        with open(path, 'rb') as f:
            self.vectorizer = pickle.load(f)
        self.is_fitted = True