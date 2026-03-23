import copy
import os
from abc import ABC, abstractmethod

import joblib


class BaseEmbedder(ABC):
    def clone(self):
        return copy.deepcopy(self)

    @abstractmethod
    def fit(self, texts):
        pass

    @abstractmethod
    def transform(self, texts):
        pass

    def fit_transform(self, texts):
        self.fit(texts)
        return self.transform(texts)
    
    def save(self, path: str):
        """Serializes the embedder to a file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str):
        """Loads the embedder from a file."""
        return joblib.load(path)
