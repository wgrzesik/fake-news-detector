from abc import ABC, abstractmethod
from typing import List, Any, Union
import re
from string import punctuation

class BaseEmbedder(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.is_fitted = False
    
    @staticmethod
    def preprocess_text(text: str) -> str:
        if not isinstance(text, str):
            return ""

        text = text.lower()
        
        text = re.sub(r'^.*?\(reuters\)\s*[-—]\s*', '', text)
        text = re.sub(r'^[a-z\s,]+\s*[-—]\s*', '', text)
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'<.*?>', '', text)
        text = text.translate(str.maketrans('', '', punctuation))
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def _clean_text(self, text: str) -> str:
        """Instance method that calls the static preprocessor."""
        return self.preprocess_text(text)

    def _preprocess_batch(self, texts: List[str]) -> List[str]:
        return [self._clean_text(t) for t in texts]
    
    @staticmethod
    def create(embedding_type: str, **kwargs):
        if embedding_type == 'tfidf':
            from .tfidf import TfidfEmbedder
            return TfidfEmbedder(**kwargs)
        elif embedding_type == 'word2vec':
            from .word2vec import Word2VecEmbedder
            return Word2VecEmbedder(**kwargs)
        elif embedding_type == 'glove':
            from .glove import GloveEmbedder
            return GloveEmbedder(**kwargs)
        elif embedding_type == 'bow':
            from .bow import BowEmbedder
            return BowEmbedder(**kwargs)
        raise ValueError(f"Unknown embedding type: {embedding_type}")
    
    @abstractmethod
    def fit(self, texts: List[str]) -> None: pass

    @abstractmethod
    def transform(self, texts: Union[str, List[str]]) -> Any: pass

    def fit_transform(self, texts: List[str]) -> Any:
        self.fit(texts)
        return self.transform(texts)

    @abstractmethod
    def save(self, path: str) -> None: pass

    @abstractmethod
    def load(self, path: str) -> None: pass