from abc import ABC, abstractmethod
from typing import List, Any, Union
import re
from string import punctuation
import io
import pickle

class BaseEmbedder(ABC):
    def __init__(self, model_name: str, preprocessing_mode: str = "classic"):
        self.model_name = model_name
        self.preprocessing_mode = preprocessing_mode  # "classic" | "bert"
        self.is_fitted = False
    
    @staticmethod
    def preprocess_classic(text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'<.*?>', '', text)
        text = text.translate(str.maketrans('', '', punctuation))
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    @staticmethod
    def preprocess_bert(text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _clean_text(self, text: str) -> str:
        if self.preprocessing_mode == "bert":
            return self.preprocess_bert(text)
        return self.preprocess_classic(text)

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
    
    def clone(self) -> "BaseEmbedder":
        """
        Safe clone via in-memory pickle.
        Required for fair benchmarking & multiprocessing.
        """
        buffer = io.BytesIO()
        pickle.dump(self, buffer)
        buffer.seek(0)
        return pickle.load(buffer)
    
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