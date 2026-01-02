from abc import ABC, abstractmethod
from typing import List, Any, Union
import re
from string import punctuation

class BaseEmbedder(ABC):
    """
    Abstract base class for all vectorization methods (TF-IDF, GloVe, BERT, etc.).
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.is_fitted = False
    
    def _clean_text(self, text: str) -> str:
        """
        Centralized text cleaning method.
        Used for both training and prediction (extension).
        """
        if not isinstance(text, str):
            return ""
            
        text = text.lower()
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'<.*?>', '', text)
        text = text.translate(str.maketrans('', '', punctuation))
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def _preprocess_batch(self, texts: List[str]) -> List[str]:
        """Helper function to clean a list of texts."""
        return [self._clean_text(t) for t in texts]
    
    @abstractmethod
    def fit(self, texts: List[str]) -> None:
        """
        Fits the vectorizer to the vocabulary (learns words).
        For pre-trained embeddings (e.g., BERT), this may be empty.
        """
        pass

    @abstractmethod
    def transform(self, texts: Union[str, List[str]]) -> Any:
        """
        Converts text(s) into numerical representation (vectors/matrices).
        """
        pass

    def fit_transform(self, texts: List[str]) -> Any:
        """
        Helper method: fits the model and returns vectors immediately.
        """
        self.fit(texts)
        return self.transform(texts)

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Saves the vectorizer state (e.g., TF-IDF vocabulary) to a file.
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """
        Loads the vectorizer state from a file.
        """
        pass