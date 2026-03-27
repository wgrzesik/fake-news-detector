import re
from string import punctuation
from typing import List


class TextPreprocessor:
    """Text preprocessing with multiple mode support (classic, bert)."""

    def __init__(self, mode: str = "classic"):
        self.mode = mode

    def preprocess(self, text: str) -> str:
        if self.mode == "classic":
            return self.preprocess_classic(text)

        if self.mode == "bert":
            return self.preprocess_bert(text)

        raise ValueError(f"Unknown preprocessing mode: {self.mode}")

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

    def transform(self, texts: List[str]) -> List[str]:
        """Apply preprocessing to a list of texts."""
        return [self.preprocess(t) for t in texts]