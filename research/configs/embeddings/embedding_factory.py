from .tfidf import TfidfEmbedder
from .bow import BowEmbedder
from .word2vec import Word2VecEmbedder
from .glove import GloveEmbedder


class EmbedderFactory:
    """Factory for creating embedding instances by name."""

    REGISTRY = {
        "tfidf": TfidfEmbedder,
        "bow": BowEmbedder,
        "word2vec": Word2VecEmbedder,
        "glove": GloveEmbedder
    }

    @staticmethod
    def create(name, **params):
        if name not in EmbedderFactory.REGISTRY:
            raise ValueError(f"Unknown embedding: {name}")

        return EmbedderFactory.REGISTRY[name](**params)
