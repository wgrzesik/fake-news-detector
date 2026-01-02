# embeddings/__init__.py

# Eksponujemy klasę bazową (żeby sprawdzanie typów isinstance działało łatwiej)
from .base_embedding import BaseEmbedder

# Eksponujemy konkretne implementacje
from .tfidf import TfidfEmbedder
from .word2vec import Word2VecEmbedder


# W przyszłości odkomentujesz to, gdy dodasz pliki:
# from .glove import GloveEmbedder
# from .bert_emb import BertEmbedder