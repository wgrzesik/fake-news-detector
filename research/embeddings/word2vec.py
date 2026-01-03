import numpy as np
import os
from typing import List, Union
from gensim.models import Word2Vec, KeyedVectors
from gensim.utils import simple_preprocess
from .base_embedding import BaseEmbedder
import pickle

class Word2VecEmbedder(BaseEmbedder):
    def __init__(self, vector_size: int = 100, window: int = 5, min_count: int = 1, model_path: str = "research/data/embeddings/GoogleNews-vectors-negative300.bin"):
        super().__init__("word2vec")
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.model = None
        self.model_path = model_path
        self.is_pretrained = False

    def _tokenize(self, texts: List[str]) -> List[List[str]]:
        clean_texts = self._preprocess_batch(texts)
        return [simple_preprocess(text) for text in clean_texts]
    
    def fit(self, texts: List[str]) -> None:    
        # STRATEGY 1: Load Pre-trained Vectors
        if self.model_path and os.path.exists(self.model_path):
            print(f"[Word2Vec] Loading pre-trained vectors from {self.model_path}")
            self.model = KeyedVectors.load_word2vec_format(self.model_path, binary=True)
            self.is_pretrained = True
            self.vector_size = self.model.vector_size
            print(f"[Word2Vec] Loaded {len(self.model)} vectors.")
        
        # STRATEGY 2: Train from Scratch
        else:
            print("[Word2Vec] No pre-trained file found. Training from scratch on dataset")
            tokenized_texts = self._tokenize(texts)
            self.model = Word2Vec(
                sentences=tokenized_texts,
                vector_size=self.vector_size,
                window=self.window,
                min_count=self.min_count,
                workers=4,
                epochs=10
            ).wv
            self.is_pretrained = False
            
        self.is_fitted = True

    def transform(self, texts: Union[str, List[str]]) -> np.ndarray:
        if not self.is_fitted or self.model is None:
            raise ValueError("Word2Vec model has not been fitted!")

        if isinstance(texts, str):
            texts = [texts]

        tokenized_texts = self._tokenize(texts)
        embeddings = np.zeros((len(tokenized_texts), self.vector_size))
        
        wv = self.model.wv if hasattr(self.model, 'wv') else self.model 

        for i, tokens in enumerate(tokenized_texts):
            valid_vectors = [wv[w] for w in tokens if w in wv]
            
            if valid_vectors:
                embeddings[i] = np.mean(valid_vectors, axis=0)
                
        return embeddings

    def save(self, path: str):
        config = {
            'is_pretrained': self.is_pretrained,
            'model_path': self.model_path,
            'vector_size': self.vector_size
        }
        
        if not self.is_pretrained:
            config['vectors'] = self.model
            
        with open(path, 'wb') as f:
            pickle.dump(config, f)

    def load(self, path: str):
        import pickle
        with open(path, 'rb') as f:
            config = pickle.load(f)
            
        self.is_pretrained = config['is_pretrained']
        self.model_path = config.get('model_path')
        self.vector_size = config['vector_size']
        
        if self.is_pretrained:
            self.model = KeyedVectors.load_word2vec_format(self.model_path, binary=True)
        else:
            self.model = config['vectors']
            
        self.is_fitted = True