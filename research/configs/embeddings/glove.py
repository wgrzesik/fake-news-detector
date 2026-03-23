import numpy as np
from .base_embedding import BaseEmbedder


class GloveEmbedder(BaseEmbedder):
    def __init__(self, model_path: str = "research/data/embeddings/glove.6B.100d.txt", embedding_dim: int = 100):
        """
        Initializes the GloVe embedder.
        Requires a pre-trained GloVe file (e.g., glove.6B.100d.txt).
        """
        self.embedding_dim = embedding_dim
        self.model_path = model_path
        self.embeddings_index = {}
        self._load_glove()

    def _load_glove(self):
        print(f"Loading GloVe embeddings from {self.model_path}...")
        try:
            with open(self.model_path, encoding="utf8") as f:
                for line in f:
                    values = line.split()
                    word = values[0]
                    coefs = np.asarray(values[1:], dtype='float32')
                    self.embeddings_index[word] = coefs
            print(f"[Success] Loaded {len(self.embeddings_index)} word vectors.")
        except FileNotFoundError:
            print(f"[Warning] GloVe file not found at {self.model_path}. "
                  "Please ensure you have downloaded it.")

    def fit(self, texts):
        # Pre-trained embeddings do not require fitting to the training data.
        pass

    def transform(self, texts):
        # Create an averaged document vector for each text
        X_vec = []
        for text in texts:
            # Simple whitespace tokenization; replace with spaCy/NLTK if preferred
            words = str(text).lower().split()
            vecs = [self.embeddings_index[w] for w in words if w in self.embeddings_index]
            
            if len(vecs) > 0:
                X_vec.append(np.mean(vecs, axis=0))
            else:
                # Fallback for empty strings or texts with zero known words
                X_vec.append(np.zeros(self.embedding_dim))
                
        return np.vstack(X_vec)
