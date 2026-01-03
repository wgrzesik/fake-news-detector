from sklearn.naive_bayes import MultinomialNB
from research.base import BaseFakeNewsModel
from research.embeddings.base_embedding import BaseEmbedder 

class MultinomialNaiveBayesModel(BaseFakeNewsModel):
    def __init__(self, dataset_name: str, embedding_type: str):
        super().__init__(dataset_name, f"mnb_{embedding_type}")

        self.embedder = BaseEmbedder.create(embedding_type)
        self.classifier = MultinomialNB(alpha=1.0)
        self.scaler = None