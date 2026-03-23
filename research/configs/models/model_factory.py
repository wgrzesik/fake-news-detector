from typing import List
from .base_model import BaseModel
from research.configs.models import SVMModel, LogisticRegressionModel, NaiveBayesModel, MultinomialNaiveBayesModel, \
    KNNModel, RandomForestModel, DecisionTreeModel, XGBoostModel


class ModelFactory:
    MODEL_REGISTRY = {
        'svm': SVMModel,
        'lr': LogisticRegressionModel,
        'nb': NaiveBayesModel,
        'mnb': MultinomialNaiveBayesModel,
        'knn': KNNModel,
        'rf': RandomForestModel,
        'dt': DecisionTreeModel,
        'xgb': XGBoostModel,
    }

    # Compatibility map: Model Type -> List of allowed embeddings types
    COMPATIBILITY_MAP = {
        # ML Models
        'svm': ['tfidf', 'word2vec', 'glove'],
        'lr': ['tfidf', 'word2vec', 'glove'],
        'nb': ['word2vec', 'glove'],
        'mnb': ['tfidf', 'bow'],

        'knn': ['word2vec', 'glove', 'bow'],
        'rf': ['tfidf', 'word2vec', 'glove', 'bow'],
        'dt': ['tfidf', 'word2vec', 'glove', 'bow'],

        'xgb': ['tfidf', 'bow'],

        # DL Models
        'lstm': ['word2vec', 'glove', 'fasttext'],

        # Transformer Models (embeddings are inherent to the pre-trained models name)
        'bert': ['bert-base-uncased', 'bert-large-uncased'],
        'roberta': ['roberta-base']
    }

    @staticmethod
    def get_model(dataset_name: str, model_type: str, embedding_type: str, **kwargs) -> BaseModel:
        """
        Instantiates a models after checking for compatibility between models and embeddings.
        """
        valid_embeddings = ModelFactory.COMPATIBILITY_MAP.get(model_type, [])
        if embedding_type not in valid_embeddings:
            raise ValueError(f"Incompatibility detected! Model '{model_type}' does not support '{embedding_type}'. "
                             f"Available options: {valid_embeddings}")

        model_class = ModelFactory.MODEL_REGISTRY[model_type]
        return model_class(dataset_name=dataset_name, embedding_type=embedding_type, **kwargs)

    @staticmethod
    def get_valid_models_for_embedding(embedding_type: str) -> List[str]:
        """Helper: returns a list of models that support a specific embeddings."""
        return [m for m, embs in ModelFactory.COMPATIBILITY_MAP.items() if embedding_type in embs]

    @staticmethod
    def get_valid_embeddings_for_model(model_type: str) -> List[str]:
        """Helper: returns a list of embeddings supported by a specific models."""
        return ModelFactory.COMPATIBILITY_MAP.get(model_type, [])
