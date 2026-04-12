from typing import List
from .base_model import BaseModel
from research.configs.models import SVMModel, LogisticRegressionModel, NaiveBayesModel, MultinomialNaiveBayesModel, \
    KNNModel, RandomForestModel, DecisionTreeModel, XGBoostModel, BertModel, RobertaModel


class ModelFactory:
    """Factory for creating model instances with embedding compatibility validation."""

    MODEL_REGISTRY = {
        'svm': SVMModel,
        'lr': LogisticRegressionModel,
        'nb': NaiveBayesModel,
        'mnb': MultinomialNaiveBayesModel,
        'knn': KNNModel,
        'rf': RandomForestModel,
        'dt': DecisionTreeModel,
        'xgb': XGBoostModel,
        'bert': BertModel,
        'roberta': RobertaModel,
    }

    # Models whose training / evaluation operates on raw text, not pre-computed vectors
    TRANSFORMER_MODELS = {'bert', 'roberta'}

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
        #'lstm': ['word2vec', 'glove'],

        # Transformer Models (embeddings are inherent to the pre-trained models name)
        'bert': ['bert-base-uncased'],
        'roberta': ['roberta-base']
    }

    # Preprocessing map: Model Type -> required preprocessing mode
    PREPROCESSING_MAP = {
        # ML Models – classic preprocessing (lowering, URL/HTML/punct/digit removal)
        'svm': 'classic',
        'lr': 'classic',
        'nb': 'classic',
        'mnb': 'classic',
        'knn': 'classic',
        'rf': 'classic',
        'dt': 'classic',
        'xgb': 'classic',

        # DL Models
        'lstm': 'classic',

        # Transformer Models – minimal preprocessing (BERT tokenizer handles the rest)
        'bert': 'bert',
        'roberta': 'bert',
    }

    @staticmethod
    def get_model(dataset_name: str, model_type: str, embedding_type: str, **kwargs) -> BaseModel:
        """
        Instantiate a model after checking compatibility between model and embedding.
        """
        valid_embeddings = ModelFactory.COMPATIBILITY_MAP.get(model_type, [])
        if embedding_type not in valid_embeddings:
            raise ValueError(f"Incompatibility detected! Model '{model_type}' does not support '{embedding_type}'. "
                             f"Available options: {valid_embeddings}")

        model_class = ModelFactory.MODEL_REGISTRY[model_type]
        return model_class(dataset_name=dataset_name, embedding_type=embedding_type, **kwargs)

    @staticmethod
    def get_valid_models_for_embedding(embedding_type: str) -> List[str]:
        """Helper: returns a list of models that support a specific embedding."""
        return [m for m, embs in ModelFactory.COMPATIBILITY_MAP.items() if embedding_type in embs]

    @staticmethod
    def get_valid_embeddings_for_model(model_type: str) -> List[str]:
        """Helper: returns a list of embeddings supported by a specific model."""
        return ModelFactory.COMPATIBILITY_MAP.get(model_type, [])

    @staticmethod
    def get_preprocessing_for_model(model_type: str, fallback: str = "classic") -> str:
        """Returns the required preprocessing mode for a given model type.
        Falls back to *fallback* when the model is not in the map."""
        return ModelFactory.PREPROCESSING_MAP.get(model_type, fallback)

    @staticmethod
    def get_all_preprocessing_modes() -> List[str]:
        """Returns a sorted list of unique preprocessing modes used across all models."""
        return sorted(set(ModelFactory.PREPROCESSING_MAP.values()))

    @staticmethod
    def is_transformer_model(model_type: str) -> bool:
        """Check if a model type is a transformer that operates on raw text."""
        return model_type in ModelFactory.TRANSFORMER_MODELS

