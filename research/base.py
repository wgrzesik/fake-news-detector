from abc import ABC, abstractmethod
import os
from typing import Any, Dict

class BaseFakeNewsModel(ABC):
    def __init__(self, dataset_name: str, model_name: str):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.save_dir = os.path.join("saved_models", dataset_name)
        os.makedirs(self.save_dir, exist_ok=True)

    @abstractmethod
    def train(self, X_train: Any, y_train: Any, X_val: Any = None, y_val: Any = None) -> None:
        """
        Train the model on the provided training set.
        """
        pass

    @abstractmethod
    def predict(self, text: str) -> Dict[str, Any]:
        """
        Prediction method for the Chrome extension (single text input).
        Returns a dictionary with labels and scores.
        """
        pass

    @abstractmethod
    def evaluate(self, X_test: Any, y_test: Any) -> Dict[str, float]:
        """
        Evaluation method for research purposes (full test set).
        Returns: {'accuracy': 0.9, 'f1': 0.88, ...}
        """
        pass

    @abstractmethod
    def save(self) -> None:
        """
        Save the model and its components (e.g., classifiers, scalers).
        """
        pass

    @abstractmethod
    def load(self) -> None:
        """
        Load the model and its components from the save directory.
        """
        pass

    def get_model_path(self, filename: str) -> str:
        """
        Helper method to generate a consistent file path for saving components.
        """
        return os.path.join(self.save_dir, f"{self.model_name}_{filename}")