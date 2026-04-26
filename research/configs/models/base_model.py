import os
import joblib
from abc import ABC
from typing import List, Dict, Any
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from research.configs.embeddings.base_embedding import BaseEmbedder
from research.configs.preprocessing.preprocessor import TextPreprocessor


class BaseModel(ABC):
    """Abstract base class for all fake news detection models."""

    def __init__(self, dataset_name: str, model_name: str, preprocessing_mode: str = "classic"):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.preprocessing_mode = preprocessing_mode
        self.embedder: BaseEmbedder = None
        self.classifier = None
        self.scaler = None

    def get_model_path(self) -> str:
        """Centralized path management for saving/loading."""
        return os.path.join("saved_models", self.dataset_name, self.model_name)

    def train(self, X_train_raw: List[str], y_train: List[int]) -> None:
        """Vectorize raw (pre-cleaned) text then fit the classifier."""
        print(f"Training {self.model_name} (Standard Path)")
        X_vec = self.embedder.fit_transform(X_train_raw)

        if self.scaler:
            X_vec = self.scaler.fit_transform(X_vec)

        self.classifier.fit(X_vec, y_train)

    def evaluate(self, X_test_raw: List[str], y_test: List[int]) -> Dict[str, float]:
        """Transform pre-cleaned text, run predictions, and return metrics."""
        X_vec = self.embedder.transform(X_test_raw)
        return self.evaluate_on_vectors(X_vec, y_test)

    def train_on_vectors(self, X_vec: np.ndarray, y_train: List[int]) -> None:
        """Fast path: handles scaling if a scaler exists"""
        if self.scaler:
            X_vec = self.scaler.fit_transform(X_vec)
        self.classifier.fit(X_vec, y_train)

    def evaluate_on_vectors(self, X_vec: np.ndarray, y_test: List[int]) -> Dict[str, float]:
        """Fast path: uses scaler if it exists"""
        if self.scaler:
            X_vec = self.scaler.transform(X_vec)
        y_pred = self.classifier.predict(X_vec)

        y_test = np.array(y_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
            "recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
            "f1_score": f1_score(y_test, y_pred, average='weighted', zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
        }
        return metrics

    def predict(self, text: str) -> Dict[str, Any]:
        """Predict the label for a single string of text."""
        if self.classifier is None or self.embedder is None:
            raise ValueError("Model not loaded or trained.")

        # Apply the same preprocessing that was used during training
        preprocessor = TextPreprocessor(mode=self.preprocessing_mode)
        processed_text = preprocessor.preprocess(text)

        vec = self.embedder.transform([processed_text])

        if self.scaler:
            vec = self.scaler.transform(vec)

        prediction = int(self.classifier.predict(vec)[0])

        score = 1.0
        if hasattr(self.classifier, "predict_proba"):
            proba = self.classifier.predict_proba(vec)[0]
            score = float(np.max(proba))
        elif hasattr(self.classifier, "decision_function"):
            decision = self.classifier.decision_function(vec)[0]
            score = float(1 / (1 + np.exp(-decision)))

        return {
            "label": "REAL" if prediction == 1 else "FAKE",
            "score": score,
            "model": self.model_name,
            "dataset": self.dataset_name
        }

    def save(self):
        """Save both the classifier and the embedder state."""
        base_path = self.get_model_path()
        os.makedirs(base_path, exist_ok=True)

        print(f"\n[Saving Model] {base_path}")

        # Verify classifier exists
        if self.classifier is None:
            raise ValueError("Cannot save: classifier is None")

        # Verify embedder exists
        if self.embedder is None:
            raise ValueError("Cannot save: embedder is None")

        # Save classifier
        classifier_path = os.path.join(base_path, "classifier.joblib")
        joblib.dump(self.classifier, classifier_path)
        print(f"Classifier saved")

        # Save scaler if exists
        if self.scaler:
            scaler_path = os.path.join(base_path, "scaler.joblib")
            joblib.dump(self.scaler, scaler_path)
            print(f"Scaler saved")

        # Save embedder
        embedder_path = os.path.join(base_path, "embedder.pkl")
        self.embedder.save(embedder_path)

        # Verify files exist
        if not os.path.exists(classifier_path):
            raise IOError("Failed to save classifier")
        if not os.path.exists(embedder_path):
            raise IOError("Failed to save embedder")

        print(f"Model saved successfully!\n")

    def load(self):
        """Load both the classifier and the embedder state."""
        base_path = self.get_model_path()

        if not os.path.exists(base_path):
            raise FileNotFoundError(f"Model path not found: {base_path}")

        # Load classifier
        classifier_path = os.path.join(base_path, "classifier.joblib")
        if not os.path.exists(classifier_path):
            raise FileNotFoundError(f"Classifier not found: {classifier_path}")

        self.classifier = joblib.load(classifier_path)

        # Load scaler if exists
        scaler_path = os.path.join(base_path, "scaler.joblib")
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
        else:
            self.scaler = None

        # Load embedder
        embedder_path = os.path.join(base_path, "embedder.pkl")
        if not os.path.exists(embedder_path):
            raise FileNotFoundError(f"Embedder not found: {embedder_path}")

        self.embedder = self.embedder.load(embedder_path)

        # Verify embedder is fitted
        if hasattr(self.embedder, 'vectorizer'):
            if not hasattr(self.embedder.vectorizer, 'vocabulary_'):
                raise ValueError("Embedder loaded but not fitted - vocabulary missing")

        print(f"Model loaded from: {base_path}")
        print(f"Classifier loaded")
        print(f"Embedder loaded and verified")
        if self.scaler:
            print(f"Scaler loaded")

    def predict_proba_on_vectors(self, X_vec: np.ndarray) -> np.ndarray:
        """Fast path: uses scaler if it exists, then returns positive class probabilities."""
        if self.scaler:
            X_vec = self.scaler.transform(X_vec)

        if hasattr(self.classifier, "predict_proba"):
            probs = self.classifier.predict_proba(X_vec)
            if len(probs.shape) == 2 and probs.shape[1] == 2:
                return probs[:, 1]
            return probs
        else:
            raise AttributeError(f"{self.classifier.__class__.__name__} does not support probability predictions.")
