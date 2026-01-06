import os
import joblib
from abc import ABC
from typing import List, Dict, Any
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from research.embeddings.base_embedding import BaseEmbedder

class BaseFakeNewsModel(ABC):
    def __init__(self, dataset_name: str, model_name: str):

        self.dataset_name = dataset_name
        self.model_name = model_name
        self.embedder: BaseEmbedder = None
        self.classifier = None 
        self.scaler = None

    def get_model_path(self) -> str:
        """Centralized path management for saving/loading"""
        return os.path.join("saved_models", self.dataset_name, self.model_name)


    def train(self, X_train_raw: List[str], y_train: List[int]):
        """Full pipeline: Clean -> Vectorize -> Fit"""
        print(f"Training {self.model_name} (Standard Path)")
        X_vec = self.embedder.fit_transform(X_train_raw)

        if self.scaler:
            X_vec = self.scaler.fit_transform(X_vec)

        self.classifier.fit(X_vec, y_train)

    def evaluate(self, X_test_raw: List[str], y_test: List[int]) -> Dict[str, float]:
        """Full pipeline: Clean -> Transform -> Predict -> Metrics"""
        X_vec = self.embedder.transform(X_test_raw)
        return self.evaluate_on_vectors(X_vec, y_test)

    def train_on_vectors(self, X_vec, y_train):
        """Fast Path: Handles scaling if a scaler exists"""
        if self.scaler:
            X_vec = self.scaler.fit_transform(X_vec)
        self.classifier.fit(X_vec, y_train)

    def evaluate_on_vectors(self, X_vec, y_test):
        """Fast Path: Uses scaler if it exists"""
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
        """Predicts the label for a single string of text"""
        if self.classifier is None or self.embedder is None:
            raise ValueError("Model not loaded or trained.")

        vec = self.embedder.transform([text])

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
        """Saves both the classifier and the embedder state"""
        base_path = self.get_model_path()
        os.makedirs(base_path, exist_ok=True)
        
        joblib.dump(self.classifier, os.path.join(base_path, "classifier.joblib"))

        if self.scaler:
            joblib.dump(self.scaler, os.path.join(base_path, "scaler.joblib"))
        
        self.embedder.save(os.path.join(base_path, "embedder.pkl"))
        print(f"\nModel saved at: {base_path}")

    def load(self):
        """Loads both the classifier and the embedder state"""
        base_path = self.get_model_path()
        
        self.classifier = joblib.load(os.path.join(base_path, "classifier.joblib"))

        scaler_path = os.path.join(base_path, "scaler.joblib")
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
        else:
            self.scaler = None

        self.embedder.load(os.path.join(base_path, "embedder.pkl"))
        print(f"Model loaded from: {base_path}")