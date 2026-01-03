import os
import joblib
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from research.base import BaseFakeNewsModel
from research.embeddings.base_embedding import BaseEmbedder 

class NaiveBayesModel(BaseFakeNewsModel):
    def __init__(self, dataset_name: str, embedding_type: str):
        super().__init__(dataset_name, f"nb_{embedding_type}")

        self.embedding_type = embedding_type
        self.dataset_name = dataset_name
        self.embedder = BaseEmbedder.create(embedding_type)
        self.classifier = GaussianNB()
        self.scaler = StandardScaler()

    def train(self, X_train, y_train, X_val=None, y_val=None):
        print(f"Starting NB Training ({self.embedding_type}) for {self.dataset_name}")
        
        X_train_vec = self.embedder.fit_transform(X_train)
        X_train_scaled = self.scaler.fit_transform(X_train_vec)
        
        y_train = np.array(y_train).astype(int)
        
        print(f"Class distribution in training: {np.unique(y_train, return_counts=True)}")
        print("Fitting Naive Bayes model...")
        
        self.classifier.fit(X_train_scaled, y_train)
        print("Training completed successfully.")

    def evaluate(self, X_test, y_test):
        print(f"Evaluating model: {self.model_name}")
        
        X_test_vec = self.embedder.transform(X_test)
        X_test_scaled = self.scaler.transform(X_test_vec)
        
        y_pred = self.classifier.predict(X_test_scaled)
        
        y_test_idx = np.array(y_test).astype(int)
        y_pred_idx = np.array(y_pred).astype(int)

        print("Confusion Matrix:")
        print(confusion_matrix(y_test_idx, y_pred_idx))

        metrics = {
            "accuracy": accuracy_score(y_test_idx, y_pred_idx),
            "precision": precision_score(y_test_idx, y_pred_idx, average='weighted', zero_division=0),
            "recall": recall_score(y_test_idx, y_pred_idx, average='weighted', zero_division=0),
            "f1_score": f1_score(y_test_idx, y_pred_idx, average='weighted', zero_division=0)
        }
        
        for k, v in metrics.items():
            print(f"{k.capitalize()}: {v:.4f}")
            
        return metrics

    def predict(self, text: str):
        vec = self.embedder.transform([text])
        vec_scaled = self.scaler.transform(vec)
        
        prediction = self.classifier.predict(vec_scaled)[0]
        proba = self.classifier.predict_proba(vec_scaled)[0]
        
        label_str = "REAL" if prediction == 1 else "FAKE"
        score = max(proba)
        
        return {
            "label": label_str,
            "score": float(score),
            "model": self.model_name,
            "raw_prediction": int(prediction)
        }
    
    def save(self):
        save_path = self.get_model_path("") 
        os.makedirs(save_path, exist_ok=True)
        
        joblib.dump(self.classifier, os.path.join(save_path, "classifier.joblib"))
        joblib.dump(self.scaler, os.path.join(save_path, "scaler.joblib"))
        self.embedder.save(os.path.join(save_path, "embedder.pkl"))
        print(f"Model saved at: {save_path}")

    def load(self):
        save_path = self.get_model_path("")
        self.classifier = joblib.load(os.path.join(save_path, "classifier.joblib"))
        self.scaler = joblib.load(os.path.join(save_path, "scaler.joblib"))
        self.embedder.load(os.path.join(save_path, "embedder.pkl"))
        print(f"Model loaded from: {save_path}")