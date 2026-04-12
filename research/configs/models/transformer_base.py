import os
from typing import List, Dict, Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

from research.configs.models.base_model import BaseModel
from research.configs.preprocessing.preprocessor import TextPreprocessor


class TextClassificationDataset(Dataset):
    """PyTorch Dataset for transformer text classification."""

    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class TransformerBaseModel(BaseModel):
    """
    Base class for HuggingFace transformer models (BERT, RoBERTa, etc.).

    Overrides the sklearn-centric BaseModel methods to work with PyTorch
    fine-tuning instead of the embedder → classifier pipeline.
    """

    # Subclasses set this to the default HuggingFace checkpoint name
    DEFAULT_PRETRAINED: str = ""

    def __init__(self, dataset_name: str, model_name: str, embedding_type: str, **kwargs):
        super().__init__(dataset_name, model_name, preprocessing_mode="bert")

        # The embedding_type for transformers is the pretrained checkpoint name
        self.pretrained_model_name = embedding_type or self.DEFAULT_PRETRAINED

        # Training hyper-parameters (Optuna-tunable) — pulled from kwargs with defaults
        self.learning_rate = float(kwargs.get("learning_rate", 2e-5))
        self.num_epochs = int(kwargs.get("num_epochs", 3))
        self.batch_size = int(kwargs.get("batch_size", 16))
        self.max_length = int(kwargs.get("max_length", 512))
        self.weight_decay = float(kwargs.get("weight_decay", 0.01))
        self.warmup_ratio = float(kwargs.get("warmup_ratio", 0.1))
        self.gradient_accumulation_steps = int(kwargs.get("gradient_accumulation_steps", 1))

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Will be initialised in _build_model()
        self.tokenizer = None
        self.transformer_model = None

        # The base class attributes are not used by transformers
        self.embedder = None
        self.classifier = None
        self.scaler = None

        self._build_model()

    def _build_model(self):
        """Load pretrained tokenizer and classification head."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name)
        self.transformer_model = AutoModelForSequenceClassification.from_pretrained(
            self.pretrained_model_name,
            num_labels=2,
        ).to(self.device)

    def train(self, X_train_raw: List[str], y_train: List[int]) -> None:
        """Fine-tune the transformer on raw text."""
        print(f"Training {self.model_name} on {self.device} "
              f"(lr={self.learning_rate}, epochs={self.num_epochs}, "
              f"batch={self.batch_size}, max_len={self.max_length})")

        dataset = TextClassificationDataset(
            X_train_raw, y_train, self.tokenizer, self.max_length
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )

        optimizer = torch.optim.AdamW(
            self.transformer_model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        total_steps = (len(dataloader) // self.gradient_accumulation_steps) * self.num_epochs
        warmup_steps = int(total_steps * self.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        self.transformer_model.train()
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            optimizer.zero_grad()
            for step, batch in enumerate(dataloader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = self.transformer_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss / self.gradient_accumulation_steps
                loss.backward()
                epoch_loss += loss.item() * self.gradient_accumulation_steps

                if (step + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.transformer_model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            avg_loss = epoch_loss / len(dataloader)
            print(f"  Epoch {epoch + 1}/{self.num_epochs} — loss: {avg_loss:.4f}")

    def train_on_vectors(self, X_vec, y_train: List[int]) -> None:
        """Transformer models don't use pre-computed vectors — redirect to train()."""
        # X_vec is actually raw text when called from the transformer branch
        self.train(X_vec, y_train)

    def evaluate(self, X_test_raw: List[str], y_test: List[int]) -> Dict[str, float]:
        """Evaluate on raw text and return metrics dict."""
        y_pred, _ = self._predict_batch(X_test_raw)
        y_test = np.array(y_test)

        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "f1_score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }

    def evaluate_on_vectors(self, X_vec, y_test: List[int]) -> Dict[str, float]:
        """Redirect to text-based evaluate()."""
        return self.evaluate(X_vec, y_test)


    def _predict_batch(self, texts: List[str]):
        """Run inference on a list of texts. Returns (predictions, probabilities)."""
        dataset = TextClassificationDataset(
            texts, [0] * len(texts), self.tokenizer, self.max_length
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        self.transformer_model.eval()
        all_preds = []
        all_probs = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                outputs = self.transformer_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(logits, dim=-1)

                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())

        return np.array(all_preds), np.array(all_probs)

    def predict(self, text: str) -> Dict[str, Any]:
        """Predict the label for a single string of text."""
        preprocessor = TextPreprocessor(mode=self.preprocessing_mode)
        processed_text = preprocessor.preprocess(text)

        preds, probs = self._predict_batch([processed_text])
        prediction = int(preds[0])
        score = float(probs[0]) if prediction == 1 else float(1.0 - probs[0])

        return {
            "label": "REAL" if prediction == 1 else "FAKE",
            "score": score,
            "model": self.model_name,
            "dataset": self.dataset_name,
        }

    def predict_proba_on_vectors(self, X_texts) -> np.ndarray:
        """Return positive-class probabilities for a list of texts."""
        _, probs = self._predict_batch(X_texts)
        return probs

    def save(self):
        """Save the fine-tuned transformer and tokenizer."""
        base_path = self.get_model_path()
        os.makedirs(base_path, exist_ok=True)
        print(f"\n[Saving Transformer Model] {base_path}")

        model_dir = os.path.join(base_path, "transformer")
        self.transformer_model.save_pretrained(model_dir)
        self.tokenizer.save_pretrained(model_dir)

        print(f"Transformer model saved successfully!\n")

    def load(self):
        """Load a fine-tuned transformer and tokenizer."""
        base_path = self.get_model_path()
        model_dir = os.path.join(base_path, "transformer")

        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Transformer model not found: {model_dir}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.transformer_model = AutoModelForSequenceClassification.from_pretrained(
            model_dir,
            num_labels=2,
        ).to(self.device)

        print(f"Transformer model loaded from: {model_dir}")

    @property
    def is_transformer(self) -> bool:
        return True
