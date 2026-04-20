import os
from abc import abstractmethod
from collections import Counter
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import optuna
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, TensorDataset

from research.configs.models.base_model import BaseModel
from research.configs.preprocessing.preprocessor import TextPreprocessor
from research.configs.embeddings import GLOVE_DEFAULT_PATH


class DLBaseModel(BaseModel):
    """
    Abstract base class for LSTM / GRU / BiLSTM / TextCNN.

    Sub-classes must implement ``_build_nn_model(embedding_matrix)``
    which returns a ``torch.nn.Module``.
    """

    # Class-level GloVe cache — loaded once, shared across all instances
    _glove_cache: Dict[str, np.ndarray] = {}

    def __init__(
        self,
        dataset_name: str,
        model_name: str,
        embedding_type: str,
        **kwargs,
    ):
        # Model named as "<arch>_<embedding>" to match the rf/svm/bert convention
        super().__init__(
            dataset_name,
            f"{model_name}_{embedding_type}",
            preprocessing_mode="classic",
        )

        self.embedding_type = embedding_type  # 'glove' | 'word2vec'

        # Hyper-parameters (Optuna-tunable)
        self.hidden_size: int = int(kwargs.get("hidden_size", 128))
        self.num_layers: int = int(kwargs.get("num_layers", 1))
        self.dropout: float = float(kwargs.get("dropout", 0.3))
        self.num_epochs: int = int(kwargs.get("num_epochs", 5))
        self.batch_size: int = int(kwargs.get("batch_size", 32))
        self.learning_rate: float = float(kwargs.get("learning_rate", 1e-3))
        self.max_seq_len: int = int(kwargs.get("max_seq_len", 200))
        self.embedding_dim: int = int(kwargs.get("embedding_dim", 100))
        self.early_stop_patience: int = int(kwargs.get("early_stop_patience", 2))
        # CNN-specific (ignored by recurrent models)
        self.num_filters: int = int(kwargs.get("num_filters", 128))
        self.kernel_sizes: List[int] = list(kwargs.get("kernel_sizes", [2, 3, 4]))

        # Runtime state (populated by train / load)
        self.vocab: Optional[List[str]] = None
        self.word2idx: Optional[Dict[str, int]] = None
        self.nn_model: Optional[nn.Module] = None

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        # Cached preprocessor
        self._preprocessor = TextPreprocessor(mode="classic")

        # BaseModel compatibility (not used by DL models)
        self.classifier = None
        self.scaler = None
        self.embedder = None

    # --- Vocabulary & sequence helpers ---

    def _build_vocab(self, texts: List[str], min_freq: int = 2) -> None:
        """Build a whitespace-split vocabulary from training texts."""
        counter: Counter = Counter()
        for text in texts:
            counter.update(str(text).lower().split())
        # Index 0 = PAD, 1 = UNK
        words = [w for w, c in counter.items() if c >= min_freq]
        self.vocab = ["<PAD>", "<UNK>"] + words
        self.word2idx = {w: i for i, w in enumerate(self.vocab)}

    def _texts_to_sequences(self, texts: List[str]) -> np.ndarray:
        """Convert a list of texts to a padded integer array."""
        seqs = []
        for text in texts:
            tokens = str(text).lower().split()[: self.max_seq_len]
            ids = [self.word2idx.get(t, 1) for t in tokens]
            pad_len = self.max_seq_len - len(ids)
            ids = ids + [0] * pad_len
            seqs.append(ids)
        return np.array(seqs, dtype=np.int64)

    # --- Embedding matrix ---

    def _load_word_vectors(self, texts: Optional[List[str]] = None) -> np.ndarray:
        """
        Build an embedding matrix aligned with ``self.vocab``.

        - ``glove``    → reads the GloVe .txt file (cached at class level).
        - ``word2vec`` → trains a Gensim Word2Vec model on *texts*.
        """
        vocab_size = len(self.vocab)
        embedding_matrix = np.zeros((vocab_size, self.embedding_dim), dtype=np.float32)

        if self.embedding_type == "glove":
            # Populate cache only on first call
            if not DLBaseModel._glove_cache:
                try:
                    with open(GLOVE_DEFAULT_PATH, encoding="utf8") as f:
                        for line in f:
                            values = line.split()
                            word = values[0]
                            coefs = np.asarray(values[1:], dtype="float32")
                            if len(coefs) == self.embedding_dim:
                                DLBaseModel._glove_cache[word] = coefs
                    print(f"[GloVe] Loaded {len(DLBaseModel._glove_cache):,} vectors.")
                except FileNotFoundError:
                    print(
                        f"[Warning] GloVe file not found at '{GLOVE_DEFAULT_PATH}'. "
                        "Using random embeddings."
                    )
            else:
                print(f"[GloVe] Using cached {len(DLBaseModel._glove_cache):,} vectors.")

            for word, idx in self.word2idx.items():
                if word in DLBaseModel._glove_cache:
                    embedding_matrix[idx] = DLBaseModel._glove_cache[word]

        elif self.embedding_type == "word2vec":
            from gensim.models import Word2Vec as _W2V

            tokenized = [str(t).lower().split() for t in (texts or [])]
            w2v = _W2V(
                sentences=tokenized,
                vector_size=self.embedding_dim,
                window=5,
                min_count=2,
                workers=4,
            )
            print(f"[Word2Vec] Trained on {len(tokenized):,} documents.")
            for word, idx in self.word2idx.items():
                if word in w2v.wv:
                    embedding_matrix[idx] = w2v.wv[word]
        else:
            print(
                f"[Warning] Unknown embedding type '{self.embedding_type}'. "
                "Using random embeddings."
            )

        return embedding_matrix

    # --- Abstract: architecture ---

    @abstractmethod
    def _build_nn_model(self, embedding_matrix: np.ndarray) -> nn.Module:
        """Return an initialised ``nn.Module`` for this architecture."""
        ...

    # --- Train / Evaluate / Predict ---

    def train(  # type: ignore[override]
        self,
        X_train_texts: List[str],
        y_train: List[int],
        trial=None,
        X_val: Optional[List[str]] = None,
        y_val: Optional[List[int]] = None,
    ) -> None:
        print(
            f"Training {self.model_name} on {self.device}  "
            f"(lr={self.learning_rate}, epochs={self.num_epochs}, "
            f"batch={self.batch_size}, max_seq={self.max_seq_len}, "
            f"hidden={self.hidden_size}, layers={self.num_layers}, "
            f"dropout={self.dropout}, emb={self.embedding_type}, "
            f"early_stop_patience={self.early_stop_patience})"
        )

        self._build_vocab(X_train_texts)
        embedding_matrix = self._load_word_vectors(texts=X_train_texts)
        self.nn_model = self._build_nn_model(embedding_matrix).to(self.device)

        X_seq = self._texts_to_sequences(X_train_texts)
        dataset = TensorDataset(
            torch.from_numpy(X_seq),
            torch.tensor(y_train, dtype=torch.long),
        )
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.nn_model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()

        best_val_f1 = -1.0
        patience_counter = 0

        for epoch in range(self.num_epochs):
            self.nn_model.train()
            total_loss = 0.0
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                logits = self.nn_model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.nn_model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            print(f"  Epoch {epoch + 1}/{self.num_epochs} — loss: {avg_loss:.4f}")

            # Optuna pruning / early stopping
            if trial is not None and X_val is not None and y_val is not None:
                val_metrics = self.evaluate(X_val, y_val)
                val_f1 = val_metrics["f1_score"]
                print(f"  Epoch {epoch + 1}/{self.num_epochs} — val_f1: {val_f1:.4f}")
                trial.report(val_f1, epoch)
                if trial.should_prune():
                    print(f"  Trial pruned at epoch {epoch + 1}")
                    raise optuna.TrialPruned()

                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stop_patience:
                        print(
                            f"  Early stopping at epoch {epoch + 1} "
                            f"(no improvement for {self.early_stop_patience} epoch(s))"
                        )
                        break

    def evaluate(  # type: ignore[override]
        self, X_test_texts: List[str], y_test: List[int]
    ) -> Dict[str, Any]:
        y_pred, _ = self._predict_batch(X_test_texts)
        y_test_arr = np.array(y_test)
        return {
            "accuracy": accuracy_score(y_test_arr, y_pred),
            "precision": precision_score(y_test_arr, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_test_arr, y_pred, average="weighted", zero_division=0),
            "f1_score": f1_score(y_test_arr, y_pred, average="weighted", zero_division=0),
            "confusion_matrix": confusion_matrix(y_test_arr, y_pred).tolist(),
        }

    def _predict_batch(self, texts: List[str]):
        """Run inference on a list of texts. Returns (predictions, full_probs)."""
        if self.nn_model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        self.nn_model.eval()
        X_seq = self._texts_to_sequences(texts)
        all_preds, all_probs = [], []

        with torch.no_grad():
            for i in range(0, len(X_seq), self.batch_size):
                batch = torch.from_numpy(X_seq[i : i + self.batch_size]).to(self.device)
                logits = self.nn_model(batch)
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())  # keep full (batch, num_classes)

        return np.array(all_preds), np.array(all_probs)

    def predict(self, text: str) -> Dict[str, Any]:  # type: ignore[override]
        processed = self._preprocessor.preprocess(text)
        preds, probs = self._predict_batch([processed])
        prediction = int(preds[0])
        score = float(probs[0][prediction])  # confidence of the chosen class
        return {
            "label": "REAL" if prediction == 1 else "FAKE",
            "score": score,
            "model": self.model_name,
            "dataset": self.dataset_name,
        }

    # Compatibility shims for the classic-ML interface used in train_models.py
    def train_on_vectors(self, X_texts: List[str], y_train: List[int]) -> None:  # type: ignore[override]
        self.train(X_texts, y_train)

    def evaluate_on_vectors(self, X_texts: List[str], y_test: List[int]) -> Dict[str, Any]:  # type: ignore[override]
        return self.evaluate(X_texts, y_test)

    def predict_proba_on_vectors(self, X_texts: List[str]) -> np.ndarray:  # type: ignore[override]
        preds, probs = self._predict_batch(X_texts)
        # Return positive-class (class=1) probabilities for sklearn-style compatibility
        return probs[:, 1]

    # --- Persistence ---

    def save(self) -> None:  # type: ignore[override]
        base_path = self.get_model_path()
        os.makedirs(base_path, exist_ok=True)
        print(f"\n[Saving DL Model] {base_path}")

        if self.nn_model is None:
            raise RuntimeError("Cannot save: nn_model is None (model not trained).")

        torch.save(self.nn_model.state_dict(), os.path.join(base_path, "model.pt"))

        meta = {
            "vocab": self.vocab,
            "word2idx": self.word2idx,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "max_seq_len": self.max_seq_len,
            "embedding_dim": self.embedding_dim,
            "embedding_type": self.embedding_type,
            "num_filters": self.num_filters,
            "kernel_sizes": self.kernel_sizes,
            "early_stop_patience": self.early_stop_patience,
        }
        joblib.dump(meta, os.path.join(base_path, "meta.joblib"))
        print("DL model saved successfully!\n")

    def load(self) -> None:  # type: ignore[override]
        base_path = self.get_model_path()
        meta_path = os.path.join(base_path, "meta.joblib")

        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"DL model meta not found: {meta_path}")

        meta = joblib.load(meta_path)
        self.vocab = meta["vocab"]
        self.word2idx = meta["word2idx"]
        self.hidden_size = meta["hidden_size"]
        self.num_layers = meta["num_layers"]
        self.dropout = meta["dropout"]
        self.max_seq_len = meta["max_seq_len"]
        self.embedding_dim = meta["embedding_dim"]
        self.embedding_type = meta["embedding_type"]
        self.num_filters = meta.get("num_filters", 128)
        self.kernel_sizes = meta.get("kernel_sizes", [2, 3, 4])
        self.early_stop_patience = meta.get("early_stop_patience", 2)

        vocab_size = len(self.vocab)
        dummy_emb = np.zeros((vocab_size, self.embedding_dim), dtype=np.float32)
        self.nn_model = self._build_nn_model(dummy_emb).to(self.device)
        self.nn_model.load_state_dict(
            torch.load(
                os.path.join(base_path, "model.pt"),
                map_location=self.device,
                weights_only=True,
            )
        )
        print(f"DL model loaded from: {base_path}")

    @property
    def is_transformer(self) -> bool:
        return False
