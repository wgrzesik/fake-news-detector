import json
import os
import shutil
from typing import List, Dict, Any, Optional

import numpy as np
import torch
from torch.amp import GradScaler, autocast
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
        self.fp16 = bool(kwargs.get("fp16", False))
        self.save_checkpoints = bool(kwargs.get("save_checkpoints", True))

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Mixed Precision – only when GPU is available
        self.use_fp16 = self.fp16 and self.device.type == "cuda"
        self.grad_scaler = GradScaler(enabled=self.use_fp16)

        # Checkpoint directory (persisted to Google Drive via colab_setup symlinks)
        self._checkpoint_dir: Optional[str] = None

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

    # Checkpoint management (Google Drive persistence)

    @property
    def checkpoint_dir(self) -> str:
        """Return (and lazily create) the checkpoint directory."""
        if self._checkpoint_dir is None:
            self._checkpoint_dir = os.path.join(
                self.get_model_path(), "checkpoints"
            )
        os.makedirs(self._checkpoint_dir, exist_ok=True)
        return self._checkpoint_dir

    def _save_checkpoint(
        self,
        epoch: int,
        optimizer: torch.optim.Optimizer,
        scheduler,
        grad_scaler: GradScaler,
        epoch_loss: float,
    ) -> str:
        """Save a training checkpoint after an epoch.

        Saved to  saved_models/<dataset>/<model>/checkpoints/
        which is symlinked to Google Drive by colab_setup.py.
        """
        ckpt_path = os.path.join(self.checkpoint_dir, "ckpt_latest")
        os.makedirs(ckpt_path, exist_ok=True)

        # Model weights & tokenizer (HuggingFace format — easy to reload)
        self.transformer_model.save_pretrained(ckpt_path)
        self.tokenizer.save_pretrained(ckpt_path)

        # Optimizer, scheduler, scaler, meta
        torch.save(
            {
                "epoch": epoch,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "grad_scaler_state_dict": grad_scaler.state_dict(),
                "epoch_loss": epoch_loss,
            },
            os.path.join(ckpt_path, "training_state.pt"),
        )

        # Human-readable meta
        meta = {
            "epoch": epoch,
            "num_epochs": self.num_epochs,
            "epoch_loss": epoch_loss,
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "fp16": self.use_fp16,
        }
        with open(os.path.join(ckpt_path, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        print(f"Checkpoint saved (epoch {epoch + 1}/{self.num_epochs}) → {ckpt_path}")
        return ckpt_path

    def _load_checkpoint(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler,
        grad_scaler: GradScaler,
    ) -> int:
        """Try to resume from the latest checkpoint.

        Returns the epoch to resume from (0 if no checkpoint found).
        """
        ckpt_path = os.path.join(self.checkpoint_dir, "ckpt_latest")
        state_file = os.path.join(ckpt_path, "training_state.pt")

        if not os.path.exists(state_file):
            return 0  # no checkpoint — start from scratch

        print(f"Resuming from checkpoint: {ckpt_path}")

        # Restore model weights
        self.transformer_model = AutoModelForSequenceClassification.from_pretrained(
            ckpt_path, num_labels=2
        ).to(self.device)

        # Restore training state
        state = torch.load(state_file, map_location=self.device, weights_only=False)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        grad_scaler.load_state_dict(state["grad_scaler_state_dict"])

        resume_epoch = state["epoch"] + 1  # start from the NEXT epoch
        print(f"Resuming training from epoch {resume_epoch + 1}/{self.num_epochs} "
              f"(last completed epoch loss: {state['epoch_loss']:.4f})")
        return resume_epoch

    def _clear_checkpoint(self):
        """Remove checkpoint directory after training completes successfully."""
        ckpt_path = os.path.join(self.checkpoint_dir, "ckpt_latest")
        if os.path.exists(ckpt_path):
            shutil.rmtree(ckpt_path, ignore_errors=True)
            print(f"Checkpoint cleared: {ckpt_path}")

    # Training
    def train(self, X_train_raw: List[str], y_train: List[int]) -> None:
        """Fine-tune the transformer on raw text.

        Automatically saves a checkpoint after every epoch (to Google Drive
        when running on Colab) and resumes from the latest checkpoint if one
        exists.
        """
        print(f"Training {self.model_name} on {self.device} "
              f"(lr={self.learning_rate}, epochs={self.num_epochs}, "
              f"batch={self.batch_size}, max_len={self.max_length}, "
              f"fp16={self.use_fp16}, checkpoints={self.save_checkpoints})")

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

        # Try to resume from checkpoint
        start_epoch = 0
        if self.save_checkpoints:
            start_epoch = self._load_checkpoint(optimizer, scheduler, self.grad_scaler)

        if start_epoch >= self.num_epochs:
            print(f"All {self.num_epochs} epochs already completed (checkpoint). Skipping training.")
            return

        self.transformer_model.train()
        for epoch in range(start_epoch, self.num_epochs):
            epoch_loss = 0.0
            optimizer.zero_grad()
            for step, batch in enumerate(dataloader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                with autocast(device_type=self.device.type, enabled=self.use_fp16):
                    outputs = self.transformer_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss / self.gradient_accumulation_steps

                self.grad_scaler.scale(loss).backward()
                epoch_loss += loss.item() * self.gradient_accumulation_steps

                if (step + 1) % self.gradient_accumulation_steps == 0:
                    self.grad_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.transformer_model.parameters(), 1.0)
                    self.grad_scaler.step(optimizer)
                    self.grad_scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()

            avg_loss = epoch_loss / len(dataloader)
            print(f"  Epoch {epoch + 1}/{self.num_epochs} — loss: {avg_loss:.4f}")

            # Save checkpoint after each epoch
            if self.save_checkpoints:
                self._save_checkpoint(epoch, optimizer, scheduler, self.grad_scaler, avg_loss)

        # Training complete — remove checkpoint to save space
        if self.save_checkpoints:
            self._clear_checkpoint()

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

                with autocast(device_type=self.device.type, enabled=self.use_fp16):
                    outputs = self.transformer_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                logits = outputs.logits.float()  # ensure float32 for softmax
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
