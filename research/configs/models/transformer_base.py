import json
import os
import shutil
from typing import List, Dict, Any, Optional

import numpy as np
import torch
from torch.amp import GradScaler, autocast
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    DataCollatorWithPadding,
)

from research.configs.models.base_model import BaseModel
from research.configs.preprocessing.preprocessor import TextPreprocessor


class PreTokenizedDataset(Dataset):
    """PyTorch Dataset with pre-tokenized inputs (tokenization happens once, not per-batch).

    Uses dynamic padding via DataCollatorWithPadding in the DataLoader,
    so sequences are only padded to the longest in each batch — not to max_length.
    """

    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128):
        # Pre-tokenize ALL texts once (no padding here — collator handles it per-batch)
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=False,  # dynamic padding handled by DataCollatorWithPadding
            max_length=max_length,
        )
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
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
        self.max_length = int(kwargs.get("max_length", 128))  # reduced from 512
        self.weight_decay = float(kwargs.get("weight_decay", 0.01))
        self.warmup_ratio = float(kwargs.get("warmup_ratio", 0.1))
        self.gradient_accumulation_steps = int(kwargs.get("gradient_accumulation_steps", 2))
        self.fp16 = bool(kwargs.get("fp16", False))
        self.save_checkpoints = bool(kwargs.get("save_checkpoints", True))
        self.num_workers = int(kwargs.get("num_workers", 2))
        # New tunable hyperparameters
        self.scheduler_type = str(kwargs.get("scheduler_type", "linear"))  # "linear" | "cosine"
        self.classifier_dropout = float(kwargs.get("classifier_dropout", 0.1))
        self.freeze_layers = int(kwargs.get("freeze_layers", 0))

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Enable cuDNN auto-tuner for faster convolutions on fixed-size inputs
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

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

        self._patch_classifier_dropout(self.classifier_dropout)
        self._freeze_encoder_layers(self.freeze_layers)

    def _patch_classifier_dropout(self, p: float):
        """Patch the classifier head's dropout probability in-place.

        Each HuggingFace architecture exposes its classifier dropout differently,
        so we detect the layout by inspecting the model and patch the nn.Dropout
        module directly rather than going through model.config.
        """
        m = self.transformer_model
        model_type = getattr(m.config, "model_type", "")

        # Candidate attribute paths, tried in order:
        # 1. BERT / RoBERTa-style   → model.dropout  (nn.Dropout before classifier Linear)
        # 2. DistilBERT-style        → model.classifier (Sequential whose [0] is Linear
        #                              and there is no standalone dropout before it, but
        #                              model.pre_classifier exists — dropout is inside
        #                              BertForSequenceClassification.dropout)
        # 3. RoBERTa classification  → model.classifier.dropout
        patched = False
        for attr_path in (
            "classifier.dropout",        # RoBERTa, DistilBERT
            "dropout",                   # BERT (BertForSequenceClassification)
        ):
            obj = m
            parts = attr_path.split(".")
            try:
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                target = getattr(obj, parts[-1])
                if isinstance(target, torch.nn.Dropout):
                    target.p = p
                    print(f"[Dropout] Patched '{attr_path}' → p={p:.3f} "
                          f"(model_type='{model_type}')")
                    patched = True
                    break
            except AttributeError:
                continue

        if not patched:
            print(f"[Dropout] WARNING: Could not find classifier head dropout "
                  f"for model_type='{model_type}'. Skipping dropout patch.")

    def _freeze_encoder_layers(self, n: int):
        """Freeze the first *n* encoder layers (0 = no freezing).

        Supports BERT/RoBERTa (model.bert.encoder.layer / model.roberta.encoder.layer)
        and DistilBERT (model.distilbert.transformer.layer).
        """
        if n <= 0:
            return

        m = self.transformer_model

        # Possible encoder layer list locations
        layer_list = None
        for attr_path in (
            "bert.encoder.layer",
            "roberta.encoder.layer",
            "distilbert.transformer.layer",
        ):
            obj = m
            try:
                for part in attr_path.split("."):
                    obj = getattr(obj, part)
                layer_list = obj
                break
            except AttributeError:
                continue

        if layer_list is None:
            print(f"[Freeze] WARNING: Could not locate encoder layers. "
                  f"No layers frozen.")
            return

        total = len(layer_list)
        n_clamped = min(n, total)
        for layer in layer_list[:n_clamped]:
            for param in layer.parameters():
                param.requires_grad = False

        print(f"[Freeze] Froze {n_clamped}/{total} encoder layers.")

    def _create_dataloader(self, texts, labels, shuffle=False):
        """Create a DataLoader with pre-tokenized dataset and dynamic padding."""
        dataset = PreTokenizedDataset(texts, labels, self.tokenizer, self.max_length)
        collator = DataCollatorWithPadding(tokenizer=self.tokenizer, return_tensors="pt")

        # num_workers > 0 can be problematic on Windows; fall back gracefully
        num_workers = self.num_workers
        persistent = num_workers > 0

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=persistent,
            collate_fn=collator,
        )

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
    def train(self, X_train_raw: List[str], y_train: List[int],
              trial=None, X_val: List[str] = None, y_val: List[int] = None) -> None:
        """Fine-tune the transformer on raw text.

        Automatically saves a checkpoint after every epoch (to Google Drive
        when running on Colab) and resumes from the latest checkpoint if one
        exists.

        Args:
            trial: Optional Optuna trial for pruning support. When provided
                   (together with X_val/y_val), the model reports intermediate
                   F1 after each epoch so Optuna can prune unpromising trials.
            X_val / y_val: Validation data for intermediate evaluation during
                   Optuna trials.
        """
        print(f"Training {self.model_name} on {self.device} "
              f"(lr={self.learning_rate}, epochs={self.num_epochs}, "
              f"batch={self.batch_size}, max_len={self.max_length}, "
              f"fp16={self.use_fp16}, checkpoints={self.save_checkpoints}, "
              f"num_workers={self.num_workers}, scheduler={self.scheduler_type}, "
              f"classifier_dropout={self.classifier_dropout}, "
              f"freeze_layers={self.freeze_layers}, "
              f"grad_accum={self.gradient_accumulation_steps})")

        # Pre-tokenize once (not per-batch)
        dataloader = self._create_dataloader(X_train_raw, y_train, shuffle=True)

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.transformer_model.parameters()),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        total_steps = (len(dataloader) // self.gradient_accumulation_steps) * self.num_epochs
        warmup_steps = int(total_steps * self.warmup_ratio)
        if self.scheduler_type == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )
        else:
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
        best_val_f1 = -1.0
        patience_counter = 0
        early_stop_patience = 1  # stop if no improvement for 1 epoch
        for epoch in range(start_epoch, self.num_epochs):
            epoch_loss = 0.0
            optimizer.zero_grad()
            for step, batch in enumerate(dataloader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

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

            # Optuna pruning: report intermediate metric after each epoch
            if trial is not None and X_val is not None and y_val is not None:
                val_metrics = self.evaluate(X_val, y_val)
                val_f1 = val_metrics["f1_score"]
                print(f"  Epoch {epoch + 1}/{self.num_epochs} — val_f1: {val_f1:.4f}")
                trial.report(val_f1, epoch)
                if trial.should_prune():
                    print(f"  Trial pruned at epoch {epoch + 1}")
                    raise __import__("optuna").TrialPruned()

                # Early stopping check
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stop_patience:
                        print(f"  Early stopping at epoch {epoch + 1} (no improvement for {early_stop_patience} epoch(s))")
                        break

            # Save checkpoint after each epoch
            if self.save_checkpoints:
                self._save_checkpoint(epoch, optimizer, scheduler, self.grad_scaler, avg_loss)

        # Training complete — remove checkpoint to save space
        if self.save_checkpoints:
            self._clear_checkpoint()

    def train_on_vectors(self, X_vec, y_train: List[int]) -> None:
        """Transformer models don't use pre-computed vectors — redirect to train()."""
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
        dataloader = self._create_dataloader(texts, [0] * len(texts), shuffle=False)

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
                logits = outputs.logits.float()
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
