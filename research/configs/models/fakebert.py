import json
import os
from typing import Any, Dict, Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput

from research.configs.models.transformer_base import TransformerBaseModel


class FakeBertCNNClassifier(nn.Module):
    """BERT encoder followed by parallel Conv1D blocks for fake-news detection."""

    def __init__(
        self,
        pretrained_model_name: str,
        num_labels: int = 2,
        num_filters: int = 128,
        kernel_sizes: Iterable[int] = (3, 4, 5),
        dense_units: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        hidden_size = self.bert.config.hidden_size
        self.kernel_sizes: Tuple[int, ...] = tuple(int(k) for k in kernel_sizes)

        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=hidden_size,
                    out_channels=num_filters,
                    kernel_size=kernel_size,
                )
                for kernel_size in self.kernel_sizes
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(num_filters * len(self.kernel_sizes), dense_units)
        self.classifier = nn.Linear(dense_units, num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask=None, labels=None):
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = bert_outputs.last_hidden_state.transpose(1, 2)

        max_kernel = max(self.kernel_sizes)
        if x.shape[-1] < max_kernel:
            x = F.pad(x, (0, max_kernel - x.shape[-1]))

        pooled_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(x))
            pooled_outputs.append(torch.max(conv_out, dim=2).values)

        features = torch.cat(pooled_outputs, dim=1)
        features = self.dropout(features)
        features = F.relu(self.dense(features))
        features = self.dropout(features)
        logits = self.classifier(features)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)


class FakeBertModel(TransformerBaseModel):
    """FakeBERT-style model: BERT contextual embeddings with a CNN classifier."""

    DEFAULT_PRETRAINED = "bert-base-uncased"

    def __init__(self, dataset_name: str, embedding_type: str = "bert-base-uncased", **kwargs):
        self.num_filters = int(kwargs.get("num_filters", 128))
        self.cnn_kernel_sizes = tuple(int(k) for k in kwargs.get("cnn_kernel_sizes", [3, 4, 5]))
        self.dense_units = int(kwargs.get("dense_units", 128))
        super().__init__(
            dataset_name=dataset_name,
            model_name=f"fakebert_{embedding_type}",
            embedding_type=embedding_type,
            **kwargs,
        )

    def _build_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name)
        self.transformer_model = FakeBertCNNClassifier(
            pretrained_model_name=self.pretrained_model_name,
            num_labels=2,
            num_filters=self.num_filters,
            kernel_sizes=self.cnn_kernel_sizes,
            dense_units=self.dense_units,
            dropout=self.classifier_dropout,
        ).to(self.device)

        self._freeze_encoder_layers(self.freeze_layers)

    def _freeze_encoder_layers(self, n: int):
        if n <= 0:
            return

        layers = self.transformer_model.bert.encoder.layer
        n_clamped = min(n, len(layers))
        for layer in layers[:n_clamped]:
            for param in layer.parameters():
                param.requires_grad = False

        print(f"[Freeze] Froze {n_clamped}/{len(layers)} BERT encoder layers.")

    def _model_config(self) -> Dict[str, Any]:
        return {
            "pretrained_model_name": self.pretrained_model_name,
            "num_filters": self.num_filters,
            "cnn_kernel_sizes": list(self.cnn_kernel_sizes),
            "dense_units": self.dense_units,
            "classifier_dropout": self.classifier_dropout,
        }

    def _restore_from_config(self, config: Dict[str, Any]):
        self.pretrained_model_name = config["pretrained_model_name"]
        self.num_filters = int(config["num_filters"])
        self.cnn_kernel_sizes = tuple(int(k) for k in config["cnn_kernel_sizes"])
        self.dense_units = int(config["dense_units"])
        self.classifier_dropout = float(config["classifier_dropout"])

    def _save_model_artifacts(self, model_dir: str):
        os.makedirs(model_dir, exist_ok=True)
        self.tokenizer.save_pretrained(os.path.join(model_dir, "tokenizer"))
        torch.save(
            {
                "model_state_dict": self.transformer_model.state_dict(),
                "config": self._model_config(),
            },
            os.path.join(model_dir, "fakebert_cnn.pt"),
        )

    def _load_model_artifacts(self, model_dir: str):
        checkpoint_path = os.path.join(model_dir, "fakebert_cnn.pt")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"FakeBERT CNN checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self._restore_from_config(checkpoint["config"])
        self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
        self.transformer_model = FakeBertCNNClassifier(
            pretrained_model_name=self.pretrained_model_name,
            num_labels=2,
            num_filters=self.num_filters,
            kernel_sizes=self.cnn_kernel_sizes,
            dense_units=self.dense_units,
            dropout=self.classifier_dropout,
        ).to(self.device)
        self.transformer_model.load_state_dict(checkpoint["model_state_dict"])

    def _save_checkpoint(self, epoch, optimizer, scheduler, grad_scaler, epoch_loss):
        ckpt_path = os.path.join(self.checkpoint_dir, "ckpt_latest")
        self._save_model_artifacts(ckpt_path)
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
        with open(os.path.join(ckpt_path, "meta.json"), "w") as f:
            json.dump(
                {
                    "epoch": epoch,
                    "num_epochs": self.num_epochs,
                    "epoch_loss": epoch_loss,
                    "model_name": self.model_name,
                    "dataset_name": self.dataset_name,
                    **self._model_config(),
                },
                f,
                indent=2,
            )
        print(f"Checkpoint saved (epoch {epoch + 1}/{self.num_epochs}) -> {ckpt_path}")
        return ckpt_path

    def _load_checkpoint(self, optimizer, scheduler, grad_scaler) -> int:
        ckpt_path = os.path.join(self.checkpoint_dir, "ckpt_latest")
        state_file = os.path.join(ckpt_path, "training_state.pt")

        if not os.path.exists(state_file):
            return 0

        print(f"Resuming from checkpoint: {ckpt_path}")
        self._load_model_artifacts(ckpt_path)
        state = torch.load(state_file, map_location=self.device, weights_only=False)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        grad_scaler.load_state_dict(state["grad_scaler_state_dict"])

        resume_epoch = state["epoch"] + 1
        print(
            f"Resuming training from epoch {resume_epoch + 1}/{self.num_epochs} "
            f"(last completed epoch loss: {state['epoch_loss']:.4f})"
        )
        return resume_epoch

    def save(self):
        base_path = self.get_model_path()
        model_dir = os.path.join(base_path, "fakebert_cnn")
        print(f"\n[Saving FakeBERT CNN Model] {model_dir}")
        self._save_model_artifacts(model_dir)
        print("FakeBERT CNN model saved successfully!\n")

    def load(self):
        base_path = self.get_model_path()
        model_dir = os.path.join(base_path, "fakebert_cnn")
        self._load_model_artifacts(model_dir)
        print(f"FakeBERT CNN model loaded from: {model_dir}")
