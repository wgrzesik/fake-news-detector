import numpy as np
import torch
import torch.nn as nn

from research.configs.models.dl_base import DLBaseModel


class _GRUNet(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        embedding_matrix: np.ndarray,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding.weight = nn.Parameter(
            torch.tensor(embedding_matrix, dtype=torch.float)
        )
        self.gru = nn.GRU(
            embedding_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)               # (B, L, D)
        _, hidden = self.gru(emb)             # hidden: (layers, B, H)
        out = hidden[-1]                      # last layer: (B, H)
        out = self.dropout(out)
        return self.fc(out)                   # (B, 2)


class GRUModel(DLBaseModel):
    """Single-layer (or multi-layer) GRU classifier."""

    def __init__(self, dataset_name: str, embedding_type: str, **kwargs):
        super().__init__(
            dataset_name=dataset_name,
            model_name="gru",
            embedding_type=embedding_type,
            **kwargs,
        )

    def _build_nn_model(self, embedding_matrix: np.ndarray) -> nn.Module:
        return _GRUNet(
            vocab_size=len(self.vocab),
            embedding_dim=self.embedding_dim,
            embedding_matrix=embedding_matrix,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )
