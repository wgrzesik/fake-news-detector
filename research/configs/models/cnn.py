import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from research.configs.models.dl_base import DLBaseModel


class _TextCNNNet(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        embedding_matrix: np.ndarray,
        num_filters: int,
        kernel_sizes: List[int],
        dropout: float,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding.weight = nn.Parameter(
            torch.tensor(embedding_matrix, dtype=torch.float)
        )
        # One Conv1d per kernel size  (input channels = embedding_dim)
        self.convs = nn.ModuleList(
            [nn.Conv1d(embedding_dim, num_filters, k) for k in kernel_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)              # (B, L, D)
        emb = emb.transpose(1, 2)            # (B, D, L) — Conv1d expects (B, C_in, L)

        pooled = []
        for conv in self.convs:
            c = F.relu(conv(emb))            # (B, num_filters, L - k + 1)
            p = F.max_pool1d(c, c.size(2))   # (B, num_filters, 1)
            pooled.append(p.squeeze(2))      # (B, num_filters)

        out = torch.cat(pooled, dim=1)       # (B, num_filters * len(kernel_sizes))
        out = self.dropout(out)
        return self.fc(out)                  # (B, 2)


class CNNModel(DLBaseModel):
    """Text-CNN classifier (Kim 2014)."""

    def __init__(self, dataset_name: str, embedding_type: str, **kwargs):
        super().__init__(
            dataset_name=dataset_name,
            model_name="cnn",
            embedding_type=embedding_type,
            **kwargs,
        )

    def _build_nn_model(self, embedding_matrix: np.ndarray) -> nn.Module:
        return _TextCNNNet(
            vocab_size=len(self.vocab),
            embedding_dim=self.embedding_dim,
            embedding_matrix=embedding_matrix,
            num_filters=self.num_filters,
            kernel_sizes=self.kernel_sizes,
            dropout=self.dropout,
        )
