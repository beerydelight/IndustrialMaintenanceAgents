"""Neural network architecture."""

from __future__ import annotations

import torch
import torch.nn as nn


class DynamicMaintModel(nn.Module):
    """Small feed-forward maintenance classifier sized to selected features."""

    def __init__(self, n_features: int) -> None:
        super().__init__()
        h1, h2 = max(16, n_features * 4), max(8, n_features * 2)
        self.net = nn.Sequential(
            nn.Linear(n_features, h1), nn.BatchNorm1d(h1), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(h1, h2), nn.ReLU(), nn.Linear(h2, 1), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
