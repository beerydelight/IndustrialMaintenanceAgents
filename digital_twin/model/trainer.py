"""Model training and checkpoint persistence."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ..config import STATS_PATH, WEIGHTS_PATH
from .architecture import DynamicMaintModel

logger = logging.getLogger(__name__)


def build_and_train_model(
    df: pd.DataFrame, features: List[str], *, deterministic: bool = False
) -> dict:
    """Train the classifier and save a JSON-safe metadata checkpoint."""
    if not features:
        raise ValueError("At least one feature is required")
    torch.manual_seed(42)
    x_raw = df[features].fillna(df[features].median()).values.astype(np.float32)
    y_raw = (df["Status"] == "Critical").astype(np.float32).values
    x, y = torch.tensor(x_raw), torch.tensor(y_raw).unsqueeze(1)
    x_min, x_max = x.min(0).values, x.max(0).values
    x_norm = (x - x_min) / (x_max - x_min).clamp(min=1e-6)
    positives = y_raw.sum()
    model = DynamicMaintModel(len(features))
    model.net[-1] = nn.Identity()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(len(y_raw) - positives) / max(positives, 1)]))
    best_loss, patience = float("inf"), 0
    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        loss = criterion(model(x_norm), y)
        loss.backward()
        optimizer.step()
        if loss.item() < best_loss - 1e-4:
            best_loss, patience = loss.item(), 0
        else:
            patience += 1
        if patience >= 40:
            logger.info("Early stop at epoch %d, loss=%.4f", epoch, best_loss)
            break
    model.net[-1] = nn.Sigmoid()
    model.eval()
    x_norm.requires_grad_(True)
    model(x_norm).mean().backward()
    x_min_list, x_max_list = x_min.tolist(), x_max.tolist()
    importance = {feature: round(float(value), 4) for feature, value in zip(features, x_norm.grad.abs().mean(0).detach().tolist())}
    stats = {
        "medians": {column: round(float(df[column].median()), 4) for column in features},
        "stds": {column: round(float(df[column].std()), 4) for column in features},
        "x_min": x_min_list, "x_max": x_max_list, "feature_names": features,
    }
    checkpoint = {
        "state_dict": model.state_dict(), "n_features": len(features), "features": features,
        "x_min": x_min_list, "x_max": x_max_list,
        "trained_on": "deterministic-synthetic" if deterministic else datetime.now().isoformat(),
        "train_rows": len(df), "best_loss": round(float(best_loss), 5), "importance": importance,
    }
    os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
    torch.save(checkpoint, WEIGHTS_PATH)
    with open(STATS_PATH, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)
    return checkpoint
