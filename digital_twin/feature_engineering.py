"""Feature selection."""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd

from .config import FEATURE_EXCLUSIONS

logger = logging.getLogger(__name__)


def auto_select_features(df: pd.DataFrame) -> List[str]:
    """Select numeric features with useful variance and label correlation."""
    if df.empty or "Status" not in df.columns:
        raise ValueError("DataFrame must contain 'Status' column and not be empty")
    numeric = [column for column in df.select_dtypes(include=[np.number]).columns if column not in FEATURE_EXCLUSIONS]
    if not numeric:
        raise ValueError("DataFrame contains no numeric features")
    label = (df["Status"] == "Critical").astype(float)
    selected = [
        column for column in numeric
        if df[column].fillna(df[column].median()).var() >= 1e-6
        and abs(df[column].fillna(df[column].median()).corr(label)) >= 0.02
    ]
    return selected or numeric[:5]
