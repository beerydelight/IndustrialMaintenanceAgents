"""Full digital twin training pipeline."""

from __future__ import annotations

import logging
from typing import Any

from .config import SENSORS_PATH, WEIGHTS_PATH
from .data_ingestion import fetch_kaggle_dataset, generate_sensor_dataset
from .feature_engineering import auto_select_features
from .model.inference import invalidate_model_cache
from .model.trainer import build_and_train_model
from .params_extractor import extract_operating_params

logger = logging.getLogger(__name__)


def digital_twin_node(extracted_text: str, *, deterministic: bool = False) -> dict[str, Any]:
    """Extract, ingest, select features, train, and return a summary."""
    logger.info("Executing digital twin pipeline")
    params = extract_operating_params(extracted_text)
    frame = None if deterministic else fetch_kaggle_dataset(params)
    if frame is None:
        frame = generate_sensor_dataset(params)
    features = auto_select_features(frame)
    checkpoint = build_and_train_model(frame, features, deterministic=deterministic)
    
    invalidate_model_cache()
    
    return {"digital_twin_state": {
        "operating_params": params, "data_source": params.get("data_source", "unknown"),
        "sensor_dataset_path": SENSORS_PATH, "features_used": features,
        "feature_importance": checkpoint["importance"], "model_path": WEIGHTS_PATH,
        "train_rows": checkpoint["train_rows"], "best_loss": checkpoint["best_loss"],
    }}
