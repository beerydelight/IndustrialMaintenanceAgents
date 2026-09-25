"""Thread-safe model loading and inference."""

from __future__ import annotations

import json
import logging
import os
import threading

import torch

from ..alerts import generate_maintenance_alerts
from ..config import PROBABILITY_THRESHOLDS, STATS_PATH, WEIGHTS_PATH
from .architecture import DynamicMaintModel

logger = logging.getLogger(__name__)
_model_cache: dict | None = None
_model_lock = threading.Lock()

_FIELD_TO_COL = {
    "temperature_c": "Temperature_C", "process_temperature_c": "Process_Temperature_C",
    "pressure_bar": "Pressure_bar", "vibration_mm_s": "Vibration_mm_s", "current_a": "Current_A",
    "power_kw": "Power_kW", "rotational_speed_rpm": "Rotational_Speed_rpm", "torque_nm": "Torque_Nm",
    "tool_wear_min": "Tool_Wear_min",
}


def _load_model_for_inference() -> dict:
    global _model_cache
    if _model_cache is None:
        with _model_lock:
            if _model_cache is None:
                if not os.path.exists(WEIGHTS_PATH):
                    raise RuntimeError("No model found. Run the pipeline first.")
                checkpoint = torch.load(WEIGHTS_PATH, weights_only=False)
                model = DynamicMaintModel(checkpoint["n_features"])
                model.load_state_dict(checkpoint["state_dict"])
                model.eval()
                _model_cache = {
                    "model": model, "features": checkpoint["features"],
                    "x_min": torch.tensor(checkpoint["x_min"]), "x_max": torch.tensor(checkpoint["x_max"]),
                    "metadata": {"trained_on": checkpoint.get("trained_on", "?"), "train_rows": checkpoint.get("train_rows", 0),
                                 "best_loss": checkpoint.get("best_loss"), "n_features": checkpoint["n_features"],
                                 "features": checkpoint["features"], "importance": checkpoint.get("importance", {})},
                }
    return _model_cache


def invalidate_model_cache() -> None:
    global _model_cache
    with _model_lock:
        _model_cache = None


def _load_stats() -> dict:
    if not os.path.exists(STATS_PATH):
        return {}
    try:
        with open(STATS_PATH, encoding="utf-8") as handle:
            content = handle.read().strip()
        return json.loads(content) if content else {}
    except (OSError, json.JSONDecodeError):
        logger.warning("Unable to load training statistics")
        return {}


def predict(readings: dict) -> dict:
    """Predict failure while imputing every missing model feature."""
    cache = _load_model_for_inference()
    features, stats = cache["features"], _load_stats()
    medians = stats.get("medians", {})
    provided = {_FIELD_TO_COL[key]: float(value) for key, value in readings.items()
                if key in _FIELD_TO_COL and value is not None}
    missing = [feature for feature in features if feature not in provided]
    fallback = sum(medians.values()) / len(medians) if medians else 0.0
    values = [provided.get(feature, float(medians.get(feature, fallback))) for feature in features]
    raw = torch.tensor([values], dtype=torch.float32)
    normalized = (raw - cache["x_min"]) / (cache["x_max"] - cache["x_min"]).clamp(min=1e-6)
    with torch.no_grad():
        probability = float(cache["model"](normalized).item())
    if probability > PROBABILITY_THRESHOLDS["critical"]:
        status = "Critical"
    elif probability > PROBABILITY_THRESHOLDS["warning"]:
        status = "Warning"
    else:
        status = "Normal"
    alert_readings = {**provided, **{feature: medians.get(feature, 0.0) for feature in missing}}
    return {
        "failure_probability": round(probability, 4), "status": status,
        "alerts": generate_maintenance_alerts(alert_readings, probability, status, features),
        "features_used": features, "features_supplied": list(provided), "features_imputed": missing,
    }
