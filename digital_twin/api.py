"""FastAPI routes for the digital twin dashboard."""

from __future__ import annotations

import json
import logging
import os

import pandas as pd
from fastapi import APIRouter, HTTPException

from .config import ALERTS_PATH, PARAMS_PATH, SENSORS_PATH, WEIGHTS_PATH
from .model.inference import _load_model_for_inference, predict
from .pipeline import digital_twin_node
from .schemas import SensorReading

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/digital-twin/predict")
def predict_failure(data: SensorReading) -> dict:
    try:
        return predict(data.model_dump())
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed") from exc


@router.get("/digital-twin/history")
def get_sensor_history() -> list[dict]:
    if not os.path.exists(SENSORS_PATH):
        raise HTTPException(404, "No sensor data")
    return pd.read_csv(SENSORS_PATH).to_dict(orient="records")


@router.get("/digital-twin/alerts")
def get_alert_history() -> list:
    if not os.path.exists(ALERTS_PATH):
        return []
    with open(ALERTS_PATH, encoding="utf-8") as handle:
        return json.load(handle)


@router.get("/digital-twin/model-info")
def get_model_info() -> dict:
    if not os.path.exists(WEIGHTS_PATH):
        raise HTTPException(404, "No model trained")
    return _load_model_for_inference()["metadata"]


@router.get("/digital-twin/operating-params")
def get_operating_params() -> dict:
    if not os.path.exists(PARAMS_PATH):
        raise HTTPException(404, "No params extracted")
    with open(PARAMS_PATH, encoding="utf-8") as handle:
        return json.load(handle)


__all__ = ["router", "digital_twin_node"]
