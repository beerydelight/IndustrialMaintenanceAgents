"""Deterministic digital-twin tool exposed to the supervisor."""

from __future__ import annotations

import re
from typing import Any

from .model.inference import predict
from .pipeline import digital_twin_node


def _extract_readings(alert_text: str) -> dict[str, float]:
    """Extract only numeric sensor inputs understood by the model."""
    patterns = {
        "temperature_c": r"(?:temperature|temp)(?:\s+(?:reached|is|was))?\s*([0-9]+(?:\.[0-9]+)?)\s*°?\s*c\b",
        "pressure_bar": r"pressure(?:\s+(?:fell to|is|was|reached))?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:bar)\b",
        "vibration_mm_s": r"vibration(?:\s+(?:amplitude|level))?(?:\s+(?:is|was|reached))?\s*([0-9]+(?:\.[0-9]+)?)\s*mm/s\b",
        "current_a": r"(?:motor\s+)?current(?:\s+(?:rose to|is|was|reached))?\s*([0-9]+(?:\.[0-9]+)?)\s*a\b",
        "power_kw": r"(?:power|load)(?:\s+(?:is|was|reached))?\s*([0-9]+(?:\.[0-9]+)?)\s*kw\b",
        "rotational_speed_rpm": r"(?:rotational\s+speed|speed)(?:\s+(?:is|was|reached))?\s*([0-9]+(?:\.[0-9]+)?)\s*rpm\b",
        "tool_wear_min": r"tool\s+wear(?:\s+(?:is|was|reached))?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:min|minutes?)\b",
    }
    readings: dict[str, float] = {}
    for field, pattern in patterns.items():
        match = re.search(pattern, alert_text, re.IGNORECASE)
        if match:
            readings[field] = float(match.group(1))
    return readings


def digital_twin_tool(input_data: str | dict[str, Any]) -> dict[str, Any]:
    """Run the deterministic train/infer/rules pipeline without an LLM call."""
    if isinstance(input_data, str):
        if not input_data.strip():
            raise ValueError("digital twin input must not be empty")
        alert_text = input_data.strip()
        readings = _extract_readings(alert_text)
    elif isinstance(input_data, dict):
        alert_text = str(input_data.get("alert_text", ""))
        readings = {
            key: float(value)
            for key, value in input_data.items()
            if key != "alert_text" and value is not None
        }
    else:
        raise TypeError("digital twin input must be alert text or sensor readings")

    trained = digital_twin_node(alert_text, deterministic=True)
    prediction = predict(readings)
    return {
        "pipeline": "deterministic_ml_rules",
        "readings": readings,
        "training": trained["digital_twin_state"],
        "prediction": prediction,
    }


__all__ = ["digital_twin_tool"]
