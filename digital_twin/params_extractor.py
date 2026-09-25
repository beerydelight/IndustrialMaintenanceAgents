"""Operating parameter extraction."""

from __future__ import annotations

import json
import logging
import os
import re

from .config import DEFAULT_PARAMS, OUTPUT_DIR, PARAMS_PATH

logger = logging.getLogger(__name__)


def _psi_to_bar(value: float) -> float:
    return round(value * 0.0689476, 2)


def _f_to_c(value: float) -> float:
    return round((value - 32) * 5 / 9, 1)


def _k_to_c(value: float) -> float:
    return round(value - 273.15, 1)


def extract_operating_params(text: str) -> dict:
    """Extract operating limits from text and persist the resulting parameters."""
    params = dict(DEFAULT_PARAMS)
    params["source"] = "pdf_regex"
    for match in re.finditer(r"(\d+\.?\d*)\s*°C", text):
        value = float(match.group(1))
        if 10 < value < 300:
            params["max_temperature_c"] = max(params["max_temperature_c"], value)
    for match in re.finditer(r"(\d+\.?\d*)\s*°F", text):
        value = _f_to_c(float(match.group(1)))
        if 10 < value < 300:
            params["max_temperature_c"] = max(params["max_temperature_c"], value)
    pressures = (
        [_psi_to_bar(float(m.group(1))) for m in re.finditer(r"(\d+\.?\d*)\s*psi", text, re.I)
         if 1 < float(m.group(1)) < 5000]
        + [float(m.group(1)) for m in re.finditer(r"(\d+\.?\d*)\s*bar", text, re.I)
           if 0.1 < float(m.group(1)) < 500]
        + [float(m.group(1)) / 100 for m in re.finditer(r"(\d+\.?\d*)\s*kPa", text, re.I)
           if 10 < float(m.group(1)) < 50000]
        + [float(m.group(1)) * 10 for m in re.finditer(r"(\d+\.?\d*)\s*MPa", text, re.I)
           if 0.01 < float(m.group(1)) < 50]
    )
    if pressures:
        params["max_pressure_bar"] = max(pressures)
        params["min_pressure_bar"] = max(0.1, min(pressures) * 0.5)
    for match in re.finditer(r"(\d+\.?\d*)\s*V\b", text, re.I):
        value = float(match.group(1))
        if 100 < value < 1000:
            params["voltage_v"] = value
    amps = [float(m.group(1)) for m in re.finditer(r"(\d+\.?\d*)\s*A\b", text, re.I)
            if 0.5 < float(m.group(1)) < 1000]
    if amps:
        params["max_current_a"] = max(amps)
    watts = [float(m.group(1)) for m in re.finditer(r"(\d+\.?\d*)\s*kW\b", text, re.I)
             if 0.1 < float(m.group(1)) < 2000]
    if watts:
        params["rated_power_kw"] = max(watts)
    for match in re.finditer(r"(\d+\.?\d*)\s*dB", text, re.I):
        value = float(match.group(1))
        if 50 < value < 130:
            params["max_vibration_mm_s"] = round(10 ** ((value - 60) / 40), 1)
    for match in re.finditer(r"(\d+\.?\d*)\s*mm/s", text, re.I):
        value = float(match.group(1))
        if 0.1 < value < 200:
            params["max_vibration_mm_s"] = max(params["max_vibration_mm_s"], value)
    rpms = [float(m.group(1)) for m in re.finditer(r"(\d+)\s*rpm", text, re.I)
            if 100 < float(m.group(1)) < 10000]
    if rpms:
        params["rated_rpm"] = max(rpms)
        params["max_vibration_mm_s"] = max(params["max_vibration_mm_s"], 7.1)
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(PARAMS_PATH, "w", encoding="utf-8") as handle:
            json.dump(params, handle, indent=2)
    except OSError:
        logger.exception("Unable to persist extracted operating parameters")
        raise
    logger.info("Operating parameters extracted: %s", params)
    return params
