"""Synthetic sensor data fallback."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from ..config import OUTPUT_DIR, PARAMS_PATH, SENSORS_PATH

logger = logging.getLogger(__name__)


def generate_sensor_dataset(
    params: dict, n_normal: int = 800, n_warning: int = 150, n_critical: int = 50
) -> pd.DataFrame:
    """Generate a calibrated, labeled sensor dataset."""
    rng = np.random.default_rng(42)
    t_max, t_min = params["max_temperature_c"], params["min_temperature_c"]
    p_max, p_min = params["max_pressure_bar"], params["min_pressure_bar"]
    v_max, i_max, w_max = params["max_vibration_mm_s"], params["max_current_a"], params["rated_power_kw"]
    rows: list[dict] = []
    base_time = datetime(2024, 1, 1)

    def add_rows(count: int, regime: str) -> None:
        for _ in range(count):
            timestamp = base_time + timedelta(hours=len(rows))
            if regime == "Normal":
                values = rng.normal(
                    (t_min + (t_max - t_min) * 0.55, p_min + (p_max - p_min) * 0.65,
                     v_max * 0.3, i_max * 0.7, w_max * 0.72),
                    ((t_max - t_min) * 0.04, (p_max - p_min) * 0.03,
                     v_max * 0.08, i_max * 0.05, w_max * 0.04),
                )
            elif regime == "Warning":
                values = (
                    rng.uniform(t_max * 0.82, t_max * 0.99),
                    rng.uniform(p_max * 0.85, p_max * 0.98) if rng.random() > 0.5
                    else rng.normal(p_min + (p_max - p_min) * 0.65, (p_max - p_min) * 0.03),
                    rng.uniform(v_max * 0.80, v_max * 0.99),
                    rng.uniform(i_max * 0.85, i_max * 0.98),
                    rng.uniform(w_max * 0.88, w_max),
                )
            else:
                values = (
                    rng.uniform(t_max, t_max * 1.25),
                    rng.uniform(p_max, p_max * 1.30) if rng.random() > 0.4
                    else rng.normal(p_min + (p_max - p_min) * 0.65, (p_max - p_min) * 0.03),
                    rng.uniform(v_max, v_max * 1.50),
                    rng.uniform(i_max, i_max * 1.20),
                    rng.uniform(w_max * 1.05, w_max * 1.30),
                )
            temperature, pressure, vibration, current, power = values
            rows.append({
                "Timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "Temperature_C": round(float(np.clip(temperature, t_min - 5, t_max * 1.5)), 2),
                "Pressure_bar": round(float(np.clip(pressure, 0, p_max * 1.5)), 3),
                "Vibration_mm_s": round(float(np.clip(vibration, 0, v_max * 2)), 3),
                "Current_A": round(float(np.clip(current, 0, i_max * 1.5)), 2),
                "Power_kW": round(float(np.clip(power, 0, w_max * 1.5)), 3),
                "Status": regime,
            })

    add_rows(n_normal, "Normal")
    add_rows(n_warning, "Warning")
    add_rows(n_critical, "Critical")
    frame = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    frame.to_csv(SENSORS_PATH, index=False)
    params["data_source"] = "synthetic_fallback"
    with open(PARAMS_PATH, "w", encoding="utf-8") as handle:
        json.dump(params, handle, indent=2)
    return frame
