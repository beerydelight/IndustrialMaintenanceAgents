"""Configuration constants for the digital twin."""

from __future__ import annotations
from pathlib import Path
import json
import os


_BASE = Path.home() / "digital_twin_data"
OUTPUT_DIR = _BASE / "outputs"
SENSORS_PATH = OUTPUT_DIR / "digital_twin_sensors.csv"
WEIGHTS_PATH = OUTPUT_DIR / "digital_twin_weights.pt"
PARAMS_PATH = OUTPUT_DIR / "digital_twin_params.json"
STATS_PATH = OUTPUT_DIR / "training_stats.json"
ALERTS_PATH = OUTPUT_DIR / "maintenance_alerts.json"
KAGGLE_DIR = _BASE / "kaggle_cache"

DEFAULT_PARAMS = {
    "max_temperature_c": 80.0, "min_temperature_c": 15.0,
    "max_pressure_bar": 16.0, "min_pressure_bar": 0.5,
    "max_vibration_mm_s": 8.0, "min_vibration_mm_s": 0.5,
    "max_current_a": 20.0, "voltage_v": 480.0,
    "rated_power_kw": 11.0, "source": "defaults",
}

PROBABILITY_THRESHOLDS = {"critical": 0.75, "warning": 0.45}

FEATURE_EXCLUSIONS = {"Timestamp", "Status", "timestamp", "status"}

ALERT_RULES = [
 
    {"feature": "Temperature_C", "param_key": "max_temperature_c", "warn_pct": 0.82, "crit_pct": 1.00,
     "action_warn": "Increase cooling flow.", "action_crit": "STOP machine. Thermal seizure risk.", 
     "component": "Cooling system"
    },

    {"feature": "Pressure_bar", "param_key": "max_pressure_bar", "warn_pct": 0.85, "crit_pct": 1.00,
     "action_warn": "Inspect relief valve.", "action_crit": "Pressure exceeds limit. Seal rupture risk.", 
     "component": "Pressure valve"
    },

    {"feature": "Vibration_mm_s", "param_key": "max_vibration_mm_s", "warn_pct": 0.80, "crit_pct": 1.00,
     "action_warn": "Check bearings (ISO 10816 B→C).", "action_crit": "Severe vibration. Stop now.", 
     "component": "Bearings/rotor"
    },

    {"feature": "Current_A", "param_key": "max_current_a", "warn_pct": 0.85, "crit_pct": 1.00,
     "action_warn": "Check mechanical overload.", "action_crit": "Overcurrent. Inspect windings.", 
     "component": "Motor"
    },
]

