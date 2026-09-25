"""Kaggle authentication, download, detection, and dataset adapters."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from typing import Callable

import numpy as np
import pandas as pd

from ..config import KAGGLE_DIR, OUTPUT_DIR, PARAMS_PATH, SENSORS_PATH

logger = logging.getLogger(__name__)

KAGGLE_DATASETS = [
    {"slug": "stephanmatzka/predictive-maintenance-dataset-ai4i-2020", "csv_hint": "predictive_maintenance",
     "fingerprint": {"air temperature [k]", "process temperature [k]", "rotational speed [rpm]"},
     "adapter": "adapt_ai4i", "label": "AI4I 2020"},
    {"slug": "nphardesty/pump-sensor-data", "csv_hint": "sensor",
     "fingerprint": {"sensor_00", "sensor_01", "machine_status"}, "adapter": "adapt_pump_sensor", "label": "Pump Sensor"},
    {"slug": "arnabbiswas1/microsoft-azure-predictive-maintenance", "csv_hint": "telemetry",
     "fingerprint": {"volt", "rotate", "pressure", "vibration", "machineid"}, "adapter": "adapt_azure_maint", "label": "Azure Predictive"},
]


def _kaggle_credentials() -> tuple[str, str] | None:
    username = os.environ.get("KAGGLE_USERNAME", "").strip()
    key = os.environ.get("KAGGLE_KEY", "").strip()

    if username and key:
        return username, key
    config_path = os.path.expanduser("~/.kaggle/kaggle.json")
    
    if os.path.exists(config_path):
        try:
            with open(config_path, encoding="utf-8") as handle:
                data = json.load(handle)
            return data["username"], data["key"]
        except (OSError, KeyError, json.JSONDecodeError):
            logger.warning("Invalid Kaggle credentials file")
    return None


def _write_kaggle_json(username: str, key: str) -> None:
    directory, path = os.path.expanduser("~/.kaggle"), os.path.expanduser("~/.kaggle/kaggle.json")
    os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"username": username, "key": key}, handle)
    os.chmod(path, 0o600)


def _download_kaggle_dataset(slug: str) -> str | None:
    destination = os.path.join(KAGGLE_DIR, slug.replace("/", "_"))
    if os.path.isdir(destination) and any(name.endswith(".csv") for name in os.listdir(destination)):
        return destination
    os.makedirs(destination, exist_ok=True)
    try:
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", slug, "-p", destination, "--unzip"],
            capture_output=True, text=True, timeout=120, check=False,
        )
        if result.returncode:
            raise RuntimeError(result.stderr[:300])
        return destination
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        logger.warning("Kaggle download failed for %s: %s", slug, exc)
        shutil.rmtree(destination, ignore_errors=True)
        return None


def _detect_dataset(frame: pd.DataFrame) -> dict | None:
    columns = {column.lower().strip() for column in frame.columns}
    return next((dataset for dataset in KAGGLE_DATASETS if dataset["fingerprint"].issubset(columns)), None)


def _find_csv(directory: str, hint: str) -> str | None:
    candidates = [
        os.path.join(root, filename)
        for root, _, files in os.walk(directory)
        for filename in files if filename.endswith(".csv")
    ]
    candidates.sort(key=lambda path: hint.lower() not in os.path.basename(path).lower())
    return candidates[0] if candidates else None


def _scale(raw: pd.Series, low: float, high: float) -> pd.Series:
    values = raw.fillna(raw.median())
    if values.max() <= values.min():
        return values
    return ((values - values.min()) / max(values.max() - values.min(), 1e-6) * (high - low) + low).round(3)


def adapt_ai4i(frame: pd.DataFrame, params: dict) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [column.strip() for column in frame.columns]
    columns = {column.lower(): column for column in frame.columns}
    output = pd.DataFrame({
        "Temperature_C": frame[columns["air temperature [k]"]].map(lambda value: round(value - 273.15, 1)),
        "Process_Temperature_C": frame[columns["process temperature [k]"]].map(lambda value: round(value - 273.15, 1)),
        "Rotational_Speed_rpm": frame[columns["rotational speed [rpm]"]],
        "Torque_Nm": frame[columns["torque [nm]"]],
        "Tool_Wear_min": frame[columns["tool wear [min]"]],
    })
    hdf, pwf, twf = columns.get("hdf"), columns.get("pwf"), columns.get("twf")
    output["Status"] = frame.apply(
        lambda row: "Critical" if (hdf and row.get(hdf, 0)) or (pwf and row.get(pwf, 0))
        else "Warning" if (twf and row.get(twf, 0)) else "Normal", axis=1
    )
    rng = np.random.default_rng(42)
    nominal = (params["max_pressure_bar"] + params["min_pressure_bar"]) / 2
    output["Pressure_bar"] = rng.normal(nominal, nominal * 0.05, len(output)).clip(
        params["min_pressure_bar"], params["max_pressure_bar"] * 1.2
    ).round(3)
    output["Timestamp"] = pd.date_range("2024-01-01", periods=len(output), freq="min").strftime("%Y-%m-%d %H:%M:%S")
    return output


def adapt_pump_sensor(frame: pd.DataFrame, params: dict) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [column.strip() for column in frame.columns]
    status_col = next((column for column in frame.columns if "machine_status" in column.lower()), None)
    if not status_col:
        raise ValueError("Missing machine_status")
    mapping = {"1": "Critical", "BROKEN": "Critical", "2": "Warning", "RECOVERING": "Warning"}
    output = pd.DataFrame()
    sensor_cols = [column for column in frame.columns if column.lower().startswith("sensor_")]
    top_sensors = frame[sensor_cols].var().sort_values(ascending=False).head(5).index.tolist()
    names = ["Pressure_bar", "Temperature_C", "Vibration_mm_s", "Current_A", "Power_kW"]
    limits = [(params["min_pressure_bar"], params["max_pressure_bar"]), (params["min_temperature_c"], params["max_temperature_c"]),
              (params["min_vibration_mm_s"], params["max_vibration_mm_s"]), (0.0, params["max_current_a"]), (0.0, params["rated_power_kw"])]
    for index, sensor in enumerate(top_sensors):
        output[names[index]] = _scale(frame[sensor], *limits[index])
    output["Status"] = frame[status_col].astype(str).str.strip().str.upper().map(lambda value: mapping.get(value, "Normal"))
    timestamps = frame.get("timestamp", pd.date_range("2024-01-01", periods=len(frame), freq="min"))
    output["Timestamp"] = pd.to_datetime(timestamps, errors="coerce").fillna(pd.Timestamp("2024-01-01")).dt.strftime("%Y-%m-%d %H:%M:%S")
    return output.sample(min(5000, len(output)), random_state=42).reset_index(drop=True)


def adapt_azure_maint(frame: pd.DataFrame, params: dict) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [column.strip().lower() for column in frame.columns]
    output = pd.DataFrame()
    aliases = {"Temperature_C": ["temperature"], "Rotational_Speed_rpm": ["rotate", "rotational_speed"],
               "Pressure_bar": ["pressure"], "Vibration_mm_s": ["vibration"], "Current_A": ["volt", "current"], "Power_kW": ["power"]}
    limits = {"Temperature_C": (params["min_temperature_c"], params["max_temperature_c"]), "Pressure_bar": (params["min_pressure_bar"], params["max_pressure_bar"]),
              "Vibration_mm_s": (params["min_vibration_mm_s"], params["max_vibration_mm_s"]), "Current_A": (0.0, params["max_current_a"]),
              "Power_kW": (0.0, params["rated_power_kw"])}
    for target, sources in aliases.items():
        source = next((candidate for candidate in sources if candidate in frame.columns), None)
        if source:
            output[target] = frame[source].fillna(frame[source].median()).round(0) if target == "Rotational_Speed_rpm" else _scale(frame[source], *limits[target])
    means, stds = output.mean(), output.std()
    def derive_status(row: pd.Series) -> str:
        scores = [abs((row.get(column, means[column]) - means[column]) / max(stds[column], 1e-6)) for column in output.columns]
        score = max(scores) if scores else 0
        return "Critical" if score > 3 else "Warning" if score > 2 else "Normal"
    output["Status"] = output.apply(derive_status, axis=1)
    timestamps = frame.get("datetime", pd.date_range("2024-01-01", periods=len(frame), freq="min"))
    output["Timestamp"] = pd.to_datetime(timestamps, errors="coerce").fillna(pd.Timestamp("2024-01-01")).dt.strftime("%Y-%m-%d %H:%M:%S")
    return output.sample(min(5000, len(output)), random_state=42).reset_index(drop=True)


ADAPTER_FN: dict[str, Callable[[pd.DataFrame, dict], pd.DataFrame]] = {
    "adapt_ai4i": adapt_ai4i, "adapt_pump_sensor": adapt_pump_sensor, "adapt_azure_maint": adapt_azure_maint,
}


def fetch_kaggle_dataset(params: dict) -> pd.DataFrame | None:
    credentials = _kaggle_credentials()
    if not credentials:
        return None
    _write_kaggle_json(*credentials)
    for dataset in KAGGLE_DATASETS:
        local = _download_kaggle_dataset(dataset["slug"])
        csv_path = _find_csv(local, dataset["csv_hint"]) if local else None
        if not csv_path:
            continue
        try:
            raw = pd.read_csv(csv_path, nrows=10000)
            detected = _detect_dataset(raw)
            if not detected:
                continue
            adapted = ADAPTER_FN[detected["adapter"]](raw, params)
            if len(adapted) < 100 or "Status" not in adapted or not (adapted["Status"].isin(["Critical", "Warning"])).any():
                continue
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            adapted.to_csv(SENSORS_PATH, index=False)
            params.update({"data_source": f"kaggle_{detected['label']}", "kaggle_slug": dataset["slug"]})
            with open(PARAMS_PATH, "w", encoding="utf-8") as handle:
                json.dump(params, handle, indent=2)
            return adapted
        except (OSError, ValueError, KeyError, pd.errors.ParserError):
            logger.exception("Adapter failed for %s", dataset["slug"])
    return None
