"""Rule-based maintenance alert engine."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import List

from .config import ALERTS_PATH, ALERT_RULES, DEFAULT_PARAMS, PARAMS_PATH, STATS_PATH

logger = logging.getLogger(__name__)


def _read_json(path: str, fallback: dict | list) -> dict | list:
    if not os.path.exists(path):
        return fallback
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        logger.warning("Unable to load JSON file %s", path)
        return fallback


def generate_maintenance_alerts(readings: dict, prob: float, status: str, features: List[str]) -> List[dict]:
    """Generate alerts and append the result to bounded alert history."""
    if status == "Normal":
        return []
    params = _read_json(PARAMS_PATH, dict(DEFAULT_PARAMS))
    stats = _read_json(STATS_PATH, {})
    alerts: list[dict] = []
    
    for rule in ALERT_RULES:
        feature, value = rule["feature"], readings.get(rule["feature"])
        if value is None:
            continue
        limit = params.get(rule["param_key"])
        if limit:
            critical_threshold, warning_threshold = limit * rule["crit_pct"], limit * rule["warn_pct"]
            if value >= critical_threshold:
                alerts.append({"severity": "Critical", "feature": feature, "value": value, "threshold": critical_threshold,
                               "component": rule["component"], "action": rule["action_crit"],
                               "urgency_score": min(100, int(prob * 100 + ((value / limit - 1) * 50)))})
            elif value >= warning_threshold:
                alerts.append({"severity": "Warning", "feature": feature, "value": value, "threshold": warning_threshold,
                               "component": rule["component"], "action": rule["action_warn"],
                               "urgency_score": min(80, int(prob * 60 + 20))})
        elif feature in stats.get("medians", {}) and feature in stats.get("stds", {}):
            z_score = abs(value - stats["medians"][feature]) / max(stats["stds"][feature], 1e-6)
            if z_score > 3:
                alerts.append({"severity": "Critical", "feature": feature, "value": value,
                               "threshold": round(stats["medians"][feature] + 3 * stats["stds"][feature], 3),
                               "component": rule["component"], "action": rule["action_crit"],
                               "urgency_score": min(100, int(z_score * 20 + prob * 40))})
            elif z_score > 2:
                alerts.append({"severity": "Warning", "feature": feature, "value": value,
                               "threshold": round(stats["medians"][feature] + 2 * stats["stds"][feature], 3),
                               "component": rule["component"], "action": rule["action_warn"],
                               "urgency_score": min(80, int(z_score * 15 + prob * 30))})
    unique = {alert["feature"]: alert for alert in alerts}
    alerts = sorted(unique.values(), key=lambda item: -item["urgency_score"])
    history = _read_json(ALERTS_PATH, [])
    if not isinstance(history, list):
        history = []
    history.append({"timestamp": datetime.utcnow().isoformat(), "status": status,
                    "failure_probability": round(prob, 4), "n_alerts": len(alerts), "alerts": alerts})
    os.makedirs(os.path.dirname(ALERTS_PATH), exist_ok=True)
    with open(ALERTS_PATH, "w", encoding="utf-8") as handle:
        json.dump(history[-200:], handle, indent=2)
    return alerts
