#!/usr/bin/env python3
"""Deterministic helpers used by the maintenance agent."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from ollama import Client
from pydantic import BaseModel, Field, ValidationError


class ToolSpec(BaseModel):
    name: str
    description: str
    params: Dict[str, Any] = Field(default_factory=dict)


class DiagnosticOutput(BaseModel):
    status: str
    diagnosis: str
    action_recommended: str
    inventory_status: str


def load_alert_input(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Input data file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Input data file is not valid JSON: {exc}") from exc

    if isinstance(payload, list):
        alert_texts = [
            item.get("alert_text", "").strip()
            for item in payload
            if isinstance(item, dict) and isinstance(item.get("alert_text"), str)
        ]
        if alert_texts:
            return "\n\n".join(alert_texts)
    if isinstance(payload, dict):
        alert_text = payload.get("alert_text")
        if isinstance(alert_text, str) and alert_text.strip():
            return alert_text.strip()
        alerts = payload.get("alerts")
        if isinstance(alerts, list):
            return " ".join(str(item).strip() for item in alerts if str(item).strip())
        message = payload.get("message")
        if isinstance(message, str):
            return message.strip()
    if isinstance(payload, str):
        return payload.strip()
    return json.dumps(payload, ensure_ascii=False)


def extract_json_object(text: Optional[str]) -> Optional[Dict[str, Any]]:
    if not text or not text.strip():
        return None

    candidate = text.strip()
    match = re.search(r"\{.*\}", candidate, re.DOTALL)
    if match:
        candidate = match.group(0)
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _number(pattern: str, text: str) -> Optional[float]:
    match = re.search(pattern, text, re.IGNORECASE)
    return float(match.group(1)) if match else None


def get_sensory_data(alert_text: str) -> Dict[str, Any]:
    """Extract machine identity and numeric sensor readings from an alert."""
    text = alert_text.strip()
    data: Dict[str, Any] = {}
    machine_id = re.search(r"\b([A-Z]{2,}-\d{2,}(?:-\d{2,})?)\b", text)
    if machine_id:
        data["machine_id"] = machine_id.group(1)

    patterns = {
        "bearing_temperature_c": r"temperature\s+(?:reached|is|was)\s+([0-9]+(?:\.[0-9]+)?)\s*°?\s*c",
        "vibration_mm_s": r"vibration(?:\s+amplitude)?\s+(?:is|was|reached)\s+([0-9]+(?:\.[0-9]+)?)\s*mm/s",
        "motor_current_a": r"motor current\s+(?:rose to|is|was)\s+([0-9]+(?:\.[0-9]+)?)\s*a",
        "coolant_pressure_psi": r"coolant pressure\s+(?:fell to|is|was)\s+([0-9]+(?:\.[0-9]+)?)\s*psi",
    }
    for name, pattern in patterns.items():
        value = _number(pattern, text)
        if value is not None:
            data[name] = int(value) if value.is_integer() else value
    return data


def check_inventory(alert_text: str) -> Dict[str, Any]:
    """Find mentioned replacement parts and classify their available stock."""
    text = alert_text.lower()
    parts: List[Dict[str, Any]] = []
    for match in re.finditer(
        r"(?:spare\s+)?([a-z][a-z -]*?\b(?:kit|filter|bearing|seal|belt|valve|motor))\s+stock\s+is\s+(\d+)",
        text,
    ):
        part = re.sub(r"\s+", " ", match.group(1)).strip()
        part = re.sub(r"^(?:and|or)\s+", "", part)
        stock = int(match.group(2))
        parts.append({"part": part, "stock": stock, "status": "available" if stock > 0 else "out_of_stock"})

    if not parts:
        return {"status": "unknown", "replacement_parts": []}
    status = "critical" if any(item["stock"] == 0 for item in parts) else "available"
    return {"status": status, "replacement_parts": parts}


def call_model(messages: List[Dict[str, str]], model_name: str, host: str) -> str:
    response = Client(host=host).chat(
        model=model_name,
        messages=messages,
        options={"temperature": 0.2, "num_predict": 250},
        format="json",
    )
    if isinstance(response, dict):
        message = response.get("message")
        content = message.get("content") if isinstance(message, dict) else None
    else:
        message = getattr(response, "message", None)
        content = getattr(message, "content", None)
    if not isinstance(content, str):
        raise ValueError("Ollama returned a response without message content")
    return content


def validate_schema(payload: Dict[str, Any]) -> Optional[DiagnosticOutput]:
    try:
        instance = DiagnosticOutput.model_validate(payload)
    except ValidationError:
        return None
    return instance if instance.status.upper() in {"OK", "WARNING", "CRITICAL"} else None


def fallback(
    alert_text: str,
    reason: str,
    inventory: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    lower_text = alert_text.lower()
    if any(keyword in lower_text for keyword in ("fire", "critical", "overtemp", "severe", "bearing", "scraping", "overload", "failure")):
        status = "CRITICAL"
        diagnosis = "The machine shows critical abnormal operating conditions likely linked to mechanical distress or thermal overload."
        action = "Immediately isolate the machine, inspect the most stressed subsystem, and replace failed components before restarting."
    elif any(keyword in lower_text for keyword in ("vibration", "temperature", "pressure", "noise", "warning")):
        status = "WARNING"
        diagnosis = "The machine is operating outside normal conditions and requires targeted inspection."
        action = "Schedule a controlled inspection and verify the affected sensor and mechanical subsystem."
    else:
        status = "OK"
        diagnosis = "The equipment appears stable and no major fault is confirmed from the alert data."
        action = "Continue routine monitoring and document the current operating state."

    inventory = inventory or check_inventory(alert_text)
    inventory_status = {
        "critical": "critical: one or more required replacement parts are out of stock",
        "available": "replacement parts are available",
        "unknown": "replacement-part stock could not be determined",
    }[inventory["status"]]
    return {
        "status": status,
        "diagnosis": f"{diagnosis} Trigger: {reason}",
        "action_recommended": action,
        "inventory_status": inventory_status,
    }


def get_tool_schema() -> List[ToolSpec]:
    return [
        ToolSpec(name="get_sensory_data", description="Extract machine sensor readings from alert text", params={"alert_text": "string"}),
        ToolSpec(name="check_inventory", description="Find replacement parts and classify their stock", params={"alert_text": "string"}),
        ToolSpec(name="llm_call", description="Call the local Ollama model to produce diagnostics", params={"model": "string", "messages": "list"}),
        ToolSpec(name="fallback", description="Produce a deterministic diagnostic without calling the model", params={"alert_text": "string", "reason": "string"}),
        ToolSpec(
            name="add_maintenance_log",
            description="Persist a structured maintenance result in the historical SQLite log",
            params={
                "equipment_id": "string",
                "maintenance_date": "ISO-8601 date",
                "issue": "string",
                "description": "string",
                "action": "string",
                "technician": "string",
                "status": "string",
            },
        ),
    ]
