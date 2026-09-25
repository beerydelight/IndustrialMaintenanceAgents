#!/usr/bin/env python3
"""Historical maintenance-log analysis agent."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError

from core import external_tools

MAX_RETRIES = 5
MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3.2:latest")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")


class ArchiveOutput(BaseModel):
    equipment_id: str
    history_summary: str
    recurring_issues: List[str] = Field(default_factory=list)
    last_maintenance: str
    risk_assessment: str


SYSTEM_PROMPT = """
You are an industrial maintenance history analyst.
Analyze only historical maintenance records returned by the archive tools. Do not
perform real-time diagnosis, invent sensor readings, or claim that a repair was
completed unless the records say so. Identify recurring issues and assess future
maintenance risk from the frequency, recency, and severity of recorded events.

Return exactly one JSON object with these fields:
{"equipment_id": "string", "history_summary": "string",
"recurring_issues": ["string"], "last_maintenance": "string",
"risk_assessment": "string"}
Return JSON only, with no markdown or commentary.
"""


class ArchiverAgent:
    """Summarize historical maintenance records for one piece of equipment."""

    def __init__(self, model_name: str = MODEL_NAME, host: str = OLLAMA_HOST, max_retries: int = MAX_RETRIES):
        if max_retries < 1:
            raise ValueError("max_retries must be at least 1")
        self.model_name = model_name
        self.host = host
        self.max_retries = max_retries

    def _call_model(self, messages: List[Dict[str, str]]) -> str:
        from ollama import Client

        response = Client(host=self.host).chat(
            model=self.model_name,
            messages=messages,
            options={"temperature": 0.2, "num_predict": 300},
            format="json",
        )
        message: Any = response.get("message") if isinstance(response, dict) else response.message
        content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
        if not isinstance(content, str):
            raise ValueError("Ollama returned a response without message content")
        return content

    @staticmethod
    def _fallback(equipment_id: str, logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not logs:
            return {
                "equipment_id": equipment_id,
                "history_summary": "No historical maintenance logs were found.",
                "recurring_issues": [],
                "last_maintenance": "unknown",
                "risk_assessment": "unknown: insufficient historical data",
            }
        dates = [str(log.get("maintenance_date") or log.get("timestamp") or "unknown") for log in logs]
        issues = sorted({str(log.get("issue") or log.get("description")) for log in logs if log.get("issue") or log.get("description")})
        return {
            "equipment_id": equipment_id,
            "history_summary": f"{len(logs)} historical maintenance record(s) found.",
            "recurring_issues": issues,
            "last_maintenance": max(dates),
            "risk_assessment": "medium: maintenance history contains prior service events",
        }

    def run(self, equipment_id: str, alert_context: Optional[str] = None) -> Dict[str, Any]:
        equipment_id = equipment_id.strip()
        if not equipment_id:
            raise ValueError("equipment_id must not be empty")
        logs = external_tools.query_maintenance_logs(equipment_id)
        records = []
        for log in logs:
            log_id = log.get("id", log.get("log_id"))
            details = external_tools.fetch_log_details(log_id) if log_id is not None else log
            if details:
                records.append(details)
        if not records:
            return self._fallback(equipment_id, records)

        context = {
            "equipment_id": equipment_id,
            "alert_context": alert_context or "",
            "maintenance_logs": records,
        }
        for attempt in range(1, self.max_retries + 1):
            try:
                raw = self._call_model([
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"{json.dumps(context, ensure_ascii=False)}\n"
                        f"Attempt {attempt}/{self.max_retries}. Return JSON only.",
                    },
                ])
                return ArchiveOutput.model_validate(json.loads(raw)).model_dump()
            except (
                ConnectionError,
                TimeoutError,
                OSError,
                ValueError,
                TypeError,
                json.JSONDecodeError,
                ValidationError,
            ):
                continue
        return self._fallback(equipment_id, records)

    def record_graph_result(
        self,
        equipment_id: str,
        final_report: str,
        maintenance_output: Dict[str, Any],
    ) -> int:
        """Persist the supervisor's completed graph result as a history record."""
        status = str(maintenance_output.get("status", "WARNING")).lower()
        issue = str(maintenance_output.get("diagnosis", "maintenance inspection"))
        action = str(
            maintenance_output.get("action_recommended", "Review the executive report")
        )
        description = final_report.strip()
        if not description:
            raise ValueError("final_report must not be empty")
        return external_tools.add_maintenance_log(
            equipment_id=equipment_id,
            maintenance_date=datetime.now(timezone.utc).date().isoformat(),
            issue=issue,
            description=description,
            action=action,
            technician="SupervisorAgent",
            status=status,
        )

    @staticmethod
    def compare_with_alert(
        equipment_id: str,
        alert_context: str,
        alert_timestamp: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compare alert keywords with recent completed maintenance records."""
        logs = external_tools.query_maintenance_logs(equipment_id)
        cutoff = None
        if alert_timestamp:
            try:
                event_time = datetime.fromisoformat(alert_timestamp.replace("Z", "+00:00"))
                cutoff = event_time - timedelta(days=90)
            except ValueError:
                cutoff = None

        alert_terms = {
            term for term in re.findall(r"[a-z]+", alert_context.lower()) if len(term) >= 5
        }
        issue_terms = {
            term
            for term in alert_terms
            if term not in {"alert", "reported", "generated", "reached", "required", "immediate"}
        }
        matches: List[Dict[str, Any]] = []
        for log in logs:
            try:
                log_time = datetime.fromisoformat(
                    str(log["maintenance_date"]).replace("Z", "+00:00")
                ).replace(tzinfo=timezone.utc)
            except (KeyError, ValueError):
                log_time = None
            if cutoff and log_time and log_time < cutoff:
                continue
            log_terms = set(
                re.findall(
                    r"[a-z]+",
                    f"{log.get('issue', '')} {log.get('description', '')}".lower(),
                )
            )
            if issue_terms & log_terms:
                matches.append(log)

        completed = [log for log in matches if log.get("status") == "completed"]
        recurring = len(matches) >= 2
        if not matches:
            recommendation = "no matching historical intervention found"
        elif recurring:
            recommendation = "recurring issue requiring escalation"
        elif completed:
            recommendation = "verify the previous repair"
        else:
            recommendation = "already addressed recently"

        return {
            "recommendation": recommendation,
            "matched_records": matches,
            "recent_completed_count": len(completed),
        }
