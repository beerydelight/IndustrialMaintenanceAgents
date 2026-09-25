#!/usr/bin/env python3
"""Real-time industrial maintenance diagnostics agent."""

from __future__ import annotations

import json
import os
from typing import Dict, Optional

from dotenv import load_dotenv

from core import tools
from core.rag_pipeline import IndustrialRAGPipeline, build_agent_context

load_dotenv()

MAX_RETRIES = 5
MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3.2:latest")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

EXPECTED_OUTPUT = {
    "status": "CRITICAL",
    "diagnosis": "Concise diagnosis",
    "action_recommended": "Specific technician action",
    "inventory_status": "available, low, or critical",
}

TOOL_SCHEMA = json.dumps([tool.model_dump() for tool in tools.get_tool_schema()], indent=2)

SYSTEM_PROMPT = f"""You are a senior industrial maintenance diagnostics agent.
Use only the alert, sensory data, inventory data, and tools provided.
Return exactly one JSON object with these fields: {json.dumps(EXPECTED_OUTPUT)}.
Valid status values are OK, WARNING, and CRITICAL. Do not return markdown or commentary.

Available tools:
{TOOL_SCHEMA}
"""


class MaintenanceAgent:
    """Diagnose a current equipment alert using sensors, inventory, and RAG."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        host: str = OLLAMA_HOST,
        max_retries: int = MAX_RETRIES,
        rag_pipeline: Optional[IndustrialRAGPipeline] = None,
    ):
        if max_retries < 1:
            raise ValueError("max_retries must be at least 1")
        self.model_name = model_name
        self.host = host
        self.max_retries = max_retries
        self.rag_pipeline = rag_pipeline or IndustrialRAGPipeline()

    def run(self, alert_text: str) -> Dict[str, str]:
        alert = alert_text.strip()
        sensory_data = tools.get_sensory_data(alert)
        inventory = tools.check_inventory(alert)
        retrieved_docs = self.rag_pipeline.retrieve(alert)
        rag_context = build_agent_context(query=alert, docs=retrieved_docs)
        context = (
            f"Alert: {alert}\n"
            f"Sensory data: {json.dumps(sensory_data, ensure_ascii=False)}\n"
            f"Inventory: {json.dumps(inventory, ensure_ascii=False)}\n\n"
            f"{rag_context}"
        )
        last_error = "model did not return a valid diagnostic"
        for attempt in range(1, self.max_retries + 1):
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"{context}\nAttempt {attempt}/{self.max_retries}. Return JSON only."},
            ]
            try:
                parsed = tools.extract_json_object(
                    tools.call_model(messages, model_name=self.model_name, host=self.host)
                )
            except (ConnectionError, TimeoutError, ValueError, OSError) as exc:
                last_error = f"model call failed: {exc}"
                continue
            validated = tools.validate_schema(parsed) if parsed is not None else None
            if validated is not None:
                return validated.model_dump()
            last_error = "model response did not match the diagnostic schema"
        return tools.fallback(alert, last_error, inventory)
