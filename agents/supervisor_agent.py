#!/usr/bin/env python3
"""LangGraph supervisor coordinating maintenance and archive workers."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Dict, Optional, TypedDict
from digital_twin import digital_twin_tool

from langgraph.graph import END, START, StateGraph

from agents.archiver_agent import ArchiverAgent
from agents.maintenance_agent import MaintenanceAgent

MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3.2:latest")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    initial_input: str
    equipment_id: str
    maintenance_output: dict | None
    archiver_output: dict | None
    digital_twin_output: dict | None
    final_report: str | None


def extract_equipment_id(state: AgentState) -> Dict[str, str]:
    """Extract an equipment identifier and normalize common natural language."""
    started = time.perf_counter()
    text = state["initial_input"].strip()
    explicit = re.search(r"\b([A-Za-z]{2,}[-_]\d{1,3}(?:[-_]\d{1,3})?)\b", text)
    if explicit:
        equipment_id = explicit.group(1).upper().replace("-", "_")
    else:
        natural = re.search(
            r"\b(pump|press|conveyor|compressor|robot|mill|machine)\s+"
            r"(?:number|no\.?|#)?\s*(one|two|three|four|five|\d+)\b",
            text,
            re.IGNORECASE,
        )
        if natural:
            number_names = {
                "one": 1,
                "two": 2,
                "three": 3,
                "four": 4,
                "five": 5,
            }
            number_text = natural.group(2).lower()
            number = number_names[number_text] if number_text in number_names else int(number_text)
            equipment_id = f"{natural.group(1).upper()}_{number:02d}"
        else:
            equipment_id = "UNKNOWN_EQUIPMENT"
    logger.info(
        "state=extract_equipment_id equipment_id=%s seconds_to_think=%.3f",
        equipment_id,
        time.perf_counter() - started,
    )
    return {"equipment_id": equipment_id}


def maintenance_worker(state: AgentState) -> Dict[str, dict]:
    """Run the real-time diagnostic worker."""
    started = time.perf_counter()
    logger.info("tool_call=MaintenanceAgent.run equipment_id=%s", state["equipment_id"])
    try:
        output = MaintenanceAgent().run(state["initial_input"])
    except Exception:
        logger.exception(
            "tool_call=MaintenanceAgent.run status=error seconds_to_think=%.3f",
            time.perf_counter() - started,
        )
        raise
    logger.info(
        "tool_call=MaintenanceAgent.run status=complete seconds_to_think=%.3f",
        time.perf_counter() - started,
    )
    return {"maintenance_output": output}


def archiver_worker(state: AgentState) -> Dict[str, dict]:
    """Run the historical analysis worker for the extracted equipment."""
    started = time.perf_counter()
    logger.info(
        "tool_call=ArchiverAgent.run equipment_id=%s",
        state["equipment_id"],
    )
    try:
        output = ArchiverAgent().run(
            state["equipment_id"],
            alert_context=state["initial_input"],
        )
    except Exception:
        logger.exception(
            "tool_call=ArchiverAgent.run status=error seconds_to_think=%.3f",
            time.perf_counter() - started,
        )
        raise
    logger.info(
        "tool_call=ArchiverAgent.run status=complete seconds_to_think=%.3f",
        time.perf_counter() - started,
    )
    return {"archiver_output": output}


def digital_twin_worker(state: AgentState) -> Dict[str, dict]:
    """Call the digital twin as a deterministic tool, never as an agent."""
    started = time.perf_counter()
    logger.info("tool_call=digital_twin_tool equipment_id=%s", state["equipment_id"])
    try:
        output = digital_twin_tool(state["initial_input"])
    except Exception:
        logger.exception(
            "tool_call=digital_twin_tool status=error seconds_to_think=%.3f",
            time.perf_counter() - started,
        )
        raise
    logger.info(
        "tool_call=digital_twin_tool status=complete seconds_to_think=%.3f",
        time.perf_counter() - started,
    )
    return {"digital_twin_output": output}


def _fallback_report(state: AgentState) -> str:
    maintenance = state["maintenance_output"] or {}
    archive = state["archiver_output"] or {}
    twin = state["digital_twin_output"] or {}
    prediction = twin.get("prediction", {})
    return (
        "Executive Maintenance Report\n\n"
        f"Equipment: {state['equipment_id']}\n"
        f"Current status: {maintenance.get('status', 'unknown')}\n"
        f"Diagnosis: {maintenance.get('diagnosis', ' unavailable')}\n"
        f"Recommended action: {maintenance.get('action_recommended', ' unavailable')}\n"
        f"Digital twin status: {prediction.get('status', 'unknown')}\n"
        f"Digital twin failure probability: {prediction.get('failure_probability', 'unknown')}\n"
        f"Historical risk: {archive.get('risk_assessment', 'unknown')}\n"
        f"History: {archive.get('history_summary', 'No historical summary available.')}\n"
    )


def _synthesize_report(state: AgentState) -> str:
    """Synthesize both worker outputs into the sole human-facing report."""
    started = time.perf_counter()
    prompt = f"""Create an Executive Maintenance Report for a technician.
Combine the current diagnostic and historical analysis. State the equipment,
current condition, immediate action, historical risk, and whether follow-up or
escalation is needed. Do not invent facts.

Equipment ID: {state['equipment_id']}
Initial input: {state['initial_input']}
Maintenance output: {json.dumps(state['maintenance_output'], ensure_ascii=False)}
Archiver output: {json.dumps(state['archiver_output'], ensure_ascii=False)}
Digital twin output: {json.dumps(state['digital_twin_output'], ensure_ascii=False)}
"""
    try:
        from ollama import Client

        response = Client(host=OLLAMA_HOST).chat(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You are a senior maintenance supervisor. Return a concise human-readable report.",
                },
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0.2, "num_predict": 400},
        )
        message: Any = response.get("message") if isinstance(response, dict) else response.message
        content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
        if isinstance(content, str) and content.strip():
            logger.info(
                "tool_call=supervisor_synthesis status=complete seconds_to_think=%.3f",
                time.perf_counter() - started,
            )
            return content.strip()
    except (ConnectionError, TimeoutError, OSError, ValueError, TypeError) as exc:
        logger.error(
            "tool_call=supervisor_synthesis status=error error=%s seconds_to_think=%.3f",
            exc,
            time.perf_counter() - started,
        )
    logger.info(
        "tool_call=supervisor_synthesis status=fallback seconds_to_think=%.3f",
        time.perf_counter() - started,
    )
    return _fallback_report(state)


def supervisor(state: AgentState) -> Dict[str, Optional[str]]:
    """Route missing work or synthesize once both worker outputs are available."""
    logger.info(
        "state=supervisor maintenance_ready=%s archiver_ready=%s",
        state["maintenance_output"] is not None,
        state["archiver_output"] is not None,
    )
    if (
        state["maintenance_output"] is not None
        and state["archiver_output"] is not None
        and state["digital_twin_output"] is not None
    ):
        final_report = _synthesize_report(state)
        log_id = ArchiverAgent().record_graph_result(
            equipment_id=state["equipment_id"],
            final_report=final_report,
            maintenance_output=state["maintenance_output"],
        )
        logger.info("tool_call=add_maintenance_log status=complete log_id=%s", log_id)
        return {"final_report": final_report}
    return {"final_report": None}


def _route_after_supervisor(state: AgentState) -> str:
    if state["final_report"] is not None:
        return END
    if state["maintenance_output"] is None:
        return "maintenance_worker"
    if state["digital_twin_output"] is None:
        return "digital_twin_worker"
    if state["archiver_output"] is None:
        return "archiver_worker"
    return END


def build_graph():
    """Build and compile the supervisor/worker graph."""
    graph = StateGraph(AgentState)
    graph.add_node("extract_equipment_id", extract_equipment_id)
    graph.add_node("supervisor", supervisor)
    graph.add_node("maintenance_worker", maintenance_worker)
    graph.add_node("archiver_worker", archiver_worker)
    graph.add_node("digital_twin_worker", digital_twin_worker)
    graph.add_edge(START, "extract_equipment_id")
    graph.add_edge("extract_equipment_id", "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        _route_after_supervisor,
        {
            "maintenance_worker": "maintenance_worker",
            "digital_twin_worker": "digital_twin_worker",
            "archiver_worker": "archiver_worker",
            END: END,
        },
    )
    graph.add_edge("maintenance_worker", "supervisor")
    graph.add_edge("digital_twin_worker", "supervisor")
    graph.add_edge("archiver_worker", "supervisor")
    return graph.compile()


def run_pipeline(human_input: str) -> str:
    """Run the strict input -> worker orchestration -> report pipeline."""
    if not human_input or not human_input.strip():
        raise ValueError("human_input must not be empty")
    initial_state: AgentState = {
        "initial_input": human_input.strip(),
        "equipment_id": "",
        "maintenance_output": None,
        "archiver_output": None,
        "digital_twin_output": None,
        "final_report": None,
    }
    result = build_graph().invoke(initial_state)
    final_report = result.get("final_report")
    if not isinstance(final_report, str) or not final_report.strip():
        raise RuntimeError("Supervisor graph completed without a final report")
    return final_report


__all__ = ["AgentState", "build_graph", "run_pipeline"]
