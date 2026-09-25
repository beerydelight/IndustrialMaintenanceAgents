"""Public API for the digital twin pipeline."""

from __future__ import annotations


def run_twin_pipeline(extracted_text: str) -> dict:
    """Called by the LangGraph supervisor. Runs the full training pipeline."""
    from .pipeline import digital_twin_node

    return digital_twin_node(extracted_text)


def predict(readings: dict) -> dict:
    """Called by the supervisor for inference-only predictions."""
    from .model.inference import predict as _predict

    return _predict(readings)


def digital_twin_tool(input_data: str | dict) -> dict:
    """Supervisor tool for deterministic digital-twin analysis."""
    from .tool import digital_twin_tool as _digital_twin_tool

    return _digital_twin_tool(input_data)


__all__ = ["digital_twin_tool", "run_twin_pipeline", "predict"]
