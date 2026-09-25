"""Backward-compatible import for the renamed maintenance agent."""

from agents.maintenance_agent import MaintenanceAgent

ReActMaintenanceAgent = MaintenanceAgent

__all__ = ["MaintenanceAgent", "ReActMaintenanceAgent"]
