"""Tool adapters exposed to historical-analysis agents."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from core import db_logic


def query_maintenance_logs(equipment_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    return db_logic.query_logs(equipment_id, limit=limit)


def fetch_log_details(log_id: int) -> Optional[Dict[str, Any]]:
    return db_logic.get_log_details(log_id)

def add_maintenance_log(
    equipment_id: str,
    maintenance_date: str,
    issue: str,
    description: str,
    action: str,
    technician: str,
    status: str,
) -> int:
    """Insert a completed or planned maintenance record."""
    return db_logic.insert_log(
        equipment_id=equipment_id,
        maintenance_date=maintenance_date,
        issue=issue,
        description=description,
        action=action,
        technician=technician,
        status=status,
    )