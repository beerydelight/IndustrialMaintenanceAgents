"""SQLite access for historical maintenance logs."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

DB_PATH = Path(
    os.getenv(
        "MAINTENANCE_LOG_DB",
        Path(__file__).resolve().parents[1] / "data" / "maintenance.db",
    )
)


def get_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def query_logs(equipment_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    if not equipment_id.strip() or not DB_PATH.exists():
        return []
    with get_connection() as connection:
        rows = connection.execute(
            """SELECT id, equipment_id, maintenance_date, issue, description, action,
                      technician, status
               FROM maintenance_logs
               WHERE equipment_id = ?
               ORDER BY maintenance_date DESC LIMIT ?""",
            (equipment_id.strip(), max(1, limit)),
        ).fetchall()
    return [dict(row) for row in rows]


def get_log_details(log_id: int) -> Optional[Dict[str, Any]]:
    if not DB_PATH.exists():
        return None
    with get_connection() as connection:
        row = connection.execute("SELECT * FROM maintenance_logs WHERE id = ?", (log_id,)).fetchone()
    return dict(row) if row else None

def insert_log(
    equipment_id: str,
    maintenance_date: str,
    issue: str,
    description: str,
    action: str,
    technician: str,
    status: str,
) -> int:
    values = {
        "equipment_id": equipment_id,
        "maintenance_date": maintenance_date,
        "issue": issue,
        "description": description,
        "action": action,
        "technician": technician,
        "status": status,
    }
    if any(not isinstance(value, str) or not value.strip() for value in values.values()):
        raise ValueError("All maintenance log fields must be non-empty strings")
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database file not found at {DB_PATH}")
    with get_connection() as connection:
        query = """INSERT INTO maintenance_logs 
            (equipment_id, maintenance_date, issue, description, action, technician, status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        parameters = (
            equipment_id.strip(),
            maintenance_date.strip(),
            issue.strip(),
            description.strip(),
            action.strip(),
            technician.strip(),
            status.strip(),
        )

        cursor = connection.execute(query, parameters)

        connection.commit()
        return cursor.lastrowid