#!/usr/bin/env python3
"""Create the local SQLite database used by the historical archive agent."""

from __future__ import annotations

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "data" / "maintenance.db"

LOGS = [
    ("HP-312-03", "2026-01-14", "bearing wear", "Drive-side bearing showed elevated vibration.", "Replaced bearing and lubricated drive housing.", "A. Martin", "completed"),
    ("HP-312-03", "2026-02-22", "coolant pressure loss", "Coolant pressure dropped below operating range.", "Replaced leaking hose and pressure-tested circuit.", "S. Chen", "completed"),
    ("HP-312-03", "2026-04-09", "bearing temperature", "Bearing temperature exceeded the service threshold.", "Inspected bearing alignment and renewed lubricant.", "A. Martin", "completed"),
    ("HP-312-03", "2026-05-18", "hydraulic pressure fluctuation", "Relief valve cycled during a production run.", "Cleaned relief valve and calibrated pressure control.", "J. Okafor", "completed"),
    ("HP-312-03", "2026-07-03", "vibration", "Drive housing vibration increased during startup.", "Tightened coupling and replaced worn seal.", "S. Chen", "completed"),
    ("HP-312-03", "2026-08-27", "bearing wear", "Noise and vibration indicated recurring bearing wear.", "Scheduled bearing kit replacement at next shutdown.", "J. Okafor", "follow_up_required"),
    ("CN-204-11", "2026-01-08", "conveyor belt tracking", "Belt drifted toward the operator side.", "Realigned rollers and adjusted tracking sensors.", "L. Garcia", "completed"),
    ("CN-204-11", "2026-03-16", "motor overheating", "Drive motor temperature rose above normal.", "Cleaned cooling vents and replaced motor fan.", "R. Patel", "completed"),
    ("CN-204-11", "2026-06-12", "conveyor belt tracking", "Intermittent belt misalignment returned.", "Replaced damaged idler and aligned conveyor frame.", "L. Garcia", "completed"),
    ("CN-204-11", "2026-08-04", "gearbox oil leak", "Oil residue found beneath gearbox housing.", "Replaced output shaft seal and topped up oil.", "R. Patel", "completed"),
    ("PU-118-07", "2026-01-25", "pump seal leak", "Small leak detected at the pump mechanical seal.", "Replaced mechanical seal and checked shaft runout.", "M. Lewis", "completed"),
    ("PU-118-07", "2026-02-28", "low discharge pressure", "Pump discharge pressure was below target.", "Cleaned inlet strainer and restored suction flow.", "N. Wilson", "completed"),
    ("PU-118-07", "2026-05-06", "pump seal leak", "Seal leakage recurred after extended operation.", "Installed upgraded seal cartridge.", "M. Lewis", "completed"),
    ("PU-118-07", "2026-08-19", "low discharge pressure", "Pressure dipped during peak demand.", "Inspected impeller for wear; monitoring required.", "N. Wilson", "follow_up_required"),
    ("CM-445-02", "2026-01-19", "compressor oil level", "Oil level below the recommended mark.", "Topped up oil and inspected for external leaks.", "D. Brown", "completed"),
    ("CM-445-02", "2026-03-30", "air filter restriction", "Compressor intake restriction alarm activated.", "Replaced intake filter and reset alarm.", "E. Rossi", "completed"),
    ("CM-445-02", "2026-06-21", "compressor oil level", "Oil consumption increased over two weeks.", "Replaced separator seal and verified oil level.", "D. Brown", "completed"),
    ("CM-445-02", "2026-09-02", "air filter restriction", "Intake differential pressure was high.", "Replaced clogged filter and cleaned intake duct.", "E. Rossi", "completed"),
    ("RO-901-05", "2026-02-03", "robot axis backlash", "Backlash exceeded calibration tolerance on axis 4.", "Adjusted gearbox preload and recalibrated axis.", "K. Singh", "completed"),
    ("RO-901-05", "2026-04-14", "gripper sensor fault", "End-effector sensor intermittently failed.", "Replaced proximity sensor and tested I/O.", "T. Nguyen", "completed"),
    ("RO-901-05", "2026-07-11", "robot axis backlash", "Axis 4 positional repeatability degraded.", "Replaced gearbox coupling and recalibrated robot.", "K. Singh", "completed"),
    ("MX-630-09", "2026-01-11", "coolant filter blockage", "Coolant flow reduced due to filter restriction.", "Replaced coolant filter and flushed lines.", "P. Evans", "completed"),
    ("MX-630-09", "2026-04-26", "spindle vibration", "Spindle vibration exceeded the inspection limit.", "Balanced spindle tooling and checked bearings.", "P. Evans", "completed"),
    ("MX-630-09", "2026-07-29", "coolant filter blockage", "Coolant flow alarm returned during machining.", "Replaced filter housing gasket and filter.", "H. Kim", "completed"),
    ("MX-630-09", "2026-09-09", "spindle vibration", "Vibration increased at high spindle speed.", "Restricted high-speed operation pending bearing inspection.", "H. Kim", "follow_up_required"),
]


def seed_database() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("DROP TABLE IF EXISTS maintenance_logs")
        connection.execute(
            """
            CREATE TABLE maintenance_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                equipment_id TEXT NOT NULL,
                maintenance_date TEXT NOT NULL,
                issue TEXT NOT NULL,
                description TEXT NOT NULL,
                action TEXT NOT NULL,
                technician TEXT NOT NULL,
                status TEXT NOT NULL
            )
            """
        )
        connection.executemany(
            """
            INSERT INTO maintenance_logs
                (equipment_id, maintenance_date, issue, description, action, technician, status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            LOGS,
        )
        connection.commit()
    print(f"Seeded {len(LOGS)} maintenance logs into {DB_PATH}")


if __name__ == "__main__":
    seed_database()
