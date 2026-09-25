"""Pydantic request and response schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SensorReading(BaseModel):
    temperature_c: float | None = Field(None, ge=-100, le=1000, description="Ambient temperature in °C.")
    process_temperature_c: float | None = Field(None, ge=-100, le=1000, description="Process temperature in °C.")
    pressure_bar: float | None = Field(None, ge=0, le=10000, description="Pressure in bar.")
    vibration_mm_s: float | None = Field(None, ge=0, le=1000, description="Vibration velocity in mm/s.")
    current_a: float | None = Field(None, ge=0, le=10000, description="Motor current in amperes.")
    power_kw: float | None = Field(None, ge=0, le=100000, description="Power in kW.")
    rotational_speed_rpm: float | None = Field(None, ge=0, le=100000, description="Rotational speed in RPM.")
    torque_nm: float | None = Field(None, ge=0, le=100000, description="Torque in N·m.")
    tool_wear_min: float | None = Field(None, ge=0, le=10000, description="Tool wear in minutes.")


class MaintenanceAlert(BaseModel):
    severity: str = Field(..., description="Warning or Critical.")
    feature: str = Field(..., description="Sensor feature that triggered the alert.")
    value: float = Field(..., description="Observed sensor value.")
    threshold: float = Field(..., description="Threshold used by the rule.")
    component: str = Field(..., description="Affected machine component.")
    action: str = Field(..., description="Recommended action.")
    urgency_score: int = Field(..., ge=0, le=100, description="Urgency from 0 to 100.")


class DigitalTwinOutput(BaseModel):
    failure_probability: float = Field(..., ge=0, le=1)
    status: str = Field(...)
    alerts: list[MaintenanceAlert] = Field(default_factory=list)
    features_used: list[str] = Field(default_factory=list)
    features_supplied: list[str] = Field(default_factory=list)
    features_imputed: list[str] = Field(default_factory=list)


class PipelineResult(BaseModel):
    digital_twin_state: dict[str, Any] = Field(...)
