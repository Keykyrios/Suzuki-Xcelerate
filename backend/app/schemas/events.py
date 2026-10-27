"""Event, simulated-action, and run-manifest entities."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import Field

from app.schemas.common import StrictModel


class SimulatedActionType(str, Enum):
    CABIN_MODE_CHANGE = "CABIN_MODE_CHANGE"
    SEAT_LAYOUT_CHANGE = "SEAT_LAYOUT_CHANGE"
    LUGGAGE_LAYOUT_CHANGE = "LUGGAGE_LAYOUT_CHANGE"
    VEHICLE_MOVE = "VEHICLE_MOVE"
    CHARGER_CONNECT = "CHARGER_CONNECT"
    CHARGER_DISCONNECT = "CHARGER_DISCONNECT"
    HVAC_ZONE_CHANGE = "HVAC_ZONE_CHANGE"
    POWERTRAIN_SELECTION = "POWERTRAIN_SELECTION"
    MAINTENANCE_FLAG = "MAINTENANCE_FLAG"


class SimulatedAction(StrictModel):
    """A virtual vehicle-side action. Never maps to physical hardware."""

    action_id: str
    scenario_id: str
    run_id: str | None
    target_type: Literal["vehicle", "charger", "cabin"]
    target_id: str
    action_type: SimulatedActionType
    requested_state: dict[str, Any]
    previous_state: dict[str, Any]
    resulting_state: dict[str, Any]
    simulated_at_utc: datetime
    duration_s: float = Field(ge=0.0)
    status: Literal["SIMULATED"] = "SIMULATED"
    hardware_commanded: Literal[False] = False
    reason_codes: list[str] = Field(default_factory=list)
    originating_decision_id: str | None = None


class StateSnapshot(StrictModel):
    """Full canonical state handed to clients on connect and after resets."""

    state_version: int = Field(ge=0)
    timestamp_utc: datetime
    scenario_id: str
    run_id: str | None
    vehicles: list[dict[str, Any]]
    chargers: list[dict[str, Any]]
    requests: list[dict[str, Any]]
    graph: dict[str, Any]
    environment: dict[str, Any]
    decision: dict[str, Any] | None
    cabin_states: dict[str, dict[str, Any]]
    recent_actions: list[dict[str, Any]]
    recommendation_stale: bool
    solver_status: str | None


class EventType(str, Enum):
    STATE_SNAPSHOT = "STATE_SNAPSHOT"
    STATE_DELTA = "STATE_DELTA"
    DECISION_UPDATED = "DECISION_UPDATED"
    SIMULATION_ACTION = "SIMULATION_ACTION"
    CHARGER_UPDATED = "CHARGER_UPDATED"
    VEHICLE_UPDATED = "VEHICLE_UPDATED"
    SOLVER_STATUS = "SOLVER_STATUS"
    CABIN_STATE_UPDATED = "CABIN_STATE_UPDATED"


class EventEnvelope(StrictModel):
    event_id: str
    event_type: EventType
    state_version: int = Field(ge=0)
    timestamp_utc: datetime
    scenario_id: str
    run_id: str | None = None
    payload: dict[str, Any]


class RunManifest(StrictModel):
    run_id: str
    timestamp_utc: datetime
    seed: int
    git_commit: str
    config_hash: str
    model_versions: dict[str, str]
    solver_name: str
    solver_version: str
    solver_status: str
    scenario_id: str
    status: str
    headline_metrics: dict[str, float]
    evidence_status: str
    notes: str = ""
