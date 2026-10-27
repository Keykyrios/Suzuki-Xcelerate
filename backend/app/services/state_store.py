"""Canonical in-memory state shared by every API surface.

One store holds the active scenario, cabin states, battery trackers, the last
decision, and recent simulated actions. Every mutation bumps a monotonic state
version and publishes a versioned event, which is what the WebSocket layer
streams and what the UI uses to reject stale updates.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from app.schemas.charger import ChargerState
from app.schemas.common import utcnow
from app.schemas.decision import Decision
from app.schemas.events import EventEnvelope, EventType, SimulatedAction, StateSnapshot
from app.schemas.scenario import Scenario
from app.schemas.vehicle import CabinState, VehicleState
from app.simulation.battery import BatteryTracker
from app.simulation.model_config import battery_params

Listener = Callable[[EventEnvelope], None]


@dataclass
class StateStore:
    """Thread-safe canonical state with versioned change notification."""

    scenario: Scenario | None = None
    run_id: str | None = None
    decision: Decision | None = None
    cabin_states: dict[str, CabinState] = field(default_factory=dict)
    battery_trackers: dict[str, BatteryTracker] = field(default_factory=dict)
    recent_actions: list[SimulatedAction] = field(default_factory=list)
    recommendation_stale: bool = False
    state_version: int = 0
    last_decision_wall_utc: datetime | None = None
    explanations: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._lock = threading.RLock()
        self._listeners: list[Listener] = []
        self._event_log: list[EventEnvelope] = []

    # -- subscriptions ----------------------------------------------------
    def subscribe(self, listener: Listener) -> Callable[[], None]:
        with self._lock:
            self._listeners.append(listener)

        def unsubscribe() -> None:
            with self._lock:
                if listener in self._listeners:
                    self._listeners.remove(listener)

        return unsubscribe

    def set_explanation(self, run_id: str, text: str) -> None:
        with self._lock:
            self.explanations[run_id] = text

    def get_explanation(self, run_id: str) -> str | None:
        with self._lock:
            return self.explanations.get(run_id)

    def recent_events(self, limit: int = 50) -> list[EventEnvelope]:
        with self._lock:
            return list(self._event_log[-limit:])

    # -- loading ----------------------------------------------------------
    def load_scenario(self, scenario: Scenario, run_id: str | None = None) -> None:
        with self._lock:
            # A loaded scenario replaces canonical state, so it must advance the
            # version: clients reject anything not newer than what they applied.
            self.state_version += 1
            self.scenario = scenario
            self.run_id = run_id
            self.decision = None
            self.cabin_states = {}
            self.recent_actions = []
            self.recommendation_stale = False
            self.battery_trackers = {
                vehicle.vehicle_id: BatteryTracker(
                    vehicle_id=vehicle.vehicle_id,
                    params=battery_params(vehicle.powertrain),
                    soc=vehicle.soc,
                    soh=vehicle.soh,
                    temperature_c=vehicle.battery_temp_c,
                )
                for vehicle in scenario.vehicles
            }
            self._publish(EventType.STATE_SNAPSHOT, self._snapshot_payload())

    # -- mutations --------------------------------------------------------
    def update_vehicle(self, vehicle_id: str, updates: dict[str, Any]) -> VehicleState:
        with self._lock:
            scenario = self._require_scenario()
            vehicles = []
            updated: VehicleState | None = None
            for vehicle in scenario.vehicles:
                if vehicle.vehicle_id == vehicle_id:
                    merged = vehicle.model_copy(update={**updates, "state_version": vehicle.state_version + 1,
                                                        "state_timestamp_utc": utcnow()})
                    updated = merged
                    vehicles.append(merged)
                else:
                    vehicles.append(vehicle)
            if updated is None:
                raise KeyError(f"unknown vehicle {vehicle_id}")
            self.scenario = scenario.model_copy(update={"vehicles": vehicles})
            self._bump(EventType.VEHICLE_UPDATED, {"vehicle": updated.model_dump(mode="json")})
            return updated

    def update_charger(self, charger_id: str, updates: dict[str, Any]) -> ChargerState:
        with self._lock:
            scenario = self._require_scenario()
            chargers = []
            updated: ChargerState | None = None
            for charger in scenario.chargers:
                if charger.charger_id == charger_id:
                    candidate = {**charger.model_dump(mode="json"), **updates}
                    merged = ChargerState.model_validate(candidate).model_copy(
                        update={"last_updated_utc": utcnow()}
                    )
                    updated = merged
                    chargers.append(merged)
                else:
                    chargers.append(charger)
            if updated is None:
                raise KeyError(f"unknown charger {charger_id}")
            self.scenario = scenario.model_copy(update={"chargers": chargers})
            self._bump(EventType.CHARGER_UPDATED, {"charger": updated.model_dump(mode="json")})
            return updated

    def set_cabin_state(self, cabin_state: CabinState) -> None:
        with self._lock:
            self.cabin_states[cabin_state.vehicle_id] = cabin_state
            self._bump(
                EventType.CABIN_STATE_UPDATED,
                {"cabin_state": cabin_state.model_dump(mode="json")},
            )

    def record_action(self, action: SimulatedAction) -> None:
        with self._lock:
            self.recent_actions.append(action)
            self.recent_actions = self.recent_actions[-100:]
            self._bump(EventType.SIMULATION_ACTION, {"action": action.model_dump(mode="json")})

    def apply_decision(self, decision: Decision) -> None:
        with self._lock:
            self.decision = decision
            self.run_id = decision.run_id
            self.recommendation_stale = decision.stale
            self.last_decision_wall_utc = utcnow()
            self._publish(EventType.DECISION_UPDATED, {"decision": decision.model_dump(mode="json")})

    def set_stale(self, stale: bool, reason: str = "") -> None:
        with self._lock:
            if self.recommendation_stale == stale:
                return
            self.recommendation_stale = stale
            self._bump(EventType.SOLVER_STATUS, {"stale": stale, "reason": reason})

    def set_run_id(self, run_id: str) -> None:
        with self._lock:
            self.run_id = run_id

    # -- reads ------------------------------------------------------------
    def require_scenario(self) -> Scenario:
        with self._lock:
            return self._require_scenario()

    def snapshot(self) -> StateSnapshot:
        with self._lock:
            return StateSnapshot.model_validate(self._snapshot_payload())

    def _require_scenario(self) -> Scenario:
        if self.scenario is None:
            raise RuntimeError("no scenario is loaded")
        return self.scenario

    def _snapshot_payload(self) -> dict[str, Any]:
        scenario = self.scenario
        if scenario is None:
            return {
                "state_version": self.state_version,
                "timestamp_utc": utcnow(),
                "scenario_id": "",
                "run_id": self.run_id,
                "vehicles": [],
                "chargers": [],
                "requests": [],
                "graph": {},
                "environment": {},
                "decision": None,
                "cabin_states": {},
                "recent_actions": [],
                "recommendation_stale": self.recommendation_stale,
                "solver_status": None,
            }
        return {
            "state_version": self.state_version,
            "timestamp_utc": utcnow(),
            "scenario_id": scenario.scenario_id,
            "run_id": self.run_id,
            "vehicles": [v.model_dump(mode="json") for v in scenario.vehicles],
            "chargers": [c.model_dump(mode="json") for c in scenario.chargers],
            "requests": [r.model_dump(mode="json") for r in scenario.requests],
            "graph": scenario.graph.model_dump(mode="json"),
            "environment": scenario.environment.model_dump(mode="json"),
            "decision": self.decision.model_dump(mode="json") if self.decision else None,
            "cabin_states": {
                k: v.model_dump(mode="json") for k, v in self.cabin_states.items()
            },
            "recent_actions": [a.model_dump(mode="json") for a in self.recent_actions[-20:]],
            "recommendation_stale": self.recommendation_stale,
            "solver_status": self.decision.status.value if self.decision else None,
        }

    # -- publishing -------------------------------------------------------
    def _bump(self, event_type: EventType, payload: dict[str, Any]) -> None:
        self.state_version += 1
        self._publish(event_type, payload)

    def _publish(self, event_type: EventType, payload: dict[str, Any]) -> None:
        envelope = EventEnvelope(
            event_id=f"evt-{self.state_version}-{event_type.value}",
            event_type=event_type,
            state_version=self.state_version,
            timestamp_utc=utcnow(),
            scenario_id=self.scenario.scenario_id if self.scenario else "",
            run_id=self.run_id,
            payload=payload,
        )
        self._event_log.append(envelope)
        self._event_log = self._event_log[-200:]
        for listener in list(self._listeners):
            try:
                listener(envelope)
            except Exception:  # noqa: BLE001 - a broken subscriber must not break state
                pass


_STORE: StateStore | None = None


def get_state_store() -> StateStore:
    global _STORE
    if _STORE is None:
        _STORE = StateStore()
    return _STORE


def reset_state_store() -> StateStore:
    global _STORE
    _STORE = StateStore()
    return _STORE
