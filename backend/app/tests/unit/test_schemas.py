"""Schema validation: field contracts, enum and range rejection, strictness."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.schemas.charger import ChargerState
from app.schemas.common import (
    CabinMode,
    ChargerStatus,
    ConstraintStatus,
    EvidenceLabel,
    PowertrainType,
    SolverStatus,
)
from app.schemas.decision import Decision, DecisionPayload, FleetMetrics
from app.schemas.events import RunManifest, SimulatedAction, SimulatedActionType
from app.schemas.reason_codes import Reason, ReasonCode
from app.schemas.scenario import Scenario, TripRequest
from app.schemas.vehicle import CabinState, Route, RouteResult, VehicleState

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def vehicle_state(**overrides) -> dict:
    payload = {
        "vehicle_id": "V01",
        "powertrain": PowertrainType.NEXUS_E.value,
        "availability": "AVAILABLE",
        "cabin_mode": CabinMode.COMMUTE.value,
        "passenger_count": 2,
        "luggage_units": 1,
        "luggage_mass_kg": 12.0,
        "soc": 0.8,
        "soh": 0.97,
        "battery_temp_c": 28.0,
        "current_zone_id": "Z01",
        "next_available_time_s": 0.0,
        "odometer_km": 1200.0,
        "charge_limit_soc": 0.9,
        "model_version": "nexus7-physics-1.0.0",
        "state_timestamp_utc": NOW,
        "state_version": 1,
    }
    payload.update(overrides)
    return payload


def charger_state(**overrides) -> dict:
    payload = {
        "charger_id": "CH01",
        "node_id": "N03",
        "status": ChargerStatus.ONLINE.value,
        "plug_count": 2,
        "plugs_in_use": 1,
        "rated_power_kw": 60.0,
        "site_limit_kw": 120.0,
        "tariff_inr_per_kwh": 12.5,
        "connectors": ["CCS2"],
        "queue_length": 1,
        "queue_estimate_min": 8.0,
        "last_updated_utc": NOW,
    }
    payload.update(overrides)
    return payload


def cabin_state(**overrides) -> dict:
    payload = {
        "vehicle_id": "V01",
        "mode": CabinMode.AIRPORT.value,
        "passenger_count": 3,
        "luggage_units": 2,
        "luggage_mass_kg": 24.0,
        "confidence": 0.9,
        "source": "camera",
        "timestamp_utc": NOW,
    }
    payload.update(overrides)
    return payload


# -- accepted entities -----------------------------------------------------
def test_vehicle_state_round_trip():
    state = VehicleState.model_validate(vehicle_state())
    assert state.soc == pytest.approx(0.8)
    back = VehicleState.model_validate(state.model_dump(mode="json"))
    assert back == state


def test_vehicle_state_exposes_unit_explicit_fields():
    fields = set(VehicleState.model_fields)
    for expected in ("battery_temp_c", "luggage_mass_kg", "odometer_km", "next_available_time_s"):
        assert expected in fields
    assert "temperature" not in fields  # bare, unit-less names are rejected by review


def test_cabin_state_contract():
    state = CabinState.model_validate(cabin_state())
    assert state.confidence <= 1.0
    assert state.override_flag is False
    assert state.luggage_mass_kg <= 400.0


def test_charger_state_contract():
    charger = ChargerState.model_validate(charger_state())
    assert charger.plug_count >= 1
    assert charger.rated_power_kw > 0.0
    assert charger.is_online


def test_trip_request_contract():
    request = TripRequest.model_validate(
        {
            "request_id": "R01",
            "pickup_zone_id": "Z01",
            "dropoff_zone_id": "Z05",
            "request_time_s": 10.0,
            "pickup_window_start_s": 0.0,
            "pickup_window_end_s": 600.0,
            "passenger_count": 2,
            "luggage_units": 1,
            "luggage_mass_kg": 15.0,
            "service_class": CabinMode.COMMUTE.value,
            "expected_fare_inr": 240.0,
            "cancellation_penalty_inr": 50.0,
        }
    )
    assert request.passenger_count <= 8
    assert request.status.value == "PENDING"


def test_route_and_scenario_entities_present():
    route_fields = set(Route.model_fields)
    assert {
        "edge_ids",
        "distance_km",
        "speed_profile_kmh",
        "grade_profile_percent",
        "traffic_factor",
        "feasible",
        "feasibility_reasons",
    } <= route_fields
    scenario_fields = set(Scenario.model_fields)
    assert {
        "scenario_id",
        "seed",
        "graph",
        "vehicles",
        "chargers",
        "requests",
        "environment",
        "demand_context",
        "faults",
        "tariffs_inr_per_kwh",
    } <= scenario_fields


def test_decision_payload_and_metrics_entities_exist():
    decision_fields = set(Decision.model_fields)
    assert {
        "run_id",
        "scenario_id",
        "status",
        "decision",
        "assignments",
        "charging_actions",
        "repositioning_actions",
        "constraints",
        "reason_codes",
        "risk",
        "solver",
        "baseline_metrics",
        "optimized_metrics",
        "evidence_status",
        "stale",
    } <= decision_fields
    payload_fields = set(DecisionPayload.model_fields)
    assert {"expected_contribution_inr", "risk_level", "reserve_margin_pct"} <= payload_fields
    assert {"completed_requests", "productive_km_ratio", "contribution_inr"} <= set(
        FleetMetrics.model_fields
    )


# -- rejection -------------------------------------------------------------
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("soc", 1.5),
        ("soc", -0.1),
        ("passenger_count", 9),
        ("luggage_units", 13),
        ("soh", 1.5),
        ("odometer_km", -1.0),
        ("powertrain", "NEXUS_Z"),
        ("cabin_mode", "LOUNGE"),
        ("availability", "MAYBE"),
    ],
)
def test_vehicle_state_rejects_out_of_range_and_unknown_enums(field, value):
    with pytest.raises(ValidationError):
        VehicleState.model_validate(vehicle_state(**{field: value}))


def test_vehicle_state_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        VehicleState.model_validate(vehicle_state(invented_field=1))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("plug_count", 0),
        ("rated_power_kw", 0.0),
        ("queue_length", 51),
        ("status", "BROKEN"),
        ("tariff_inr_per_kwh", -1.0),
    ],
)
def test_charger_state_rejects_invalid_values(field, value):
    with pytest.raises(ValidationError):
        ChargerState.model_validate(charger_state(**{field: value}))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("confidence", 1.2),
        ("passenger_count", -1),
        ("luggage_mass_kg", 500.0),
        ("source", "cloud"),
    ],
)
def test_cabin_state_rejects_invalid_values(field, value):
    with pytest.raises(ValidationError):
        CabinState.model_validate(cabin_state(**{field: value}))


def test_simulated_action_is_always_simulated_and_never_hardware():
    action = SimulatedAction.model_validate(
        {
            "action_id": "act-1",
            "scenario_id": "demo",
            "run_id": None,
            "target_type": "vehicle",
            "target_id": "V01",
            "action_type": SimulatedActionType.CABIN_MODE_CHANGE.value,
            "requested_state": {"mode": "POOL"},
            "previous_state": {"mode": "COMMUTE"},
            "resulting_state": {"mode": "POOL"},
            "simulated_at_utc": NOW,
            "duration_s": 0.0,
        }
    )
    assert action.status == "SIMULATED"
    assert action.hardware_commanded is False
    assert len(SimulatedActionType) == 9  # the nine required action types


def test_simulated_action_rejects_hardware_commanded_true():
    with pytest.raises(ValidationError):
        SimulatedAction.model_validate(
            {
                "action_id": "act-2",
                "scenario_id": "demo",
                "run_id": None,
                "target_type": "vehicle",
                "target_id": "V01",
                "action_type": SimulatedActionType.VEHICLE_MOVE.value,
                "requested_state": {},
                "previous_state": {},
                "resulting_state": {},
                "simulated_at_utc": NOW,
                "duration_s": 1.0,
                "hardware_commanded": True,
            }
        )


def test_run_manifest_contract():
    manifest = RunManifest.model_validate(
        {
            "run_id": "run_000001",
            "timestamp_utc": NOW,
            "seed": 7,
            "git_commit": "abc",
            "config_hash": "deadbeef",
            "model_versions": {"physics": "1"},
            "solver_name": "CP-SAT",
            "solver_version": "9.14",
            "solver_status": SolverStatus.FEASIBLE.value,
            "scenario_id": "demo",
            "status": "completed",
            "headline_metrics": {"contribution_inr": 1.0},
            "evidence_status": EvidenceLabel.MODEL_ESTIMATE.value,
        }
    )
    assert manifest.seed == 7
    with pytest.raises(ValidationError):
        RunManifest.model_validate(manifest.model_dump() | {"git_commit": 42})


# -- enums and reason-code catalog ----------------------------------------
def test_solver_status_enum_covers_failure_modes():
    statuses = {s.value for s in SolverStatus}
    assert {"OPTIMAL", "FEASIBLE", "NO_FEASIBLE_PLAN", "TIMEOUT", "ERROR", "BASELINE_FALLBACK"} <= statuses


def test_constraint_status_enum():
    assert {c.value for c in ConstraintStatus} == {"PASS", "FAIL", "SKIPPED"}


def test_reason_catalog_values_match_member_names():
    for member in ReasonCode:
        assert member.value == member.name


def test_reason_render_includes_subject():
    reason = Reason(code=ReasonCode.CHARGER_OFFLINE, subject="CH02")
    assert "CHARGER_OFFLINE" in reason.render()
    assert "CH02" in reason.render()


def test_evidence_status_enum_matches_labels():
    decision_label = {e.value for e in EvidenceLabel}
    config_label = {e.value for e in __import__("app.config.evidence", fromlist=["EvidenceStatus"]).EvidenceStatus}
    assert decision_label == config_label