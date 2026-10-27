"""Reason codes: known failure modes emit structured, renderable reasons."""

from __future__ import annotations

import pytest

from app.optimization.baseline import baseline_plan
from app.optimization.dispatch import evaluate_candidate
from app.optimization.repositioning import plan_repositioning
from app.schemas.reason_codes import Reason, ReasonCode
from app.simulation.charging import ChargingEconomics, evaluate_charger
from app.simulation.vehicle_dynamics import PhysicsInputError, loaded_mass_kg


def charging_economics() -> ChargingEconomics:
    return ChargingEconomics(
        queue_cost_inr_per_min=6.0,
        detour_cost_inr_per_km=12.0,
        lost_productive_inr_per_min=9.0,
        energy_inr_per_kwh=9.0,
        dc_surcharge_inr_per_kwh=2.5,
        wear_inr_per_kwh=1.5,
    )


def test_catalog_values_are_their_own_names():
    for member in ReasonCode:
        assert member.value == member.name
        assert member.value.isupper()
    assert len(ReasonCode) >= 40


def test_capacity_overload_reports_its_code(vehicle_params):
    with pytest.raises(PhysicsInputError) as excinfo:
        loaded_mass_kg(vehicle_params, vehicle_params.passenger_capacity + 2, 0.0)
    assert excinfo.value.reason_code == ReasonCode.PASSENGER_CAPACITY_EXCEEDED
    with pytest.raises(PhysicsInputError) as excess:
        loaded_mass_kg(vehicle_params, 0, vehicle_params.max_payload_kg * 3)
    assert excess.value.reason_code == ReasonCode.PAYLOAD_EXCEEDED


def test_offline_charger_reports_its_code(charger_factory, bev_battery):
    estimation = evaluate_charger(
        charger=charger_factory(status="OFFLINE"),
        battery=bev_battery,
        economics=charging_economics(),
        soc_now=0.5,
        target_soc=0.9,
        detour_km=1.0,
        plug_count_in_use=1,
        hour_of_day=10,
        tariff_bands=(16.0, 7.0),
        tariff_windows=(18, 23, 0, 6),
    )
    assert ReasonCode.CHARGER_OFFLINE.value in estimation.reason_codes


def test_low_soc_candidate_reports_reserve_reason(demo_scenario, cfg):
    vehicle = demo_scenario.vehicles[0].model_copy(update={"soc": 0.03})
    request = max(demo_scenario.requests, key=lambda r: r.luggage_mass_kg + r.passenger_count)
    candidate = evaluate_candidate(demo_scenario, vehicle, request, cfg=cfg)
    assert candidate.feasible is False
    combined = " ".join(candidate.reason_codes)
    assert "RESERVE" in combined or "VEHICLE_BELOW" in combined


def test_offline_vehicle_is_reported_as_unavailable(demo_scenario, cfg):
    vehicle = demo_scenario.vehicles[0].model_copy(update={"availability": "OFFLINE"})
    request = demo_scenario.requests[0]
    candidate = evaluate_candidate(demo_scenario, vehicle, request, cfg=cfg)
    assert candidate.feasible is False
    assert ReasonCode.VEHICLE_UNAVAILABLE.value in candidate.reason_codes


def test_capacity_rejection_codes_flow_into_the_candidate(demo_scenario, cfg):
    request = max(demo_scenario.requests, key=lambda r: r.luggage_units)
    crowded = request.model_copy(
        update={"passenger_count": 8, "luggage_units": 12, "luggage_mass_kg": 300.0}
    )
    candidate = evaluate_candidate(demo_scenario, demo_scenario.vehicles[0], crowded, cfg=cfg)
    codes = set(candidate.reason_codes)
    assert codes & {
        ReasonCode.PASSENGER_CAPACITY_EXCEEDED.value,
        ReasonCode.LUGGAGE_CAPACITY_EXCEEDED.value,
        ReasonCode.PAYLOAD_EXCEEDED.value,
    }


def test_baseline_explains_served_and_unserved_outcomes(demo_scenario, cfg):
    outcome = baseline_plan(demo_scenario, cfg=cfg)
    assert ReasonCode.BASELINE_SELECTED_NEAREST.value in outcome.reason_codes
    for request_id in outcome.unserved:
        assert (
            f"{ReasonCode.REQUEST_UNASSIGNED_BASELINE.value}:{request_id}"
            in outcome.reason_codes
        )


def test_repositioning_code_set_covers_accept_and_decline(demo_scenario, cfg):
    from app.simulation.scenario_generator import demand_records_for

    records = demand_records_for(demo_scenario, cfg)
    plan = plan_repositioning(demo_scenario, records, cfg=cfg)
    known = {code.value for code in ReasonCode}
    for reason in plan.skipped.values():
        assert reason in known
    for code in plan.reason_codes:
        assert code in known


def test_powertrain_reason_codes_are_selectable_and_stable():
    assert ReasonCode.POWERTRAIN_X_NOT_JUSTIFIED.value == "POWERTRAIN_X_NOT_JUSTIFIED"
    assert ReasonCode.POWERTRAIN_E_PREFERRED_COST.value in {c.value for c in ReasonCode}


def test_reason_render_is_identifiable_not_prose():
    reason = Reason(
        code=ReasonCode.CHARGER_QUEUE_ADJUSTED_PREFERRED,
        subject="CH-02",
        detail="queue 3, wait 24 min",
        value=24.0,
        unit="min",
    )
    rendered = reason.render()
    assert rendered.startswith(ReasonCode.CHARGER_QUEUE_ADJUSTED_PREFERRED.value)
    assert "CH-02" in rendered
    assert "queue 3" not in rendered  # detail is data, not prose in the code label

    bare = Reason(code=ReasonCode.SOLVER_TIMEOUT_BASELINE_FALLBACK)
    assert bare.render() == ReasonCode.SOLVER_TIMEOUT_BASELINE_FALLBACK.value


def test_solver_failure_codes_exist_for_every_documented_path():
    expected = {
        ReasonCode.SOLVER_TIMEOUT_BASELINE_FALLBACK.value,
        ReasonCode.SOLVER_ERROR_BASELINE_FALLBACK.value,
        ReasonCode.SOLVER_NO_FEASIBLE_PLAN.value,
        ReasonCode.SOLVER_FALLBACK_DISABLED.value,
    }
    assert expected <= {code.value for code in ReasonCode}


def test_scenario_fault_codes_exist():
    assert {
        ReasonCode.HEAT_WAVE_APPLIED.value,
        ReasonCode.TRAFFIC_SPIKE_APPLIED.value,
        ReasonCode.DEMAND_SURGE_APPLIED.value,
        ReasonCode.LOW_SOH_VEHICLE_IN_FLEET.value,
    } <= {code.value for code in ReasonCode}


def test_cabin_and_vision_codes_exist():
    assert {
        ReasonCode.MODE_PROPOSED_AIRPORT.value,
        ReasonCode.MODE_OVERRIDE_APPLIED.value,
        ReasonCode.CABIN_LOW_CONFIDENCE_MANUAL_FALLBACK.value,
        ReasonCode.CABIN_STALE_MANUAL_CONFIRMATION.value,
        ReasonCode.CABIN_STATE_DEBOUNCED.value,
    } <= {code.value for code in ReasonCode}


def test_reason_is_a_typed_entity_rejecting_unknown_codes():
    with pytest.raises(Exception):
        Reason(code="NOT_A_CODE")


def test_every_code_carries_at_most_one_subject_suffix_pattern():
    # Codes that identify a subject are rendered as CODE:SUBJECT; a raw catalog
    # entry never contains a space (prose lives in the decision explanation).
    for member in ReasonCode:
        assert " " not in member.value
        assert "-" not in member.value or member.value.startswith("CHARGER_CH") is False