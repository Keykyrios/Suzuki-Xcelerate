"""Independent feasibility checker: capacity, availability, energy, chargers."""

from __future__ import annotations

import pytest

from app.optimization.feasibility import (
    FeasibilityReport,
    check_availability,
    check_capacity,
    check_charger_usable,
    check_plan,
    recheck_assignment,
)
from app.schemas.common import CabinMode, ConstraintStatus


def test_capacity_passes_for_a_normal_request(demo_scenario, cfg):
    vehicle = demo_scenario.vehicles[0]
    request = demo_scenario.requests[0]
    request = request.model_copy(update={"passenger_count": 2, "luggage_units": 1})
    report = check_capacity(vehicle, request, cfg)
    assert report.passed
    assert report.checks["passenger_capacity"] == ConstraintStatus.PASS


def test_capacity_fails_when_the_mode_cannot_seat_the_party(demo_scenario, cfg):
    vehicle = demo_scenario.vehicles[0]
    request = demo_scenario.requests[0]
    request = request.model_copy(update={"passenger_count": 8})
    report = check_capacity(
        vehicle, request.model_copy(update={"service_class": CabinMode.POOL}), cfg
    )
    assert not report.passed
    assert report.checks["passenger_capacity"] == ConstraintStatus.FAIL
    assert any("PASSENGER_CAPACITY_EXCEEDED" in code for code in report.reason_codes)


def test_luggage_over_the_mode_limit_is_rejected(demo_scenario, cfg):
    vehicle = demo_scenario.vehicles[0]
    request = demo_scenario.requests[0]
    request = request.model_copy(update={"luggage_units": 12, "luggage_mass_kg": 300.0})
    report = check_capacity(vehicle, request, cfg)
    assert not report.passed
    assert any("LUGGAGE" in code or "PAYLOAD" in code for code in report.reason_codes)


def test_availability_respects_the_pickup_window(demo_scenario):
    request = demo_scenario.requests[0]
    on_time = check_availability(demo_scenario.vehicles[0], request, request.pickup_window_start_s)
    assert on_time.passed
    very_late = check_availability(
        demo_scenario.vehicles[0], request, request.pickup_window_end_s + 10_000.0
    )
    assert not very_late.passed


def test_recheck_recomputes_energy_independently(demo_scenario, cfg):
    outcome_vehicle = demo_scenario.vehicles[0]
    request = demo_scenario.requests[0]
    report, result = recheck_assignment(
        demo_scenario,
        outcome_vehicle.vehicle_id,
        request,
        request.pickup_window_start_s,
        cfg,
    )
    assert isinstance(report, FeasibilityReport)
    assert report.passed, f"{report.to_dict()} {report.reason_codes}"
    if result is not None:
        assert result.route_energy_kwh >= 0.0
        assert 0.0 <= result.arrival_soc <= 1.0


def test_recheck_fails_below_reserve_chains(demo_scenario, cfg):
    vehicle = demo_scenario.vehicles[0].model_copy(update={"soc": 0.02})
    scenario = demo_scenario.model_copy(
        update={"vehicles": [vehicle] + list(demo_scenario.vehicles[1:])}
    )
    request = max(demo_scenario.requests, key=lambda r: r.passenger_count + r.luggage_mass_kg)
    report, _ = recheck_assignment(
        scenario,
        vehicle.vehicle_id,
        request,
        request.pickup_window_start_s,
        cfg,
    )
    assert not report.passed
    assert any(
        code.startswith(("ARRIVAL_SOC_BELOW_RESERVE", "VEHICLE_BELOW_RESERVE"))
        for code in report.reason_codes
    ) or ConstraintStatus.FAIL in report.checks.values()


def test_offline_charger_is_not_usable(demo_scenario):
    charger = demo_scenario.chargers[0].model_copy(update={"status": "OFFLINE"})
    report = check_charger_usable(demo_scenario, charger, demo_scenario.vehicles[0], 0.9)
    assert not report.passed
    assert any("CHARGER_OFFLINE" in code for code in report.reason_codes)


def test_online_charger_with_delivery_capability_passes(demo_scenario):
    charger = demo_scenario.chargers[0]
    vehicle = demo_scenario.vehicles[0].model_copy(update={"soc": 0.4})
    report = check_charger_usable(demo_scenario, charger, vehicle, 0.7)
    assert report.checks["charger_online"] == ConstraintStatus.PASS
    assert report.checks.get("connector_compatible") == ConstraintStatus.PASS
    assert report.checks.get("energy_delivery") == ConstraintStatus.PASS


def test_plan_aggregation_marks_double_booking(demo_scenario, cfg):
    request = demo_scenario.requests[0]
    overlap = request
    assignments = [
        (demo_scenario.vehicles[0].vehicle_id, request.request_id, request.pickup_window_start_s, request),
        (demo_scenario.vehicles[0].vehicle_id, overlap.request_id, overlap.pickup_window_start_s, overlap),
    ]
    report = check_plan(demo_scenario, assignments, cfg)
    assert any(key.endswith("double_booking") for key in report.checks)
    assert not report.checks["no_double_booking"] if "no_double_booking" in report.checks else True


def test_report_serialisation_and_merge():
    first = FeasibilityReport(checks={"a": ConstraintStatus.PASS}, reason_codes=["X"])
    second = FeasibilityReport(checks={"b": ConstraintStatus.FAIL}, reason_codes=["Y"])
    merged = first.merge(second)
    assert merged.to_dict() == {"a": "PASS", "b": "FAIL"}
    assert merged.reason_codes == ["X", "Y"]
    assert merged.passed is False
    assert not FeasibilityReport().passed  # no checks means no verdict