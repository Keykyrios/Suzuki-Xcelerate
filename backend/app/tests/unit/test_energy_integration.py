"""Energy integration: contributor identity and SOC/energy consistency."""

from __future__ import annotations

import pytest

from app.schemas.common import ConstraintStatus


def test_contributor_identity_holds_for_a_full_route(run_route):
    result = run_route()
    contributors = result.energy_contributors_kwh
    net = (
        contributors["traction_kwh"]
        + contributors["hvac_kwh"]
        + contributors["auxiliary_kwh"]
        + contributors["regen_recovered_kwh"]  # stored as a negative quantity
    )
    assert net == pytest.approx(result.route_energy_kwh, abs=0.01)


def test_soc_change_matches_integrated_energy(run_route, bev_battery, cfg):
    capacity = bev_battery.usable_energy_kwh
    result = run_route(departure_soc=0.8)
    soc_drop = result.departure_soc - result.arrival_soc
    implied = result.route_energy_kwh / capacity
    assert soc_drop == pytest.approx(implied, abs=2e-4)


def test_cumulative_energy_series_is_consistent_with_the_headline(run_route):
    result = run_route()
    series = result.cumulative_energy_kwh
    assert series, "cumulative energy series must not be empty"
    assert series[0] >= 0.0
    assert series[-1] == pytest.approx(result.route_energy_kwh, abs=0.01)


def test_series_lengths_match(run_route):
    result = run_route()
    lengths = {
        len(result.time_s),
        len(result.speed_mps),
        len(result.acceleration_m_s2),
        len(result.wheel_power_kw),
        len(result.traction_battery_power_kw),
        len(result.hvac_power_kw),
        len(result.auxiliary_power_kw),
        len(result.regen_power_kw),
        len(result.cumulative_energy_kwh),
        len(result.soc_trace),
    }
    assert len(lengths) == 1


def test_time_and_speed_series_are_sane(run_route):
    result = run_route()
    assert all(t > 0 for t in result.time_s)
    assert result.time_s == sorted(result.time_s)
    assert all(v >= 0.0 for v in result.speed_mps)
    assert all(0.0 <= s <= 1.0 for s in result.soc_trace)


def test_soc_never_exceeds_its_upper_bound(run_route):
    result = run_route(departure_soc=0.98)
    assert max(result.soc_trace) <= 1.0


def test_reported_energy_per_100km_matches_distance(run_route):
    result = run_route()
    expected = result.route_energy_kwh / result.distance_km * 100.0
    assert result.energy_kwh_per_100km == pytest.approx(expected, rel=1e-6, abs=0.05)


def test_route_result_records_model_provenance(run_route, cfg):
    result = run_route()
    assert result.config_hash == cfg.hash
    assert result.evidence_status.value == "MODEL_ESTIMATE"
    assert result.calibrated is False
    assert result.model_version.startswith("nexus7-physics")


def test_reserve_check_passes_for_a_generously_charged_short_trip(run_route, cfg):
    result = run_route(departure_soc=0.9)
    assert result.constraint_checks["arrival_soc_reserve"] == ConstraintStatus.PASS
    assert result.reserve_margin > 0.0


def test_route_simulation_is_deterministic(run_route):
    first = run_route()
    second = run_route()
    assert first.model_dump() == second.model_dump()