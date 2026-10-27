"""Regeneration bounds: fraction, motor limit, acceptance, headroom, low speed."""

from __future__ import annotations

import pytest

from app.simulation.battery import charge_acceptance_kw


def test_regen_never_exceeds_the_configured_peak(vehicle_params, run_route):
    result = run_route(departure_soc=0.6, distance_km=5.0, speed_kmh=60.0)
    ceiling = vehicle_params.regen_max_power_kw + 1e-3  # rounding tolerance
    assert max(result.regen_power_kw) <= ceiling


def test_regen_is_zero_during_cruise_and_acceleration(run_route):
    result = run_route(departure_soc=0.6)
    for power, accel, speed in zip(
        result.regen_power_kw, result.acceleration_m_s2, result.speed_mps
    ):
        if accel >= 0.0 and speed > 0.0:
            assert power == 0.0


def test_regen_is_cut_at_low_speed(vehicle_params, run_route):
    result = run_route(departure_soc=0.6, distance_km=1.5, speed_kmh=15.0)
    below_floor = [
        power
        for power, speed in zip(result.regen_power_kw, result.speed_mps)
        if 0.0 < speed < vehicle_params.regen_min_speed_m_s
    ]
    assert all(power == 0.0 for power in below_floor)
    assert max(result.regen_power_kw) > 0.0  # recovery still happens above the floor


def test_regen_respects_battery_acceptance_sample_by_sample(
    vehicle_params, bev_battery, numerics, run_route
):
    result = run_route(departure_soc=0.75, distance_km=5.0, speed_kmh=60.0)
    capacity_kwh = bev_battery.usable_energy_kwh
    for power, soc in zip(result.regen_power_kw, result.soc_trace):
        acceptance = charge_acceptance_kw(bev_battery, soc)
        headroom_kw = max(0.0, (1.0 - soc) * capacity_kwh) * 3600.0 / numerics.integration_dt_s
        ceiling = min(vehicle_params.regen_max_power_kw, acceptance, headroom_kw)
        assert power <= ceiling + 1e-3
        assert power >= 0.0


def test_no_recovery_is_credited_while_the_pack_is_full(run_route):
    result = run_route(departure_soc=1.0, distance_km=6.0, speed_kmh=60.0)
    for power, soc in zip(result.regen_power_kw, result.soc_trace):
        if soc >= 1.0:
            assert power == 0.0
    assert max(result.soc_trace) <= 1.0


def test_regeneration_never_raises_soc_above_its_departure_value(run_route):
    result = run_route(departure_soc=0.6, distance_km=6.0, speed_kmh=60.0)
    assert max(result.soc_trace) <= result.departure_soc + 1e-9


def test_soc_trace_never_goes_above_full(run_route):
    result = run_route(departure_soc=0.999, distance_km=5.0, speed_kmh=60.0)
    assert max(result.soc_trace) <= 1.0


def test_recovery_fraction_and_path_efficiency_bound_the_recovered_energy(vehicle_params, run_route):
    result = run_route(departure_soc=0.6, distance_km=5.0, speed_kmh=60.0)
    recovered = -result.energy_contributors_kwh["regen_recovered_kwh"]
    traction = result.energy_contributors_kwh["traction_kwh"]
    assert recovered >= 0.0
    # Recovery can never exceed what the drive side put in, times the two factors.
    bound = (
        traction
        * vehicle_params.regen_recovery_fraction
        * vehicle_params.regen_path_efficiency
        / max(vehicle_params.drivetrain_efficiency, 1e-6)
    )
    assert recovered <= bound + 1e-6


def test_regen_power_series_is_never_negative(vehicle_params, run_route):
    result = run_route(departure_soc=0.999, distance_km=5.0, speed_kmh=60.0)
    assert min(result.regen_power_kw) >= 0.0
    assert all(0.0 <= s <= 1.0 for s in result.soc_trace)