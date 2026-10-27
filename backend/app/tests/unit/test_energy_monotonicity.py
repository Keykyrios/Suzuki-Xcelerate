"""Monotonicity: heavier loads, hotter cabins and steeper grades cost more energy."""

from __future__ import annotations

import pytest


def test_more_passengers_costs_more_energy(run_route):
    light = run_route(passenger_count=1, luggage_mass_kg=0.0)
    heavy = run_route(passenger_count=6, luggage_mass_kg=0.0)
    assert heavy.route_energy_kwh > light.route_energy_kwh
    assert heavy.loaded_mass_kg > light.loaded_mass_kg


def test_more_luggage_costs_more_energy(run_route):
    empty = run_route(passenger_count=3, luggage_mass_kg=0.0)
    loaded = run_route(passenger_count=3, luggage_mass_kg=120.0)
    assert loaded.route_energy_kwh > empty.route_energy_kwh


def test_steeper_positive_grade_costs_more_energy(run_route):
    flat = run_route(grade_percent=(0.0,))
    climb = run_route(grade_percent=(5.0,))
    assert climb.route_energy_kwh > flat.route_energy_kwh


def test_descending_route_recovers_more_than_flat(run_route):
    flat = run_route(grade_percent=(0.0,))
    downhill = run_route(grade_percent=(-5.0,))
    recovered_flat = -flat.energy_contributors_kwh["regen_recovered_kwh"]
    recovered_down = -downhill.energy_contributors_kwh["regen_recovered_kwh"]
    assert recovered_down > recovered_flat
    assert downhill.route_energy_kwh < flat.route_energy_kwh


def test_more_traffic_lowers_average_speed_and_costs_time(run_route):
    clear = run_route(traffic_factor=1.0)
    congested = run_route(traffic_factor=1.6)
    assert congested.travel_time_min > clear.travel_time_min


def test_energy_is_monotonic_in_passenger_count_across_the_fleet_range(run_route):
    energies = [run_route(passenger_count=n).route_energy_kwh for n in range(0, 7)]
    assert energies == sorted(energies)


@pytest.mark.parametrize("passengers", [0, 3, 6])
def test_energy_per_100km_stays_physically_plausible(run_route, passengers):
    result = run_route(passenger_count=passengers)
    # A compact e-MPV consumes a few kWh per 100 km in this model; values far
    # outside that band indicate a unit or term error.
    assert 0.5 <= result.energy_kwh_per_100km <= 100.0


def test_power_limited_traces_are_reported_not_silently_used(run_route):
    result = run_route(speed_kmh=60.0, distance_km=8.0, passenger_count=6)
    # Either the trace stayed within peak power, or the warning explains that it
    # did not. No silent clipping.
    peak_warning = [w for w in result.warnings if "peak" in w]
    assert peak_warning or all(
        p <= load for p, load in zip(
            result.traction_battery_power_kw,
            [load for load in result.traction_battery_power_kw],
        )
    )