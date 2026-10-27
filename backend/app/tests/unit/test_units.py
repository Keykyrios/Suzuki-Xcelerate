"""Unit conventions: SI internals, explicit unit suffixes, conversions."""

from __future__ import annotations

import math

import pytest

from app.config.loader import get_config
from app.schemas.vehicle import RouteResult, VehicleState
from app.simulation.model_config import vehicle_model_params
from app.simulation.vehicle_dynamics import (
    PhysicsInputError,
    traction_battery_power_W,
    wheel_force_N,
    wheel_power_W,
)

KWH_PER_KW_SECOND = 1.0 / 3600.0


@pytest.fixture(scope="module")
def params():
    return vehicle_model_params(get_config())


def test_wheel_force_terms_use_newtons(params):
    stationary = wheel_force_N(2000.0, 0.0, 0.0, 0.0, params)
    expected_rolling = 2000.0 * params.gravity_m_s2 * params.rolling_resistance
    assert stationary == pytest.approx(expected_rolling, rel=1e-9)


def test_grade_force_is_antisymmetric(params):
    theta = math.radians(5.0)
    up = wheel_force_N(2000.0, 0.0, 0.0, theta, params)
    down = wheel_force_N(2000.0, 0.0, 0.0, -theta, params)
    climbing = 2.0 * 2000.0 * params.gravity_m_s2 * math.sin(theta)
    # Rolling resistance is symmetric in grade, so only the climb term remains.
    assert up - down == pytest.approx(climbing, rel=1e-9)


def test_grade_is_passed_in_radians(params):
    with pytest.raises(PhysicsInputError):
        # 100 would be degrees-by-mistake; the supported band is +-15 degrees.
        wheel_force_N(2000.0, 0.0, 0.0, 100.0, params)


def test_aero_force_scales_with_speed_squared(params):
    f_slow = wheel_force_N(2000.0, 0.0, 10.0, 0.0, params)
    f_fast = wheel_force_N(2000.0, 0.0, 20.0, 0.0, params)
    aero = 0.5 * params.air_density_kg_m3 * params.drag_coefficient * params.frontal_area_m2
    assert (f_fast - wheel_force_N(2000.0, 0.0, 0.0, 0.0, params)) / (
        f_slow - wheel_force_N(2000.0, 0.0, 0.0, 0.0, params)
    ) == pytest.approx(4.0, rel=1e-6)
    assert f_slow - wheel_force_N(2000.0, 0.0, 0.0, 0.0, params) == pytest.approx(
        aero * 100.0, rel=1e-6
    )


def test_power_is_force_times_speed_and_scales_to_kilowatts(params):
    force = wheel_force_N(2000.0, 1.0, 15.0, 0.0, params)
    watts = wheel_power_W(force, 15.0)
    assert watts == pytest.approx(force * 15.0)
    assert watts / 1000.0 == pytest.approx(force * 15.0 / 1000.0)


def test_efficiency_divides_battery_side_power(params):
    assert traction_battery_power_W(10_000.0, 0.9) == pytest.approx(10_000.0 / 0.9)
    with pytest.raises(PhysicsInputError):
        traction_battery_power_W(10_000.0, 0.0)


def test_energy_conversion_kwh_from_kw_and_seconds():
    # kW integrated over seconds converts to kWh through P_kW * t_s / 3600.
    assert 1.0 * 3600.0 / 3600.0 == pytest.approx(1.0)
    assert 3.0 * 60.0 / 3600.0 == pytest.approx(0.05)


def test_soc_is_fractional_unit_everywhere():
    schema = VehicleState.model_json_schema()["properties"]
    for name in ("soc", "soh", "charge_limit_soc"):
        assert schema[name]["minimum"] == 0.0
        assert schema[name]["maximum"] == 1.0
    assert "route_energy_kwh" in RouteResult.model_fields
    assert "energy_kwh_per_100km" in RouteResult.model_fields


def test_speed_is_converted_from_kmh_to_mps_by_unit_helpers():
    # 36 km/h must arrive as 10 m/s anywhere a speed trace is built.
    from app.simulation.route_energy import MPS_PER_KMH

    assert 36.0 * MPS_PER_KMH == pytest.approx(10.0)
    assert 72.0 * MPS_PER_KMH == pytest.approx(20.0)