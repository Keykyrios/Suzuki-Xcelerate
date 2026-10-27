"""Every longitudinal-dynamics equation term and input guard, verified exactly."""

from __future__ import annotations

import math

import pytest

from app.schemas.reason_codes import ReasonCode
from app.simulation.vehicle_dynamics import (
    PhysicsInputError,
    kinetic_speed_after_accel,
    regen_wheel_power_W,
    traction_battery_power_W,
    wheel_force_N,
    wheel_power_W,
)


def test_each_force_term_matches_its_definition(vehicle_params):
    p = vehicle_params
    mass, accel, speed, grade = 2500.0, 0.8, 20.0, math.radians(3.0)
    total = wheel_force_N(mass, accel, speed, grade, p)

    inertia = mass * accel
    rolling = mass * p.gravity_m_s2 * p.rolling_resistance * math.cos(grade)
    climbing = mass * p.gravity_m_s2 * math.sin(grade)
    aero = 0.5 * p.air_density_kg_m3 * p.drag_coefficient * p.frontal_area_m2 * speed**2
    assert total == pytest.approx(inertia + rolling + climbing + aero, rel=1e-12)


def test_inertia_term_is_linear_in_mass_and_acceleration(vehicle_params):
    base = wheel_force_N(1000.0, 1.0, 0.0, 0.0, vehicle_params)
    double_mass = wheel_force_N(2000.0, 1.0, 0.0, 0.0, vehicle_params)
    zero_accel = wheel_force_N(1000.0, 0.0, 0.0, 0.0, vehicle_params)
    term = 1000.0 * 1.0
    assert (base - zero_accel) == pytest.approx(term, rel=1e-9)
    assert (double_mass - 2 * zero_accel) == pytest.approx(2 * term, rel=1e-9)


def test_stationary_vehicle_on_level_road_has_only_rolling_resistance(vehicle_params):
    force = wheel_force_N(2000.0, 0.0, 0.0, 0.0, vehicle_params)
    expected = 2000.0 * vehicle_params.gravity_m_s2 * vehicle_params.rolling_resistance
    assert force == pytest.approx(expected, rel=1e-12)


def test_rolling_term_vanishes_as_the_grade_reference_frame_changes_sign_consistently(vehicle_params):
    grade = math.radians(7.0)
    up = wheel_force_N(2000.0, 0.0, 0.0, grade, vehicle_params)
    down = wheel_force_N(2000.0, 0.0, 0.0, -grade, vehicle_params)
    # cos(+g) == cos(-g): only the sin term can differ.
    assert (up - down) == pytest.approx(
        4000.0 * vehicle_params.gravity_m_s2 * math.sin(grade), rel=1e-9
    )


def test_wheel_power_is_force_times_speed(vehicle_params):
    force = wheel_force_N(2300.0, 0.5, 12.0, 0.0, vehicle_params)
    assert wheel_power_W(force, 12.0) == pytest.approx(force * 12.0)
    assert wheel_power_W(force, 0.0) == 0.0


def test_drive_power_is_wheel_power_divided_by_efficiency(vehicle_params):
    eta = vehicle_params.drivetrain_efficiency
    assert traction_battery_power_W(25_000.0, eta) == pytest.approx(25_000.0 / eta)
    # Negative wheel power is not drawn from the battery on the drive path.
    assert traction_battery_power_W(-5_000.0, eta) == 0.0


def test_regen_reports_only_the_negative_wheel_power():
    assert regen_wheel_power_W(-8_000.0) == pytest.approx(8_000.0)
    assert regen_wheel_power_W(3_000.0) == 0.0


def test_speed_integration_of_kinematics_matches_constant_accel():
    speed = kinetic_speed_after_accel(5.0, 1.5, 4.0)
    assert speed == pytest.approx(11.0)
    assert kinetic_speed_after_accel(5.0, -10.0, 1.0) == 0.0  # never negative


def test_invalid_inputs_raise_instead_of_being_repaired(vehicle_params):
    with pytest.raises(PhysicsInputError):
        wheel_force_N(-100.0, 0.0, 0.0, 0.0, vehicle_params)
    with pytest.raises(PhysicsInputError):
        wheel_force_N(2000.0, float("nan"), 0.0, 0.0, vehicle_params)
    with pytest.raises(PhysicsInputError):
        wheel_force_N(2000.0, 0.0, float("inf"), 0.0, vehicle_params)
    with pytest.raises(PhysicsInputError):
        wheel_force_N(2000.0, 0.0, 0.0, math.radians(45.0), vehicle_params)
    with pytest.raises(PhysicsInputError):
        traction_battery_power_W(1000.0, 1.5)
    with pytest.raises(PhysicsInputError):
        wheel_power_W(float("nan"), 1.0)


def test_physics_error_carries_a_reason_code(vehicle_params):
    with pytest.raises(PhysicsInputError) as excinfo:
        wheel_force_N(2000.0, 0.0, 0.0, math.radians(45.0), vehicle_params)
    assert isinstance(excinfo.value.reason_code, ReasonCode)