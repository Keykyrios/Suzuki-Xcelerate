"""Loaded mass, payload accounting, and capacity enforcement."""

from __future__ import annotations

import pytest

from app.schemas.reason_codes import ReasonCode
from app.simulation.vehicle_dynamics import PhysicsInputError, loaded_mass_kg


def test_loaded_mass_sums_every_configured_component(vehicle_params):
    passengers, luggage = 4, 60.0
    result = loaded_mass_kg(vehicle_params, passengers, luggage)
    expected_payload = (
        passengers * vehicle_params.passenger_mass_kg
        + luggage
        + vehicle_params.other_payload_kg
    )
    assert result.payload_kg == pytest.approx(expected_payload, rel=1e-12)
    assert result.loaded_mass_kg == pytest.approx(
        vehicle_params.curb_mass_kg + expected_payload, rel=1e-12
    )


def test_mass_grows_monotonically_with_passengers(vehicle_params):
    masses = [
        loaded_mass_kg(vehicle_params, n, 0.0).loaded_mass_kg for n in range(0, 5)
    ]
    assert masses == sorted(masses)
    assert masses[1] - masses[0] == pytest.approx(vehicle_params.passenger_mass_kg)


def test_mass_grows_with_luggage(vehicle_params):
    light = loaded_mass_kg(vehicle_params, 2, 5.0)
    heavy = loaded_mass_kg(vehicle_params, 2, 50.0)
    assert heavy.loaded_mass_kg - light.loaded_mass_kg == pytest.approx(45.0)


def test_passenger_count_above_capacity_is_rejected_with_reason_code(vehicle_params):
    with pytest.raises(PhysicsInputError) as excinfo:
        loaded_mass_kg(vehicle_params, vehicle_params.passenger_capacity + 1, 0.0)
    assert excinfo.value.reason_code == ReasonCode.PASSENGER_CAPACITY_EXCEEDED


def test_payload_above_limit_is_rejected_instead_of_capped(vehicle_params):
    with pytest.raises(PhysicsInputError) as excinfo:
        loaded_mass_kg(vehicle_params, 0, vehicle_params.max_payload_kg * 2.0)
    assert excinfo.value.reason_code == ReasonCode.PAYLOAD_EXCEEDED


def test_negative_passengers_or_luggage_are_rejected(vehicle_params):
    with pytest.raises(PhysicsInputError):
        loaded_mass_kg(vehicle_params, -1, 0.0)
    with pytest.raises(PhysicsInputError):
        loaded_mass_kg(vehicle_params, 1, -5.0)


def test_non_finite_payload_inputs_are_rejected(vehicle_params):
    with pytest.raises(PhysicsInputError):
        loaded_mass_kg(vehicle_params, 2, float("nan"))
    with pytest.raises(PhysicsInputError):
        loaded_mass_kg(vehicle_params, 2, float("inf"))


def test_max_payload_is_configured_not_hardcoded(vehicle_params, cfg):
    assert vehicle_params.max_payload_kg == pytest.approx(
        float(cfg.v("package.max_payload_kg"))
    )
    assert vehicle_params.curb_mass_kg == float(cfg.v("package.curb_mass_kg"))