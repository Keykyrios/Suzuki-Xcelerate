"""Deterministic longitudinal vehicle dynamics.

Pure functions only: inputs in, values out. Every function validates finiteness
and physical plausibility and raises PhysicsInputError instead of silently
repairing impossible inputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from app.schemas.reason_codes import ReasonCode
from app.simulation.model_config import VehicleModelParams


class PhysicsInputError(ValueError):
    """Invalid physics input; carries the reason code surfaced to operators."""

    def __init__(self, message: str, reason_code: ReasonCode = ReasonCode.ROUTE_ZONE_UNKNOWN):
        super().__init__(message)
        self.reason_code = reason_code


@dataclass(frozen=True)
class MassResult:
    loaded_mass_kg: float
    payload_kg: float


def _require_finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise PhysicsInputError(f"{name} must be finite, got {value}")


def _require_non_negative(name: str, value: float) -> None:
    _require_finite(name, value)
    if value < 0:
        raise PhysicsInputError(f"{name} must be >= 0, got {value}")


def loaded_mass_kg(
    params: VehicleModelParams,
    passenger_count: int,
    luggage_mass_kg: float,
) -> MassResult:
    """curb + passengers + luggage + other payload, with a hard payload check.

    Capacity or payload violations raise instead of silently capping.
    """
    if passenger_count < 0:
        raise PhysicsInputError(
            f"passenger_count must be >= 0, got {passenger_count}",
            ReasonCode.PASSENGER_CAPACITY_EXCEEDED,
        )
    if passenger_count > params.passenger_capacity:
        raise PhysicsInputError(
            f"passenger_count {passenger_count} exceeds capacity {params.passenger_capacity}",
            ReasonCode.PASSENGER_CAPACITY_EXCEEDED,
        )
    _require_non_negative("luggage_mass_kg", luggage_mass_kg)
    passenger_mass = passenger_count * params.passenger_mass_kg
    payload_kg = passenger_mass + luggage_mass_kg + params.other_payload_kg
    if payload_kg > params.max_payload_kg:
        raise PhysicsInputError(
            f"payload {payload_kg:.1f} kg exceeds the {params.max_payload_kg:.1f} kg limit",
            ReasonCode.PAYLOAD_EXCEEDED,
        )
    return MassResult(loaded_mass_kg=params.curb_mass_kg + payload_kg, payload_kg=payload_kg)


def wheel_force_N(
    mass_kg: float,
    acceleration_m_s2: float,
    speed_mps: float,
    grade_rad: float,
    params: VehicleModelParams,
) -> float:
    """F = m*a + m*g*Crr*cos(theta) + m*g*sin(theta) + 0.5*rho*Cd*A*v^2."""
    _require_non_negative("mass_kg", mass_kg)
    _require_finite("acceleration_m_s2", acceleration_m_s2)
    _require_non_negative("speed_mps", speed_mps)
    _require_finite("grade_rad", grade_rad)
    if abs(grade_rad) > math.radians(15.0):
        raise PhysicsInputError(f"grade_rad {grade_rad} outside the supported band")
    inertia = mass_kg * acceleration_m_s2
    rolling = mass_kg * params.gravity_m_s2 * params.rolling_resistance * math.cos(grade_rad)
    climbing = mass_kg * params.gravity_m_s2 * math.sin(grade_rad)
    aero = 0.5 * params.air_density_kg_m3 * params.drag_coefficient * params.frontal_area_m2 * speed_mps**2
    return inertia + rolling + climbing + aero


def wheel_power_W(force_N: float, speed_mps: float) -> float:
    _require_finite("force_N", force_N)
    _require_non_negative("speed_mps", speed_mps)
    return force_N * speed_mps


def traction_battery_power_W(wheel_power_W_value: float, drivetrain_efficiency: float) -> float:
    """Battery-side drive power: positive wheel power divided by eta."""
    _require_finite("wheel_power_W", wheel_power_W_value)
    if not 0.0 < drivetrain_efficiency <= 1.0:
        raise PhysicsInputError(f"drivetrain_efficiency {drivetrain_efficiency} outside (0, 1]")
    return max(wheel_power_W_value, 0.0) / drivetrain_efficiency


def regen_wheel_power_W(wheel_power_W_value: float) -> float:
    """Recoverable negative wheel power (positive value)."""
    _require_finite("wheel_power_W", wheel_power_W_value)
    return max(-wheel_power_W_value, 0.0)


def kinetic_speed_after_accel(
    speed_mps: float, acceleration_m_s2: float, dt_s: float
) -> float:
    _require_non_negative("speed_mps", speed_mps)
    _require_finite("acceleration_m_s2", acceleration_m_s2)
    _require_non_negative("dt_s", dt_s)
    return max(0.0, speed_mps + acceleration_m_s2 * dt_s)
