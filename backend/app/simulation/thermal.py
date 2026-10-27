"""Occupancy-aware HVAC load and lumped battery thermal response."""

from __future__ import annotations

from app.schemas.common import CabinMode
from app.simulation.model_config import BatteryModelParams, HvacParams


def hvac_power_kw(
    params: HvacParams,
    mode: CabinMode,
    occupied_zones: int,
    ambient_temp_c: float,
    solar_fraction: float = 1.0,
) -> float:
    """Base load + per-zone increments, scaled by mode and ambient temperature.

    Result is clamped into the configured HVAC design band (base load up to
    base*ambient_max_factor) so the demonstrator never reports out-of-band
    values.
    """
    mode_factor = params.mode_factors.get(mode.value.lower(), 1.0)
    zones = max(0, occupied_zones)
    cooling_demand_c = max(0.0, ambient_temp_c - params.ambient_reference_c)
    raw = (
        params.base_load_kw
        + params.per_occupied_zone_kw * zones
        + params.solar_load_kw * max(0.0, min(1.0, solar_fraction))
    ) * mode_factor + params.ambient_slope_kw_per_c * cooling_demand_c
    ceiling = params.base_load_kw * params.ambient_max_factor
    return max(params.base_load_kw, min(raw, ceiling))


def comfort_proxy(
    params: HvacParams,
    mode: CabinMode,
    occupied_zones: int,
    ambient_temp_c: float,
    hvac_kw: float,
) -> float:
    """PMV-inspired display proxy in [-1, +1]; design band is [-0.5, +0.5].

    This is a transparency aid, not a validated comfort measurement.
    """
    offset_c = ambient_temp_c - params.comfort_temp_reference_c
    unconditioned = offset_c * params.comfort_temp_sensitivity_per_c
    relief = hvac_kw / max(params.base_load_kw, 1e-6) * 0.35
    occupancy_penalty = 0.05 * max(0, occupied_zones - 2)
    raw = unconditioned - relief * (0.5 if offset_c > 0 else 0.2) + occupancy_penalty
    mode_factor = params.mode_factors.get(mode.value.lower(), 1.0)
    return max(-1.0, min(1.0, raw / max(mode_factor, 0.1)))


def updated_battery_temp_c(
    params: BatteryModelParams,
    current_temp_c: float,
    ambient_temp_c: float,
    internal_loss_kw: float,
    dt_s: float,
) -> float:
    """First-order lumped thermal response to ambient and internal losses."""
    tau = max(params.thermal_time_constant_s, 1.0)
    # First-order relaxation toward an equilibrium set by ambient temperature
    # plus the adiabatic rise from internal losses.
    equilibrium_c = ambient_temp_c + internal_loss_kw * params.thermal_gain_per_kw_loss
    alpha = 1.0 - pow(2.718281828459045, -max(dt_s, 0.0) / tau)
    return current_temp_c + (equilibrium_c - current_temp_c) * alpha
