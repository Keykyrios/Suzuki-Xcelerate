"""Typed views over the configuration registry for the physics modules.

Physics functions are pure: every tunable arrives through one of the frozen
parameter dataclasses built here, so no model code reads YAML or hard-codes
constants.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.config.loader import ConfigBundle, get_config
from app.schemas.common import CabinMode, PowertrainType


@dataclass(frozen=True)
class VehicleModelParams:
    passenger_capacity: int
    curb_mass_kg: float
    passenger_mass_kg: float
    other_payload_kg: float
    luggage_mass_per_unit_kg: float
    max_payload_kg: float
    drag_coefficient: float
    frontal_area_m2: float
    rolling_resistance: float
    air_density_kg_m3: float
    gravity_m_s2: float
    drivetrain_efficiency: float
    peak_traction_power_kw: float
    max_acceleration_m_s2: float
    max_deceleration_m_s2: float
    regen_recovery_fraction: float
    regen_max_power_kw: float
    regen_path_efficiency: float
    regen_min_speed_m_s: float
    auxiliary_load_kw: float


@dataclass(frozen=True)
class HvacParams:
    base_load_kw: float
    per_occupied_zone_kw: float
    ambient_reference_c: float
    ambient_slope_kw_per_c: float
    ambient_max_factor: float
    solar_load_kw: float
    mode_factors: dict[str, float]
    comfort_band: float
    comfort_temp_reference_c: float
    comfort_temp_sensitivity_per_c: float


@dataclass(frozen=True)
class BatteryModelParams:
    usable_energy_kwh: float
    max_charge_power_dc_kw: float
    max_charge_power_ac_kw: float
    taper_start_soc: float
    taper_min_fraction: float
    charge_efficiency: float
    initial_soh: float
    soh_floor: float
    soh_uncertainty_base: float
    soh_uncertainty_per_1000_cycles: float
    soh_confidence_high_max: float
    soh_confidence_medium_max: float
    thermal_time_constant_s: float
    thermal_resistance_kw_per_c: float
    thermal_gain_per_kw_loss: float
    optimal_temp_c: float
    throughput_wear_per_kwh: float
    c_rate_wear_factor: float
    c_rate_reference: float
    temp_wear_factor: float
    dod_wear_factor: float
    dod_reference: float
    wear_acceleration_factor: float
    equivalent_cycle_capacity_kwh: float


@dataclass(frozen=True)
class NumericsParams:
    integration_dt_s: float


@dataclass(frozen=True)
class ModeLayout:
    max_passengers: int
    max_luggage_units: int
    luggage_class_allowed: tuple[str, ...]


def mode_layouts(cfg: ConfigBundle | None = None) -> dict[CabinMode, ModeLayout]:
    """Passenger/luggage limits of each configured cabin mode."""
    if cfg is None:
        cfg = get_config()
    out: dict[CabinMode, ModeLayout] = {}
    for mode in CabinMode:
        record = cfg.vehicle.parameter(f"cabin_modes.{mode.value.lower()}")
        out[mode] = ModeLayout(
            max_passengers=int(record.max_passengers),
            max_luggage_units=int(record.max_luggage_units),
            luggage_class_allowed=tuple(record.luggage_class_allowed),
        )
    return out


def vehicle_model_params(cfg: ConfigBundle | None = None) -> VehicleModelParams:
    c = cfg or get_config()
    return VehicleModelParams(
        passenger_capacity=int(c.v("package.passenger_capacity")),
        curb_mass_kg=float(c.v("package.curb_mass_kg")),
        passenger_mass_kg=float(c.v("package.passenger_mass_kg")),
        other_payload_kg=float(c.v("package.other_payload_kg")),
        luggage_mass_per_unit_kg=float(c.v("package.luggage_mass_per_unit_kg")),
        max_payload_kg=float(c.v("package.max_payload_kg")),
        drag_coefficient=float(c.v("aerodynamics.drag_coefficient")),
        frontal_area_m2=float(c.v("aerodynamics.frontal_area_m2")),
        rolling_resistance=float(c.v("resistance.rolling_resistance_coefficient")),
        air_density_kg_m3=float(c.v("resistance.air_density_kg_m3")),
        gravity_m_s2=float(c.v("resistance.gravity_m_s2")),
        drivetrain_efficiency=float(c.v("drivetrain.traction_efficiency")),
        peak_traction_power_kw=float(c.v("drivetrain.peak_traction_power_kw")),
        max_acceleration_m_s2=float(c.v("drivetrain.max_acceleration_m_s2")),
        max_deceleration_m_s2=float(c.v("drivetrain.max_deceleration_m_s2")),
        regen_recovery_fraction=float(c.v("regeneration.recovery_fraction")),
        regen_max_power_kw=float(c.v("regeneration.max_regen_power_kw")),
        regen_path_efficiency=float(c.v("regeneration.path_efficiency")),
        regen_min_speed_m_s=float(c.v("regeneration.min_speed_m_s")),
        auxiliary_load_kw=float(c.v("auxiliary.auxiliary_load_kw")),
    )


def hvac_params(cfg: ConfigBundle | None = None) -> HvacParams:
    c = cfg or get_config()
    raw = c.v("hvac.mode_factor")
    return HvacParams(
        base_load_kw=float(c.v("hvac.base_load_kw")),
        per_occupied_zone_kw=float(c.v("hvac.per_occupied_zone_kw")),
        ambient_reference_c=float(c.v("hvac.ambient_reference_c")),
        ambient_slope_kw_per_c=float(c.v("hvac.ambient_slope_kw_per_c")),
        ambient_max_factor=float(c.v("hvac.ambient_max_factor")),
        solar_load_kw=float(c.v("hvac.solar_load_kw")),
        mode_factors={str(k).lower(): float(v) for k, v in dict(raw).items()},
        comfort_band=float(c.v("hvac.comfort_proxy_band")),
        comfort_temp_reference_c=float(c.v("hvac.comfort_temp_reference_c")),
        comfort_temp_sensitivity_per_c=float(c.v("hvac.comfort_temp_sensitivity_per_c")),
    )


def battery_params(
    powertrain: PowertrainType, cfg: ConfigBundle | None = None
) -> BatteryModelParams:
    c = cfg or get_config()
    key = "bev_usable_energy_kwh" if powertrain == PowertrainType.NEXUS_E else "rex_usable_energy_kwh"
    return BatteryModelParams(
        usable_energy_kwh=float(c.v(f"battery.{key}")),
        max_charge_power_dc_kw=float(c.v("battery.max_charge_power_dc_kw")),
        max_charge_power_ac_kw=float(c.v("battery.max_charge_power_ac_kw")),
        taper_start_soc=float(c.v("battery.taper_start_soc")),
        taper_min_fraction=float(c.v("battery.taper_min_fraction")),
        charge_efficiency=float(c.v("battery.charge_efficiency")),
        initial_soh=float(c.v("battery.initial_soh")),
        soh_floor=float(c.v("battery.soh_floor")),
        soh_uncertainty_base=float(c.v("battery.soh_uncertainty_base")),
        soh_uncertainty_per_1000_cycles=float(c.v("battery.soh_uncertainty_per_1000_cycles")),
        soh_confidence_high_max=float(c.v("battery.soh_confidence_high_max")),
        soh_confidence_medium_max=float(c.v("battery.soh_confidence_medium_max")),
        thermal_time_constant_s=float(c.v("battery.thermal_time_constant_s")),
        thermal_resistance_kw_per_c=float(c.v("battery.thermal_resistance_kw_per_c")),
        thermal_gain_per_kw_loss=float(c.v("battery.thermal_gain_per_kw_loss")),
        optimal_temp_c=float(c.v("battery.optimal_temp_c")),
        throughput_wear_per_kwh=float(c.v("degradation.throughput_wear_per_kwh")),
        c_rate_wear_factor=float(c.v("degradation.c_rate_wear_factor")),
        c_rate_reference=float(c.v("degradation.c_rate_reference")),
        temp_wear_factor=float(c.v("degradation.temp_wear_factor")),
        dod_wear_factor=float(c.v("degradation.dod_wear_factor")),
        dod_reference=float(c.v("degradation.dod_reference")),
        wear_acceleration_factor=float(c.v("degradation.wear_acceleration_factor")),
        equivalent_cycle_capacity_kwh=float(c.v("degradation.equivalent_cycle_capacity_kwh")),
    )


def numerics_params(cfg: ConfigBundle | None = None) -> NumericsParams:
    c = cfg or get_config()
    return NumericsParams(integration_dt_s=float(c.v("numerics.integration_dt_s")))


MODEL_VERSION = "nexus7-physics-1.0.0"
