"""Vehicle, cabin, battery, and route entities."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from app.schemas.common import (
    CabinMode,
    CabinSource,
    ConfidenceLabel,
    ConstraintStatus,
    EvidenceLabel,
    LuggageClass,
    PowertrainType,
    StrictModel,
    VehicleAvailability,
)


class SeatRegion(StrictModel):
    region_id: str
    occupied: bool


class CabinState(StrictModel):
    vehicle_id: str
    mode: CabinMode
    passenger_count: int = Field(ge=0, le=8)
    luggage_units: int = Field(ge=0, le=12)
    luggage_class_counts: dict[LuggageClass, int] = Field(default_factory=dict)
    luggage_mass_kg: float = Field(ge=0.0, le=400.0)
    seat_occupancy: list[SeatRegion] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    source: CabinSource
    timestamp_utc: datetime
    override_flag: bool = False
    proposed_mode: CabinMode | None = None
    mode_proposal_reason: str | None = None


class BatteryState(StrictModel):
    usable_energy_kwh: float = Field(gt=0.0)
    soc: float = Field(ge=0.0, le=1.0)
    soh: float = Field(ge=0.0, le=1.0)
    soh_uncertainty: float = Field(ge=0.0, le=1.0)
    temperature_c: float = Field(ge=-40.0, le=90.0)
    equivalent_throughput_kwh: float = Field(ge=0.0)
    equivalent_cycles: float = Field(ge=0.0)
    max_charge_power_kw: float = Field(gt=0.0)
    charge_limit_soc: float = Field(ge=0.0, le=1.0)


class BatteryPassport(StrictModel):
    vehicle_id: str
    soh: float
    soh_uncertainty: float
    confidence: ConfidenceLabel
    equivalent_throughput_kwh: float
    equivalent_cycles: float
    temperature_exposure: dict[str, float]
    recent_charge_profile: list[dict]
    marginal_wear_cost_inr_per_kwh: float
    replacement_horizon_km: float | None
    evidence_status: EvidenceLabel
    notes: str


class VehicleState(StrictModel):
    vehicle_id: str
    powertrain: PowertrainType
    availability: VehicleAvailability
    cabin_mode: CabinMode
    passenger_count: int = Field(ge=0, le=8)
    luggage_units: int = Field(ge=0, le=12)
    luggage_mass_kg: float = Field(ge=0.0, le=400.0)
    soc: float = Field(ge=0.0, le=1.0)
    soh: float = Field(ge=0.0, le=1.0)
    battery_temp_c: float
    current_zone_id: str
    current_edge_id: str | None = None
    next_available_time_s: float = Field(ge=0.0)
    odometer_km: float = Field(ge=0.0)
    charge_limit_soc: float = Field(ge=0.0, le=1.0)
    hvac_zones_active: list[str] = Field(default_factory=list)
    maintenance_flags: list[str] = Field(default_factory=list)
    model_version: str
    state_timestamp_utc: datetime
    state_version: int = Field(ge=0)


class Route(StrictModel):
    route_id: str
    origin_zone_id: str
    destination_zone_id: str
    edge_ids: list[str]
    distance_km: float = Field(ge=0.0)
    expected_travel_time_min: float = Field(ge=0.0)
    speed_profile_kmh: list[float] = Field(default_factory=list)
    grade_profile_percent: list[float] = Field(default_factory=list)
    traffic_factor: float = Field(gt=0.0)
    route_energy_kwh: float | None = None
    feasible: bool = True
    feasibility_reasons: list[str] = Field(default_factory=list)


class RouteResult(StrictModel):
    """Full output of one deterministic route-energy simulation."""

    route_id: str
    vehicle_id: str
    distance_km: float
    travel_time_min: float
    departure_soc: float
    arrival_soc: float
    soc_reserve_min: float
    reserve_margin: float
    route_energy_kwh: float
    energy_kwh_per_100km: float
    loaded_mass_kg: float
    passenger_count: int
    luggage_mass_kg: float
    time_s: list[float]
    speed_mps: list[float]
    acceleration_m_s2: list[float]
    wheel_power_kw: list[float]
    traction_battery_power_kw: list[float]
    hvac_power_kw: list[float]
    auxiliary_power_kw: list[float]
    regen_power_kw: list[float]
    cumulative_energy_kwh: list[float]
    soc_trace: list[float]
    energy_contributors_kwh: dict[str, float]
    ambient_temp_c: float
    traffic_factor: float
    mean_grade_percent: float
    warnings: list[str] = Field(default_factory=list)
    constraint_checks: dict[str, ConstraintStatus] = Field(default_factory=dict)
    model_version: str
    config_hash: str
    evidence_status: EvidenceLabel = EvidenceLabel.MODEL_ESTIMATE
    calibrated: bool = False
