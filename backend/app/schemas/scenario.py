"""Scenario entities: graph, demand, requests, and the canonical scenario."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import Field, field_validator

from app.schemas.charger import ChargerState
from app.schemas.common import (
    CabinMode,
    CabinSource,
    DayType,
    EvidenceLabel,
    LuggageClass,
    PowertrainType,
    RequestStatus,
    StrictModel,
)
from app.schemas.vehicle import VehicleState


class ZoneKind(str):
    URBAN = "urban"
    AIRPORT = "airport"
    CORRIDOR = "corridor"


class GraphZone(StrictModel):
    zone_id: str
    name: str
    x_m: float
    y_m: float
    kind: str = "urban"
    is_intercity_terminal: bool = False


class RoadEdge(StrictModel):
    edge_id: str
    from_node: str
    to_node: str
    distance_km: float = Field(gt=0.0)
    free_flow_speed_kmh: float = Field(gt=0.0)
    grade_profile_percent: list[float]
    traffic_sensitivity: float = Field(ge=0.0, le=1.5)
    toll_inr: float = Field(ge=0.0, default=0.0)
    bidirectional: bool = True
    flags: list[str] = Field(default_factory=list)

    @field_validator("grade_profile_percent")
    @classmethod
    def _grades_in_range(cls, v: list[float]) -> list[float]:
        if not v:
            raise ValueError("grade profile must contain at least one segment")
        for g in v:
            if not -12.0 <= g <= 12.0:
                raise ValueError(f"grade segment {g}% outside the supported +/-12% band")
        return v

    @property
    def mean_grade_percent(self) -> float:
        return sum(self.grade_profile_percent) / len(self.grade_profile_percent)

    @property
    def base_travel_time_min(self) -> float:
        return self.distance_km / (self.free_flow_speed_kmh / 60.0)


class CityGraph(StrictModel):
    graph_id: str
    version: str
    zones: list[GraphZone]
    edges: list[RoadEdge]


class DemandRecord(StrictModel):
    record_id: str
    hour_of_day: int = Field(ge=0, le=23)
    day_type: DayType
    service_class: CabinMode
    origin_zone_id: str
    destination_zone_id: str
    request_count: int = Field(ge=0)
    source_status: EvidenceLabel


class DemandContext(StrictModel):
    daily_requests_mean: int
    dispersion: float
    demand_multiplier: float = Field(gt=0.0)
    day_type: DayType
    hour_of_day: int = Field(ge=0, le=23)
    provenance: EvidenceLabel = EvidenceLabel.SYNTHETIC


class ODFlow(StrictModel):
    origin_zone_id: str
    destination_zone_id: str
    trips_per_day: float = Field(ge=0.0)
    provenance: EvidenceLabel


class TripRequest(StrictModel):
    request_id: str
    pickup_zone_id: str
    dropoff_zone_id: str
    request_time_s: float = Field(ge=0.0)
    pickup_window_start_s: float
    pickup_window_end_s: float
    passenger_count: int = Field(ge=1, le=8)
    luggage_units: int = Field(ge=0, le=12)
    luggage_classes: dict[LuggageClass, int] = Field(default_factory=dict)
    luggage_mass_kg: float = Field(ge=0.0, le=400.0)
    service_class: CabinMode
    expected_fare_inr: float = Field(ge=0.0)
    cancellation_penalty_inr: float = Field(ge=0.0)
    status: RequestStatus = RequestStatus.PENDING
    assigned_vehicle_id: str | None = None


class FaultInjection(StrictModel):
    fault_id: str
    name: str
    target: str
    parameter: str
    value: Any
    target_id: str | None = None


class EnvironmentControls(StrictModel):
    ambient_temp_c: float = Field(ge=-10.0, le=55.0)
    traffic_factor: float = Field(gt=0.0, le=3.0)
    road_grade_percent: float | None = Field(default=None, ge=-12.0, le=12.0)
    demand_multiplier: float = Field(default=1.0, gt=0.0, le=5.0)


class CabinSpec(StrictModel):
    mode: CabinMode
    passenger_count: int = Field(ge=0, le=8)
    luggage_count: int = Field(ge=0, le=12)
    luggage_mass_kg: float = Field(ge=0.0, le=400.0)
    luggage_classes: dict[LuggageClass, int] = Field(default_factory=dict)
    source: CabinSource = CabinSource.SCENARIO
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class RouteSpec(StrictModel):
    origin_zone: str
    destination_zone: str


class ScenarioSpec(StrictModel):
    """Validated scenario request as accepted by the API."""

    scenario_id: str = Field(min_length=1, max_length=120)
    seed: int = 0
    vehicle_id: str | None = None
    cabin: CabinSpec | None = None
    environment: EnvironmentControls | None = None
    route: RouteSpec | None = None
    fault_injection: dict[str, Any] | None = None
    requests_override: list[TripRequest] | None = None


class Scenario(StrictModel):
    """Canonical, serializable, replayable world state for one run."""

    scenario_id: str
    seed: int
    graph: CityGraph
    vehicles: list[VehicleState]
    chargers: list[ChargerState]
    requests: list[TripRequest]
    environment: EnvironmentControls
    demand_context: DemandContext
    od_flows: list[ODFlow] = Field(default_factory=list)
    faults: list[FaultInjection] = Field(default_factory=list)
    tariffs_inr_per_kwh: dict[str, float] = Field(default_factory=dict)
    created_utc: datetime
    scenario_version: str
    provenance: EvidenceLabel = EvidenceLabel.SYNTHETIC

    def vehicle(self, vehicle_id: str) -> VehicleState:
        for v in self.vehicles:
            if v.vehicle_id == vehicle_id:
                return v
        raise KeyError(f"unknown vehicle: {vehicle_id}")

    def request(self, request_id: str) -> TripRequest:
        for r in self.requests:
            if r.request_id == request_id:
                return r
        raise KeyError(f"unknown request: {request_id}")

    def charger(self, charger_id: str):
        for c in self.chargers:
            if c.charger_id == charger_id:
                return c
        raise KeyError(f"unknown charger: {charger_id}")
