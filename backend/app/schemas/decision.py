"""Decision entities produced by the optimizer and baseline dispatcher."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field

from app.schemas.common import (
    CabinMode,
    ConstraintStatus,
    EvidenceLabel,
    PowertrainType,
    SolverStatus,
    StrictModel,
)


class Assignment(StrictModel):
    request_id: str
    vehicle_id: str
    pickup_zone_id: str
    dropoff_zone_id: str
    pickup_eta_s: float
    pickup_deadline_s: float
    deadhead_km: float
    trip_distance_km: float
    route_energy_kwh: float
    departure_soc: float
    arrival_soc: float
    expected_fare_inr: float
    contribution_inr: float
    selected_charger_id: str | None = None


class ChargingCostBreakdown(StrictModel):
    electricity_inr: float
    waiting_inr: float
    detour_inr: float
    lost_productive_inr: float
    degradation_inr: float

    @property
    def total_inr(self) -> float:
        return (
            self.electricity_inr
            + self.waiting_inr
            + self.detour_inr
            + self.lost_productive_inr
            + self.degradation_inr
        )


class ChargingAction(StrictModel):
    charger_id: str
    vehicle_id: str
    energy_kwh: float
    effective_power_kw: float
    charge_duration_min: float
    queue_wait_min: float
    detour_km: float
    plug_in_time_s: float
    target_soc: float
    arrival_soc: float
    cost: ChargingCostBreakdown
    reason_codes: list[str] = Field(default_factory=list)


class RepositioningAction(StrictModel):
    vehicle_id: str
    from_zone_id: str
    to_zone_id: str
    deadhead_km: float
    deadhead_energy_kwh: float
    deadhead_time_min: float
    forecast_confidence: float
    expected_benefit_inr: float
    cost_inr: float
    reason_codes: list[str] = Field(default_factory=list)


class PowertrainComparison(StrictModel):
    mission_id: str
    recommended: PowertrainType
    alternative: PowertrainType
    cost_nexus_e_inr: float
    cost_nexus_x_inr: float
    cost_difference_inr: float
    switching_inputs: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)
    explanation: str = ""


class RejectedAlternative(StrictModel):
    kind: Literal["assignment", "charger", "repositioning", "powertrain"]
    subject_id: str
    reason_codes: list[str] = Field(default_factory=list)
    cost_inr: float | None = None


class RiskMetrics(StrictModel):
    enabled: bool
    alpha: float
    lambda_weight: float
    loss_definition: str
    expected_profit_inr: float
    cvar_loss_inr: float
    scenario_count: int
    tail_probability: float


class SolverReport(StrictModel):
    name: str
    version: str
    status: SolverStatus
    runtime_ms: float = Field(ge=0.0)
    fallback_used: bool = False
    message: str = ""


class FleetMetrics(StrictModel):
    """Paired comparison metrics between baseline and optimized plans."""

    completed_requests: int = Field(ge=0)
    unserved_requests: int = Field(ge=0)
    deadhead_km: float = Field(ge=0.0)
    productive_km: float = Field(ge=0.0)
    total_km: float = Field(ge=0.0)
    productive_km_ratio: float = Field(ge=0.0, le=1.0)
    total_energy_kwh: float = Field(ge=0.0)
    energy_per_vehicle_km_kwh: float = Field(ge=0.0)
    charging_wait_min: float = Field(ge=0.0)
    charging_detour_km: float = Field(ge=0.0)
    missed_productive_min: float = Field(ge=0.0)
    min_soc: float = Field(ge=0.0, le=1.0)
    reserve_violations: int = Field(ge=0)
    gross_revenue_inr: float
    energy_cost_inr: float
    wear_cost_inr: float
    charging_opportunity_cost_inr: float
    missed_service_cost_inr: float
    contribution_inr: float


class DecisionPayload(StrictModel):
    """Headline operator-facing summary of the selected plan."""

    cabin_mode: CabinMode
    powertrain_recommendation: PowertrainType
    route_energy_kwh: float
    energy_kwh_per_100km: float
    departure_soc_pct: float
    arrival_soc_pct: float
    reserve_soc_pct: float
    reserve_margin_pct: float
    selected_charger_id: str | None
    estimated_charge_wait_min: float | None
    expected_contribution_inr: float
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]


class Decision(StrictModel):
    run_id: str
    scenario_id: str
    status: SolverStatus
    created_utc: datetime
    decision: DecisionPayload
    assignments: list[Assignment]
    charging_actions: list[ChargingAction]
    repositioning_actions: list[RepositioningAction]
    powertrain: PowertrainComparison | None
    constraints: dict[str, ConstraintStatus]
    reason_codes: list[str]
    rejected_alternatives: list[RejectedAlternative]
    risk: RiskMetrics | None
    solver: SolverReport
    baseline_metrics: FleetMetrics | None
    optimized_metrics: FleetMetrics | None
    warnings: list[str]
    evidence_status: EvidenceLabel
    stale: bool = False
    recheck_passed: bool | None = None
