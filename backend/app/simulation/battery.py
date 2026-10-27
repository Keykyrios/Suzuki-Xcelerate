"""Battery state, degradation proxy, and passport.

The degradation model is a transparent semi-empirical throughput proxy: wear
scales with C-rate, temperature exposure, and depth of discharge. It is a cost
signal for operations, not a validated lifetime prediction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from app.schemas.common import ConfidenceLabel, EvidenceLabel
from app.schemas.vehicle import BatteryPassport, BatteryState
from app.simulation.model_config import BatteryModelParams


def charge_acceptance_kw(params: BatteryModelParams, soc: float, connector: str = "DC") -> float:
    """Maximum battery charge power at a given SOC (taper model)."""
    limit = (
        params.max_charge_power_dc_kw
        if connector.upper().startswith("DC")
        else params.max_charge_power_ac_kw
    )
    if soc <= params.taper_start_soc:
        fraction = 1.0
    elif soc >= 1.0:
        fraction = params.taper_min_fraction
    else:
        span = 1.0 - params.taper_start_soc
        fraction = 1.0 - (1.0 - params.taper_min_fraction) * (soc - params.taper_start_soc) / span
    return limit * fraction


def wear_per_kwh(
    params: BatteryModelParams,
    temperature_c: float,
    c_rate: float,
    depth_of_discharge: float,
    soh: float,
) -> float:
    """Semi-empirical wear multiplier; monotonic in every stress axis."""
    base = params.throughput_wear_per_kwh
    crate_term = 1.0 + params.c_rate_wear_factor * max(0.0, c_rate - params.c_rate_reference)
    temp_term = 1.0 + params.temp_wear_factor * max(0.0, temperature_c - params.optimal_temp_c)
    dod_term = 1.0 + params.dod_wear_factor * max(0.0, depth_of_discharge - params.dod_reference)
    effective_soh = max(soh, params.soh_floor)
    aging_term = 1.0 + params.wear_acceleration_factor * max(0.0, 1.0 / effective_soh - 1.0)
    return base * crate_term * temp_term * dod_term * aging_term


def soh_uncertainty(params: BatteryModelParams, equivalent_cycles: float) -> float:
    """Estimate spread grows with accumulated cycling."""
    return params.soh_uncertainty_base + params.soh_uncertainty_per_1000_cycles * (
        equivalent_cycles / 1000.0
    )


def confidence_label(params: BatteryModelParams, uncertainty: float) -> ConfidenceLabel:
    if uncertainty <= params.soh_confidence_high_max:
        return ConfidenceLabel.HIGH
    if uncertainty <= params.soh_confidence_medium_max:
        return ConfidenceLabel.MEDIUM
    return ConfidenceLabel.LOW


@dataclass
class BatteryTracker:
    """Mutable per-vehicle battery history; deterministic given its inputs."""

    vehicle_id: str
    params: BatteryModelParams
    soc: float = 1.0
    soh: float = 1.0
    temperature_c: float = 25.0
    throughput_kwh: float = 0.0
    equivalent_cycles: float = 0.0
    damage_index: float = 0.0
    crate_ewma: float = 0.0
    max_depth_of_discharge: float = 0.0
    temperature_seconds: dict[str, float] = field(
        default_factory=lambda: {
            "below_10C": 0.0,
            "10_to_25C": 0.0,
            "25_to_35C": 0.0,
            "35_to_45C": 0.0,
            "above_45C": 0.0,
        }
    )
    charge_profile: list[dict] = field(default_factory=list)
    _ewma_alpha: float = 0.1

    def __post_init__(self) -> None:
        self.soc = _clamp01(self.soc)
        self.soh = min(1.0, max(self.params.soh_floor, self.soh))

    def copy(self) -> "BatteryTracker":
        import copy as _copy

        return _copy.deepcopy(self)

    # -- derived ----------------------------------------------------------
    def effective_capacity_kwh(self) -> float:
        return self.params.usable_energy_kwh * self.soh

    def current_uncertainty(self) -> float:
        return soh_uncertainty(self.params, self.equivalent_cycles)

    def current_confidence(self) -> ConfidenceLabel:
        return confidence_label(self.params, self.current_uncertainty())

    def marginal_wear_cost_inr_per_kwh(self, replacement_cost_inr_per_kwh: float) -> float:
        """INR of expected pack-value loss per kWh of throughput right now."""
        return replacement_cost_inr_per_kwh * wear_per_kwh(
            self.params,
            temperature_c=self.temperature_c,
            c_rate=self.crate_ewma if self.crate_ewma > 0 else 1.0,
            depth_of_discharge=max(self.max_depth_of_discharge, self.params.dod_reference),
            soh=self.soh,
        )

    def replacement_horizon_km(
        self, recent_wear_per_km: float | None
    ) -> float | None:
        """Supportable only with a recent wear rate; otherwise unknown."""
        if recent_wear_per_km is None or recent_wear_per_km <= 0:
            return None
        remaining = max(0.0, self.soh - self.params.soh_floor)
        horizon = remaining / recent_wear_per_km
        if not math.isfinite(horizon):
            return None
        return horizon

    # -- state transitions ------------------------------------------------
    def _bucket_temperature(self, dt_s: float) -> None:
        t = self.temperature_c
        if t < 10.0:
            self.temperature_seconds["below_10C"] += dt_s
        elif t < 25.0:
            self.temperature_seconds["10_to_25C"] += dt_s
        elif t < 35.0:
            self.temperature_seconds["25_to_35C"] += dt_s
        elif t < 45.0:
            self.temperature_seconds["35_to_45C"] += dt_s
        else:
            self.temperature_seconds["above_45C"] += dt_s

    def advance(
        self,
        energy_delta_kwh: float,
        dt_s: float,
        temperature_c: float,
        c_rate: float,
        depth_of_discharge: float | None = None,
    ) -> None:
        """Apply one integration step of battery-side energy flow."""
        capacity = self.effective_capacity_kwh()
        self.soc = _clamp01(self.soc - energy_delta_kwh / capacity)
        self.temperature_c = temperature_c
        self._bucket_temperature(max(dt_s, 0.0))
        throughput = abs(energy_delta_kwh)
        self.throughput_kwh += throughput
        self.equivalent_cycles += throughput / self.params.equivalent_cycle_capacity_kwh
        dod = 1.0 - self.soc if depth_of_discharge is None else depth_of_discharge
        self.max_depth_of_discharge = max(self.max_depth_of_discharge, dod)
        self.crate_ewma += (abs(c_rate) - self.crate_ewma) * self._ewma_alpha
        step_wear = wear_per_kwh(
            self.params,
            temperature_c=temperature_c,
            c_rate=abs(c_rate),
            depth_of_discharge=dod,
            soh=self.soh,
        )
        self.damage_index += step_wear * throughput
        self.soh = max(self.params.soh_floor, 1.0 - self.damage_index)

    def record_charge_session(
        self, energy_kwh: float, mean_power_kw: float, connector: str, soc_start: float
    ) -> None:
        self.charge_profile.append(
            {
                "energy_kwh": round(energy_kwh, 3),
                "mean_power_kw": round(mean_power_kw, 2),
                "connector": connector,
                "soc_start": round(soc_start, 3),
                "soc_end": round(self.soc, 3),
            }
        )
        self.charge_profile = self.charge_profile[-10:]

    # -- export -----------------------------------------------------------
    def to_state(self) -> BatteryState:
        return BatteryState(
            usable_energy_kwh=self.effective_capacity_kwh(),
            soc=self.soc,
            soh=self.soh,
            soh_uncertainty=self.current_uncertainty(),
            temperature_c=self.temperature_c,
            equivalent_throughput_kwh=self.throughput_kwh,
            equivalent_cycles=self.equivalent_cycles,
            max_charge_power_kw=charge_acceptance_kw(self.params, self.soc),
            charge_limit_soc=self.params.taper_start_soc,
        )

    def passport(
        self,
        replacement_cost_inr_per_kwh: float,
        recent_wear_per_km: float | None = None,
    ) -> BatteryPassport:
        horizon = self.replacement_horizon_km(recent_wear_per_km)
        return BatteryPassport(
            vehicle_id=self.vehicle_id,
            soh=self.soh,
            soh_uncertainty=self.current_uncertainty(),
            confidence=self.current_confidence(),
            equivalent_throughput_kwh=self.throughput_kwh,
            equivalent_cycles=self.equivalent_cycles,
            temperature_exposure={k: round(v / 3600.0, 3) for k, v in self.temperature_seconds.items()},
            recent_charge_profile=list(self.charge_profile),
            marginal_wear_cost_inr_per_kwh=self.marginal_wear_cost_inr_per_kwh(
                replacement_cost_inr_per_kwh
            ),
            replacement_horizon_km=horizon,
            evidence_status=EvidenceLabel.MODEL_ESTIMATE,
            notes="Semi-empirical throughput proxy; not a certified diagnosis.",
        )


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))
