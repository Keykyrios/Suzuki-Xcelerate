"""Economics reporting for the operator panel and evidence exports.

Every figure this service returns is either a configured assumption or a value
already computed by the model; nothing is estimated on the panel's behalf.
"""

from __future__ import annotations

from dataclasses import asdict

from app.config.loader import ConfigBundle, get_config
from app.economics.model import (
    TcoBreakdown,
    capital_daily_inr,
    cost_structure,
    fleet_day_economics,
    reference_case,
    time_of_use_tariff,
    tco_breakdown,
)
from app.schemas.common import EvidenceLabel, PowertrainType
from app.schemas.decision import Decision
from app.services.state_store import StateStore, get_state_store

CURRENCY_PATHS = ("currency.code", "currency.base_year")


def get_economics_service(
    cfg: ConfigBundle | None = None, store: StateStore | None = None
) -> "EconomicsService":
    return EconomicsService(cfg or get_config(), store or get_state_store())


class EconomicsService:
    """Read-only economics layer; nothing here mutates canonical state."""

    def __init__(self, cfg: ConfigBundle, store: StateStore):
        self.cfg = cfg
        self.store = store

    # -- operator panel ---------------------------------------------------
    def cost_structure_report(self) -> dict:
        """Configured cost structure for both configurations, in INR."""
        out: dict[str, dict[str, float | str]] = {}
        for powertrain in PowertrainType:
            structure = cost_structure(powertrain, self.cfg)
            payload = asdict(structure)
            payload["capital_daily_inr"] = capital_daily_inr(powertrain, self.cfg)
            payload["currency"] = self.cfg.eco(CURRENCY_PATHS[0])
            payload["evidence_status"] = self.cfg.economics.parameter(
                "operating_costs.maintenance_reserve_inr_per_km"
            ).status.value
            out[powertrain.value] = payload
        return out

    def tariff_report(self, hour_of_day: int = 12) -> dict:
        """Configured time-of-use bands plus the effective band for one hour."""
        band, label = time_of_use_tariff(hour_of_day, self.cfg)
        peak_param = self.cfg.economics.parameter("electricity.peak_inr_per_kwh")
        return {
            "hour_of_day": hour_of_day,
            "band": label,
            "effective_inr_per_kwh": band,
            "peak_inr_per_kwh": float(peak_param.value),
            "offpeak_inr_per_kwh": float(self.cfg.eco("electricity.offpeak_inr_per_kwh")),
            "shoulder_inr_per_kwh": float(self.cfg.eco("electricity.shoulder_inr_per_kwh")),
            "dc_fast_surcharge_inr_per_kwh": float(
                self.cfg.eco("electricity.dc_fast_surcharge_inr_per_kwh")
            ),
            "peak_window_h": [
                int(self.cfg.eco("electricity.peak_window_start_h")),
                int(self.cfg.eco("electricity.peak_window_end_h")),
            ],
            "offpeak_window_h": [
                int(self.cfg.eco("electricity.offpeak_window_start_h")),
                int(self.cfg.eco("electricity.offpeak_window_end_h")),
            ],
            "currency": self.cfg.eco(CURRENCY_PATHS[0]),
            "base_year": int(self.cfg.eco(CURRENCY_PATHS[1])),
            "evidence_status": peak_param.status.value,
        }

    def tco_report(
        self,
        *,
        energy_kwh_per_km: float | None = None,
        energy_price_inr_per_kwh: float | None = None,
        battery_wear_inr_per_kwh: float | None = None,
        utilisation_km_per_day: float | None = None,
        hour_of_day: int = 12,
        decision: Decision | None = None,
    ) -> dict:
        """Cost per total vehicle-kilometre for both configurations.

        Energy intensity defaults to the recorded decision's optimized metrics;
        with no decision and no explicit input the report refuses to invent a
        consumption figure.
        """
        if energy_kwh_per_km is None:
            energy_kwh_per_km = self._decided_energy_intensity(decision)
        if energy_kwh_per_km is None:
            raise ValueError(
                "energy_kwh_per_km is required until a decision has been recorded"
            )
        if energy_price_inr_per_kwh is None:
            energy_price_inr_per_kwh, _label = time_of_use_tariff(hour_of_day, self.cfg)
        if battery_wear_inr_per_kwh is None:
            battery_wear_inr_per_kwh = float(
                self.cfg.eco("capital.battery_replacement_inr_per_kwh")
            ) * float(self.cfg.v("degradation.throughput_wear_per_kwh"))

        out: dict[str, dict[str, float]] = {}
        for powertrain in PowertrainType:
            breakdown = tco_breakdown(
                energy_kwh_per_km=energy_kwh_per_km,
                energy_price_inr_per_kwh=energy_price_inr_per_kwh,
                battery_wear_inr_per_kwh=battery_wear_inr_per_kwh,
                utilisation_km_per_day=utilisation_km_per_day,
                powertrain=powertrain,
                cfg=self.cfg,
            )
            report = dict(asdict(breakdown))
            report["non_energy_inr_per_km"] = breakdown.non_energy_inr_per_km
            out[powertrain.value] = report
        return {
            "energy_kwh_per_km": round(float(energy_kwh_per_km), 4),
            "energy_price_inr_per_kwh": round(float(energy_price_inr_per_kwh), 3),
            "battery_wear_inr_per_kwh": round(float(battery_wear_inr_per_kwh), 3),
            "hour_of_day": hour_of_day,
            "by_powertrain": out,
            "currency": self.cfg.eco(CURRENCY_PATHS[0]),
            "evidence_status": EvidenceLabel.MODEL_ESTIMATE.value,
        }

    def fleet_day_report(self, decision: Decision | None = None) -> dict:
        """One operating day from the recorded decision; honest when absent."""
        decision = decision or self.store.decision
        metrics = None if decision is None else decision.optimized_metrics
        if metrics is None:
            return {
                "available": False,
                "reason": "no decision has been recorded in this session",
                "reference_case": reference_case(self.cfg),
                "evidence_status": EvidenceLabel.SCENARIO_ASSUMPTION.value,
            }
        powertrain = decision.decision.powertrain_recommendation
        day = fleet_day_economics(
            gross_revenue_inr=metrics.gross_revenue_inr,
            energy_cost_inr=metrics.energy_cost_inr,
            wear_cost_inr=metrics.wear_cost_inr,
            charging_opportunity_cost_inr=metrics.charging_opportunity_cost_inr,
            missed_service_cost_inr=metrics.missed_service_cost_inr,
            total_km=metrics.total_km,
            powertrain=powertrain,
            cfg=self.cfg,
        )
        payload = asdict(day)
        payload.update(
            {
                "contribution_inr": day.contribution_inr,
                "operating_cost_inr": day.operating_cost_inr,
                "owner_driver_cashflow_inr": day.owner_driver_cashflow_inr,
                "powertrain": powertrain.value,
                "run_id": decision.run_id,
                "currency": self.cfg.eco(CURRENCY_PATHS[0]),
                "evidence_status": decision.evidence_status.value,
            }
        )
        return payload

    def reference_case_report(self) -> dict:
        """Stored concept-case outputs, exposed as model estimates."""
        return {
            "metrics": reference_case(self.cfg),
            "currency": self.cfg.eco(CURRENCY_PATHS[0]),
            "base_year": int(self.cfg.eco(CURRENCY_PATHS[1])),
            "evidence_status": EvidenceLabel.SCENARIO_ASSUMPTION.value,
            "source_or_rationale": self.cfg.economics.parameter(
                "reference_case.tco_inr_per_km"
            ).source_or_rationale,
        }

    # -- helpers ----------------------------------------------------------
    def _decided_energy_intensity(self, decision: Decision | None) -> float | None:
        decision = decision or self.store.decision
        if decision is None or decision.optimized_metrics is None:
            return None
        return decision.optimized_metrics.energy_per_vehicle_km_kwh

    def currency(self) -> str:
        return str(self.cfg.eco(CURRENCY_PATHS[0]))