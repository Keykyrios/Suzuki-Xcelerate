"""Pure economics calculations: fares, cost structure, TCO, and contribution.

Every figure is computed from configured assumptions in constant 2026 INR.
Reference-case numbers stored in configuration are model outputs of the concept
business case; they are reproduced for comparison, never asserted as results.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config.loader import ConfigBundle, get_config
from app.schemas.common import CabinMode, PowertrainType


@dataclass(frozen=True)
class FareBreakdown:
    base_inr: float
    per_pax_km_inr: float
    fixed_inr: float
    minimum_applied: bool

    @property
    def total_inr(self) -> float:
        return self.base_inr + self.per_pax_km_inr + self.fixed_inr


def fare_for(
    service_class: CabinMode,
    passengers: int,
    distance_km: float,
    cfg: ConfigBundle | None = None,
) -> FareBreakdown:
    """Configured fare for one trip; no fare is derived outside this function."""
    c = cfg or get_config()
    key = f"fares.{service_class.value.lower()}_inr_per_pax_km"
    if c.economics.__contains__(key):
        rate = float(c.eco(key))
    else:
        rate = float(c.eco("fares.pool_inr_per_pax_km"))
    per_pax = rate * max(passengers, 0) * max(distance_km, 0.0)
    fixed = float(c.eco("fares.airport_fixed_inr")) if service_class == CabinMode.AIRPORT else 0.0
    minimum = float(c.eco("fares.min_fare_inr"))
    base = 0.0
    total = per_pax + fixed
    applied_min = False
    if total < minimum and (passengers > 0 or distance_km > 0):
        base = minimum - total
        applied_min = True
    return FareBreakdown(
        base_inr=round(base, 2),
        per_pax_km_inr=round(per_pax, 2),
        fixed_inr=round(fixed, 2),
        minimum_applied=applied_min,
    )


def time_of_use_tariff(hour_of_day: int, cfg: ConfigBundle | None = None) -> tuple[float, str]:
    """Effective electricity tariff for an hour: peak / off-peak / shoulder."""
    c = cfg or get_config()
    peak = float(c.eco("electricity.peak_inr_per_kwh"))
    offpeak = float(c.eco("electricity.offpeak_inr_per_kwh"))
    shoulder = float(c.eco("electricity.shoulder_inr_per_kwh"))
    peak_start = int(c.eco("electricity.peak_window_start_h"))
    peak_end = int(c.eco("electricity.peak_window_end_h"))
    off_start = int(c.eco("electricity.offpeak_window_start_h"))
    off_end = int(c.eco("electricity.offpeak_window_end_h"))

    def in_window(hour: int, start: int, end: int) -> bool:
        if start <= end:
            return start <= hour < end
        return hour >= start or hour < end

    if in_window(hour_of_day, peak_start, peak_end):
        return peak, "peak"
    if in_window(hour_of_day, off_start, off_end):
        return offpeak, "offpeak"
    return shoulder, "shoulder"


@dataclass(frozen=True)
class CostStructure:
    """Daily and per-km operating cost inputs derived from configuration."""

    maintenance_inr_per_km: float
    tyre_inr_per_km: float
    insurance_inr_per_day: float
    downtime_cost_inr_per_h: float
    driver_daily_inr: float
    capital_daily_inr: float
    battery_replacement_inr_per_kwh: float
    reference_daily_km: float

    @property
    def fixed_daily_inr(self) -> float:
        return self.insurance_inr_per_day + self.capital_daily_inr


def capital_daily_inr(powertrain: PowertrainType, cfg: ConfigBundle | None = None) -> float:
    """Depreciation plus straight-line finance, net of residual value, per day."""
    c = cfg or get_config()
    capex = float(c.eco("capital.vehicle_capex_inr"))
    if powertrain == PowertrainType.NEXUS_X:
        capex += float(c.eco("capital.rex_capex_delta_inr"))
    years = float(c.eco("capital.depreciation_years"))
    residual = float(c.eco("capital.residual_fraction"))
    rate = float(c.eco("capital.finance_rate_per_year"))
    depreciation = capex * (1.0 - residual) / max(years, 1.0) / 365.0
    interest = capex * rate / 365.0
    return round(depreciation + interest, 2)


def cost_structure(powertrain: PowertrainType, cfg: ConfigBundle | None = None) -> CostStructure:
    c = cfg or get_config()
    maintenance = float(c.eco("operating_costs.maintenance_reserve_inr_per_km"))
    if powertrain == PowertrainType.NEXUS_X:
        maintenance += float(c.opt("powertrain.rex_maintenance_delta_inr_per_km"))
    return CostStructure(
        maintenance_inr_per_km=maintenance,
        tyre_inr_per_km=float(c.eco("operating_costs.tyre_reserve_inr_per_km")),
        insurance_inr_per_day=float(c.eco("operating_costs.insurance_inr_per_day")),
        downtime_cost_inr_per_h=float(c.eco("operating_costs.downtime_cost_inr_per_h")),
        driver_daily_inr=float(c.eco("operating_costs.driver_daily_cost_inr")),
        capital_daily_inr=capital_daily_inr(powertrain, c),
        battery_replacement_inr_per_kwh=float(c.eco("capital.battery_replacement_inr_per_kwh")),
        reference_daily_km=float(c.eco("capital.daily_operating_km")),
    )


@dataclass(frozen=True)
class TripEconomics:
    revenue_inr: float
    energy_cost_inr: float
    wear_cost_inr: float
    deadhead_cost_inr: float
    service_cost_inr: float

    @property
    def contribution_inr(self) -> float:
        return self.revenue_inr - (
            self.energy_cost_inr + self.wear_cost_inr + self.deadhead_cost_inr + self.service_cost_inr
        )


def trip_economics(
    *,
    service_class: CabinMode,
    passengers: int,
    trip_distance_km: float,
    deadhead_km: float,
    energy_kwh: float,
    energy_price_inr_per_kwh: float,
    wear_cost_inr: float,
    structure: CostStructure,
    pickup_time_min: float,
    pickup_time_weight_inr_per_min: float,
) -> TripEconomics:
    """Revenue and direct cost of one served request."""
    fare = fare_for(service_class, passengers, trip_distance_km)
    service_cost = deadhead_km * structure.maintenance_inr_per_km + pickup_time_min * (
        pickup_time_weight_inr_per_min
    )
    return TripEconomics(
        revenue_inr=fare.total_inr,
        energy_cost_inr=energy_kwh * energy_price_inr_per_kwh,
        wear_cost_inr=wear_cost_inr,
        deadhead_cost_inr=deadhead_km * (
            structure.maintenance_inr_per_km + structure.tyre_inr_per_km
        ),
        service_cost_inr=round(service_cost, 2),
    )


@dataclass(frozen=True)
class TcoBreakdown:
    """Total cost of ownership per kilometre, from configured assumptions."""

    energy_inr_per_km: float
    maintenance_inr_per_km: float
    tyre_inr_per_km: float
    capital_inr_per_km: float
    insurance_inr_per_km: float
    battery_wear_inr_per_km: float
    tco_inr_per_km: float
    utilisation_km_per_day: float

    @property
    def non_energy_inr_per_km(self) -> float:
        return (
            self.maintenance_inr_per_km
            + self.tyre_inr_per_km
            + self.capital_inr_per_km
            + self.insurance_inr_per_km
            + self.battery_wear_inr_per_km
        )


def tco_breakdown(
    *,
    energy_kwh_per_km: float,
    energy_price_inr_per_kwh: float,
    battery_wear_inr_per_kwh: float,
    utilisation_km_per_day: float | None = None,
    powertrain: PowertrainType = PowertrainType.NEXUS_E,
    cfg: ConfigBundle | None = None,
) -> TcoBreakdown:
    """Cost per total vehicle-kilometre at the configured utilisation."""
    c = cfg or get_config()
    structure = cost_structure(powertrain, c)
    km = utilisation_km_per_day if utilisation_km_per_day is not None else structure.reference_daily_km
    km = max(km, 1.0)
    capital = structure.capital_daily_inr / km
    insurance = structure.insurance_inr_per_day / km
    energy = energy_kwh_per_km * energy_price_inr_per_kwh
    wear = battery_wear_inr_per_kwh * energy_kwh_per_km
    total = energy + structure.maintenance_inr_per_km + structure.tyre_inr_per_km + capital + insurance + wear
    return TcoBreakdown(
        energy_inr_per_km=round(energy, 3),
        maintenance_inr_per_km=round(structure.maintenance_inr_per_km, 3),
        tyre_inr_per_km=round(structure.tyre_inr_per_km, 3),
        capital_inr_per_km=round(capital, 3),
        insurance_inr_per_km=round(insurance, 3),
        battery_wear_inr_per_km=round(wear, 3),
        tco_inr_per_km=round(total, 3),
        utilisation_km_per_day=km,
    )


@dataclass(frozen=True)
class FleetDayEconomics:
    gross_revenue_inr: float
    energy_cost_inr: float
    wear_cost_inr: float
    charging_opportunity_cost_inr: float
    missed_service_cost_inr: float
    fixed_daily_inr: float
    maintenance_tyre_inr: float
    driver_cost_inr: float

    @property
    def contribution_inr(self) -> float:
        return self.gross_revenue_inr - self.operating_cost_inr

    @property
    def operating_cost_inr(self) -> float:
        return (
            self.energy_cost_inr
            + self.wear_cost_inr
            + self.charging_opportunity_cost_inr
            + self.missed_service_cost_inr
            + self.fixed_daily_inr
            + self.maintenance_tyre_inr
        )

    @property
    def owner_driver_cashflow_inr(self) -> float:
        return self.contribution_inr - self.driver_cost_inr


def fleet_day_economics(
    *,
    gross_revenue_inr: float,
    energy_cost_inr: float,
    wear_cost_inr: float,
    charging_opportunity_cost_inr: float,
    missed_service_cost_inr: float,
    total_km: float,
    driver_cost_inr: float | None = None,
    powertrain: PowertrainType = PowertrainType.NEXUS_E,
    cfg: ConfigBundle | None = None,
) -> FleetDayEconomics:
    """One operating day of fleet economics at configured cost structure."""
    c = cfg or get_config()
    structure = cost_structure(powertrain, c)
    driver = driver_cost_inr if driver_cost_inr is not None else structure.driver_daily_inr
    return FleetDayEconomics(
        gross_revenue_inr=round(gross_revenue_inr, 2),
        energy_cost_inr=round(energy_cost_inr, 2),
        wear_cost_inr=round(wear_cost_inr, 2),
        charging_opportunity_cost_inr=round(charging_opportunity_cost_inr, 2),
        missed_service_cost_inr=round(missed_service_cost_inr, 2),
        fixed_daily_inr=round(structure.fixed_daily_inr, 2),
        maintenance_tyre_inr=round(
            total_km * (structure.maintenance_inr_per_km + structure.tyre_inr_per_km), 2
        ),
        driver_cost_inr=round(driver, 2),
    )


def reference_case(cfg: ConfigBundle | None = None) -> dict[str, float]:
    """The stored reference-case outputs, exposed with their model-estimate label."""
    c = cfg or get_config()
    return {
        "tco_inr_per_km": float(c.eco("reference_case.tco_inr_per_km")),
        "owner_driver_income_inr_per_day": float(c.eco("reference_case.owner_driver_income_inr_per_day")),
        "operator_contribution_inr_per_vehicle_day": float(
            c.eco("reference_case.operator_contribution_inr_per_vehicle_day")
        ),
        "breakeven_km_per_day": float(c.eco("reference_case.breakeven_km_per_day")),
        "utilisation_km_per_day": float(c.eco("reference_case.utilisation_km_per_day")),
    }
