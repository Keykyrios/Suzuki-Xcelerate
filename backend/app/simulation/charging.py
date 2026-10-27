"""Charging model: queue, detour, site power, taper, deadlines, and cost.

`C_charge = electricity + waiting + detour + lost_productive + degradation`.
Every term comes from configuration; travel-to-charger, queue, plug time and
delivery estimates are computed here and consumed by the dispatch optimizer
and the charging scheduler.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from app.schemas.charger import ChargerState
from app.schemas.common import ChargerStatus
from app.schemas.decision import ChargingCostBreakdown
from app.schemas.reason_codes import ReasonCode
from app.simulation.model_config import BatteryModelParams
from app.simulation.battery import charge_acceptance_kw


@dataclass(frozen=True)
class ChargingEconomics:
    """Monetary weights applied to a charging event (all INR)."""

    queue_cost_inr_per_min: float
    detour_cost_inr_per_km: float
    lost_productive_inr_per_min: float
    energy_inr_per_kwh: str | float
    dc_surcharge_inr_per_kwh: float
    wear_inr_per_kwh: float


@dataclass(frozen=True)
class ChargeEstimate:
    """Result of evaluating one vehicle-charger pairing."""

    charger_id: str
    energy_kwh: float
    effective_power_kw: float
    charge_duration_min: float
    queue_wait_min: float
    detour_km: float
    plug_in_time_s: float
    target_soc: float
    arrival_soc: float
    cost: ChargingCostBreakdown
    feasible: bool
    reason_codes: list[str]


def effective_charge_power_kw(
    battery: BatteryModelParams,
    charger: ChargerState,
    plug_count_in_use: int,
    soc: float,
    connector: str | None = None,
) -> tuple[float, list[str]]:
    """Power actually delivered: min(vehicle taper, charger rating, site share).

    Returns the power and the constraint reason codes that bound it.
    """
    reasons: list[str] = []
    conn = connector or (charger.connectors[0] if charger.connectors else "DC")
    vehicle_limit = charge_acceptance_kw(battery, soc, connector=conn)
    charger_limit = charger.rated_power_kw

    site_share = charger.site_limit_kw / max(1, plug_count_in_use)
    site_limited = site_share < charger_limit
    if site_limited:
        reasons.append(ReasonCode.CHARGER_SITE_LIMIT_APPLIED.value)
        charger_limit = site_share

    if charger_limit >= battery.max_charge_power_dc_kw:
        reasons.append(ReasonCode.CHARGER_VEHICLE_LIMIT_APPLIED.value)

    power = min(vehicle_limit, charger_limit)
    if soc > battery.taper_start_soc:
        reasons.append(ReasonCode.CHARGER_TAPER_APPLIED.value)
    if power <= 0.0 or not math.isfinite(power):
        reasons.append(ReasonCode.CHARGER_ENERGY_DELIVERY_CONSTRAINED.value)
        power = 0.0
    return power, reasons


def energy_required_kwh(
    battery: BatteryModelParams, soc: float, target_soc: float
) -> float:
    """Battery-side energy needed to move from `soc` to `target_soc`."""
    target = max(0.0, min(target_soc, 1.0))
    if target <= soc:
        return 0.0
    return battery.usable_energy_kwh * (target - soc)


def charge_duration_min(
    battery: BatteryModelParams,
    charger: ChargerState,
    plug_count_in_use: int,
    soc_start: float,
    soc_end: float,
    connector: str | None = None,
) -> tuple[float, float, float]:
    """Integrated delivery time under the taper model.

    Returns (minutes, mean_effective_kw, energy_kwh) — the taper curve is
    integrated step by step instead of assuming constant peak power.
    """
    if soc_end <= soc_start:
        return 0.0, 0.0, 0.0
    dt_s = 30.0
    soc = soc_start
    energy = 0.0
    elapsed_s = 0.0
    power_integral = 0.0
    while soc < soc_end - 1e-9 and elapsed_s < 24 * 3600:
        power, _reasons = effective_charge_power_kw(
            battery, charger, plug_count_in_use, soc, connector
        )
        if power <= 0.0:
            break
        delta = power * dt_s / 3600.0 / max(battery.usable_energy_kwh, 1e-6)
        soc = min(soc_end, soc + delta)
        energy += power * dt_s / 3600.0
        power_integral += power * dt_s
        elapsed_s += dt_s
    mean_power = power_integral / elapsed_s if elapsed_s > 0 else 0.0
    return elapsed_s / 60.0, mean_power, energy


def queue_wait_min(charger: ChargerState, plug_count_in_use: int) -> float:
    """Estimated wait for a free plug given the current queue."""
    available = max(0, charger.plug_count - plug_count_in_use)
    if available > 0 and charger.queue_length == 0:
        return 0.0
    ahead = charger.queue_length + max(0, plug_count_in_use - charger.plug_count)
    if ahead <= 0:
        return 0.0
    # Sessions ahead are served in parallel across available plugs.
    waves = math.ceil(ahead / max(min(charger.plug_count, max(available, 1)), 1))
    per_session_min = max(charger.queue_estimate_min, 1.0)
    return waves * per_session_min


def grid_energy_kwh(battery_energy_kwh: float, battery: BatteryModelParams) -> float:
    """Battery energy converted to metered grid energy through charge efficiency."""
    return battery_energy_kwh / max(battery.charge_efficiency, 1e-6)


def electricity_cost_inr(
    grid_kwh: float,
    tariff_inr_per_kwh: float,
    is_dc_fast: bool,
    dc_surcharge_inr_per_kwh: float,
    hour_of_day: int,
    bands: tuple[float, float],
    windows: tuple[int, int, int, int],
) -> float:
    """Station tariff with time-of-use overrides and a DC fast surcharge.

    The station's own tariff is the shoulder price; the configured peak band
    raises the rate inside the evening peak window and the off-peak band lowers
    it inside the night window.
    """
    peak_start, peak_end, offpeak_start, offpeak_end = windows
    peak_band, offpeak_band = bands
    rate = tariff_inr_per_kwh
    if _in_window(hour_of_day, peak_start, peak_end):
        rate = max(rate, peak_band)
    elif _in_window(hour_of_day, offpeak_start, offpeak_end):
        rate = min(rate, offpeak_band)
    if is_dc_fast:
        rate += dc_surcharge_inr_per_kwh
    return grid_kwh * rate


def _in_window(hour: int, start: int, end: int) -> bool:
    if start <= end:
        return start <= hour < end
    return hour >= start or hour < end


def evaluate_charger(
    *,
    charger: ChargerState,
    battery: BatteryModelParams,
    economics: ChargingEconomics,
    soc_now: float,
    target_soc: float,
    detour_km: float,
    detour_time_min: float = 0.0,
    plug_count_in_use: int,
    hour_of_day: int,
    tariff_bands: tuple[float, float],
    tariff_windows: tuple[int, int, int, int],
    connector: str | None = None,
) -> ChargeEstimate:
    """Full cost/feasibility estimate for charging this vehicle at this charger."""
    reasons: list[str] = []
    if charger.status != ChargerStatus.ONLINE:
        outline = ChargeEstimate(
            charger_id=charger.charger_id,
            energy_kwh=0.0,
            effective_power_kw=0.0,
            charge_duration_min=0.0,
            queue_wait_min=0.0,
            detour_km=detour_km,
            plug_in_time_s=0.0,
            target_soc=target_soc,
            arrival_soc=soc_now,
            cost=ChargingCostBreakdown(
                electricity_inr=0.0, waiting_inr=0.0, detour_inr=0.0,
                lost_productive_inr=0.0, degradation_inr=0.0,
            ),
            feasible=False,
            reason_codes=[ReasonCode.CHARGER_OFFLINE.value],
        )
        return outline

    duration_min, mean_power, battery_energy = charge_duration_min(
        battery, charger, plug_count_in_use, soc_now, target_soc, connector
    )
    _, power_reasons = effective_charge_power_kw(
        battery, charger, plug_count_in_use, soc_now, connector
    )
    reasons.extend(power_reasons)

    wait = queue_wait_min(charger, plug_count_in_use)
    if wait > 0:
        reasons.append(ReasonCode.DEADLINE_TIGHTENED_BY_QUEUE.value)

    grid_kwh = grid_energy_kwh(battery_energy, battery)
    electricity = electricity_cost_inr(
        grid_kwh,
        charger.tariff_inr_per_kwh,
        is_dc_fast=charger.rated_power_kw >= 50.0,
        dc_surcharge_inr_per_kwh=float(economics.dc_surcharge_inr_per_kwh),
        hour_of_day=hour_of_day,
        bands=tariff_bands,
        windows=tariff_windows,
    )
    waiting = wait * economics.queue_cost_inr_per_min
    detour = detour_km * economics.detour_cost_inr_per_km
    lost = (
        wait + duration_min + max(0.0, detour_time_min)
    ) * economics.lost_productive_inr_per_min
    degradation = battery_energy * economics.wear_inr_per_kwh

    plug_in_time_s = (wait + duration_min + max(0.0, detour_time_min)) * 60.0

    feasible = duration_min > 0.0 and mean_power > 0.0

    return ChargeEstimate(
        charger_id=charger.charger_id,
        energy_kwh=round(battery_energy, 3),
        effective_power_kw=round(mean_power, 2),
        charge_duration_min=round(duration_min, 2),
        queue_wait_min=round(wait, 2),
        detour_km=round(detour_km, 3),
        plug_in_time_s=round(plug_in_time_s, 1),
        target_soc=round(target_soc, 4),
        arrival_soc=round(soc_now, 4),
        cost=ChargingCostBreakdown(
            electricity_inr=round(electricity, 2),
            waiting_inr=round(waiting, 2),
            detour_inr=round(detour, 2),
            lost_productive_inr=round(lost, 2),
            degradation_inr=round(degradation, 2),
        ),
        feasible=feasible,
        reason_codes=sorted(set(reasons)),
    )


def mission_target_soc(
    battery: BatteryModelParams,
    soc_now: float,
    mission_energy_kwh: float,
    uncertainty_reserve: float,
    max_target_soc: float,
    operating_max: float,
) -> tuple[float, list[str]]:
    """Charge only for the next mission plus reserve, capped by the operating band."""
    reasons: list[str] = [ReasonCode.CHARGE_TARGET_ONLY_MISSION_ENERGY.value]
    target = soc_now + (
        mission_energy_kwh * (1.0 + max(0.0, uncertainty_reserve))
        / max(battery.usable_energy_kwh, 1e-6)
    )
    cap = min(max_target_soc, operating_max)
    if target > cap:
        target = cap
        reasons.append(ReasonCode.MISSION_TARGET_SOC_CAPPED.value)
    return max(soc_now, min(target, 1.0)), reasons


def fits_deadline(est: ChargeEstimate, available_at_s: float, deadline_s: float) -> bool:
    """Travel + queue + plug time must fit inside the mission deadline."""
    return available_at_s + est.plug_in_time_s <= deadline_s


def total_cost_inr(cost: ChargingCostBreakdown) -> float:
    return (
        cost.electricity_inr
        + cost.waiting_inr
        + cost.detour_inr
        + cost.lost_productive_inr
        + cost.degradation_inr
    )