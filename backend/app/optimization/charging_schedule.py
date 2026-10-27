"""Charging scheduler: day-ahead plan and rolling correction.

Both modes share one planner: for every vehicle with a known future mission,
compute the energy it will need (mission energy plus the configured
uncertainty reserve, capped by the operating band), then pick the
charger-and-time-slot combination with the lowest queue-adjusted total cost
that still fits the mission deadline. The day-ahead mode plans the full
horizon in configured slots; the rolling mode replans inside a short
look-ahead window with the same logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.config.loader import ConfigBundle, get_config
from app.optimization.dispatch import CostWeights, cost_weights
from app.schemas.charger import ChargerState
from app.schemas.common import ChargerStatus
from app.schemas.decision import ChargingAction
from app.schemas.reason_codes import ReasonCode
from app.schemas.scenario import Scenario
from app.simulation.charging import (
    ChargingEconomics,
    charge_duration_min,
    evaluate_charger,
    mission_target_soc,
    queue_wait_min,
)
from app.simulation.city_graph import GraphModel, RoutePlanningError
from app.simulation.model_config import battery_params

DAY_AHEAD = "day_ahead"
ROLLING = "rolling"


@dataclass(frozen=True)
class MissionNeed:
    """A scheduled mission that charging must support."""

    vehicle_id: str
    start_s: float
    deadline_s: float
    energy_kwh: float


@dataclass
class ScheduleResult:
    mode: str
    actions: list[ChargingAction]
    skipped: dict[str, str] = field(default_factory=dict)
    reason_codes: list[str] = field(default_factory=list)


def _economics(cfg: ConfigBundle, weights: CostWeights) -> ChargingEconomics:
    return ChargingEconomics(
        queue_cost_inr_per_min=float(cfg.opt("charging.queue_cost_inr_per_min")),
        detour_cost_inr_per_km=float(cfg.opt("charging.detour_cost_inr_per_km")),
        lost_productive_inr_per_min=float(cfg.opt("charging.lost_productive_revenue_inr_per_min")),
        energy_inr_per_kwh=weights.energy_inr_per_kwh,
        dc_surcharge_inr_per_kwh=float(cfg.eco("electricity.dc_fast_surcharge_inr_per_kwh")),
        wear_inr_per_kwh=weights.battery_replacement_inr_per_kwh
        / float(cfg.eco("capital.battery_replacement_inr_per_kwh"))
        * weights.energy_inr_per_kwh,
    )


def _slot_hours(cfg: ConfigBundle, mode: str, now_s: float) -> list[float]:
    """Slot start times (seconds from scenario start) for the given mode."""
    slot_min = float(cfg.opt("charging.day_ahead_slot_minutes"))
    horizon_h = float(cfg.opt("charging.day_ahead_horizon_h"))
    if mode == ROLLING:
        horizon_min = float(cfg.opt("charging.rolling_lookahead_min"))
        window_s = horizon_min * 60.0
    else:
        window_s = horizon_h * 3600.0
    slot_s = slot_min * 60.0
    start = (int(now_s / slot_s) + 1) * slot_s
    return [t for t in _frange(start, start + window_s, slot_s)]


def _frange(start: float, stop: float, step: float) -> list[float]:
    values: list[float] = []
    current = start
    while current < stop:
        values.append(current)
        current += step
    return values


def plan_charging(
    scenario: Scenario,
    missions: list[MissionNeed],
    *,
    mode: str = DAY_AHEAD,
    now_s: float = 0.0,
    cfg: ConfigBundle | None = None,
) -> ScheduleResult:
    """Plan charging for every mission; both scheduler modes share this path."""
    c = cfg or get_config()
    if mode not in (DAY_AHEAD, ROLLING):
        raise ValueError(f"unknown charging schedule mode '{mode}'")
    weights = cost_weights(c)
    economics = _economics(c, weights)
    model = GraphModel(scenario.graph, min_speed_factor=float(c.op("traffic.min_speed_factor")))
    reserve = float(c.op("soc_policy.soc_reserve_min"))
    operating_min = float(c.op("soc_policy.soc_operating_min"))
    operating_max = float(c.op("soc_policy.soc_operating_max"))
    uncertainty = float(c.opt("charging.mission_uncertainty_reserve"))
    max_target = float(c.opt("charging.max_target_soc"))
    slots = _slot_hours(c, mode, now_s)
    start_hour = int(c.demo_v("world_defaults.horizon_start_hour"))
    tariff_windows = (
        int(c.eco("electricity.peak_window_start_h")),
        int(c.eco("electricity.peak_window_end_h")),
        int(c.eco("electricity.offpeak_window_start_h")),
        int(c.eco("electricity.offpeak_window_end_h")),
    )
    tariff_bands = (
        float(c.eco("electricity.peak_inr_per_kwh")),
        float(c.eco("electricity.offpeak_inr_per_kwh")),
    )

    actions: list[ChargingAction] = []
    skipped: dict[str, str] = {}
    reasons: list[str] = []

    missions_by_vehicle: dict[str, list[MissionNeed]] = {}
    for mission in missions:
        missions_by_vehicle.setdefault(mission.vehicle_id, []).append(mission)

    for vehicle in scenario.vehicles:
        vehicle_missions = sorted(
            missions_by_vehicle.get(vehicle.vehicle_id, []), key=lambda m: m.start_s
        )
        if not vehicle_missions:
            continue
        battery = battery_params(vehicle.powertrain, c)
        first = vehicle_missions[0]
        total_energy = sum(m.energy_kwh for m in vehicle_missions)
        target_soc, target_reasons = mission_target_soc(
            battery, vehicle.soc, total_energy, uncertainty, max_target, operating_max
        )
        reasons.extend(target_reasons)
        if target_soc <= vehicle.soc + 1e-6:
            skipped[vehicle.vehicle_id] = ReasonCode.CHARGE_TARGET_ONLY_MISSION_ENERGY.value
            continue

        best: tuple[float, float, ChargerState, float, object] | None = None
        for charger in scenario.chargers:
            try:
                path = model.shortest_path(
                    vehicle.current_zone_id, charger.node_id, scenario.environment.traffic_factor
                )
            except RoutePlanningError:
                continue
            for slot_s in slots:
                hour = (start_hour + (slot_s / 3600.0)) % 24.0
                estimate = evaluate_charger(
                    charger=charger,
                    battery=battery,
                    economics=economics,
                    soc_now=vehicle.soc,
                    target_soc=target_soc,
                    detour_km=path.distance_km,
                    plug_count_in_use=charger.plugs_in_use,
                    hour_of_day=int(hour),
                    tariff_bands=tariff_bands,
                    tariff_windows=tariff_windows,
                )
                if not estimate.feasible:
                    continue
                ready_s = slot_s + estimate.plug_in_time_s
                if slot_s + estimate.plug_in_time_s > first.deadline_s:
                    continue
                total_cost = estimate.cost.total_inr
                if best is None or total_cost < best[0]:
                    best = (total_cost, slot_s, charger, ready_s, estimate)

        if best is None:
            skipped[vehicle.vehicle_id] = ReasonCode.DEADLINE_TIGHTENED_BY_QUEUE.value
            continue

        _total_cost, slot_s, charger, ready_s, estimate = best
        path = model.shortest_path(
            vehicle.current_zone_id, charger.node_id, scenario.environment.traffic_factor
        )
        duration_min, mean_power, battery_energy = charge_duration_min(
            battery, charger, charger.plugs_in_use, vehicle.soc, target_soc
        )
        wait = queue_wait_min(charger, charger.plugs_in_use)
        actions.append(
            ChargingAction(
                charger_id=charger.charger_id,
                vehicle_id=vehicle.vehicle_id,
                energy_kwh=round(estimate.energy_kwh, 3),
                effective_power_kw=round(mean_power, 2),
                charge_duration_min=round(duration_min, 2),
                queue_wait_min=round(wait, 2),
                detour_km=round(path.distance_km, 3),
                plug_in_time_s=round(slot_s, 1),
                target_soc=round(target_soc, 4),
                arrival_soc=round(vehicle.soc, 4),
                cost=estimate.cost,
                reason_codes=sorted(
                    {
                        ReasonCode.CHARGE_TARGET_ONLY_MISSION_ENERGY.value,
                        *target_reasons,
                    }
                ),
            )
        )
        if mode == ROLLING:
            reasons.append(ReasonCode.ROLLING_CORRECTION_APPLIED.value)
        else:
            reasons.append(ReasonCode.DAY_AHEAD_SCHEDULE_REPLANNED.value)

    return ScheduleResult(mode=mode, actions=actions, skipped=skipped, reason_codes=reasons)


def invalidate_plans(
    scenario: Scenario, offline_charger_ids: list[str]
) -> list[str]:
    """Vehicle ids whose plan depended on a charger that just went offline."""
    if not offline_charger_ids:
        return []
    affected = []
    for charger in scenario.chargers:
        if charger.charger_id in offline_charger_ids and charger.status != ChargerStatus.ONLINE:
            affected.append(charger.charger_id)
    return affected
