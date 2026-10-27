"""Transparent baseline dispatcher.

Serves each request with the nearest available vehicle that passes the same
hard feasibility checks the optimizer uses. When no vehicle can serve a
mission on its current charge, the baseline plugs the best-positioned vehicle
into the earliest suitable charger and retries. It performs no proactive
repositioning and no risk modeling - it is the deliberately simple challenger
the optimizer must beat.
"""

from __future__ import annotations

from app.config.loader import ConfigBundle, get_config
from app.optimization.dispatch import (
    Candidate,
    CostWeights,
    cost_weights,
    evaluate_candidate,
)
from app.schemas.common import SolverStatus
from app.schemas.decision import SolverReport
from app.schemas.reason_codes import ReasonCode
from app.schemas.scenario import Scenario, TripRequest
from app.simulation.charging import (
    ChargingEconomics,
    evaluate_charger,
    mission_target_soc,
)
from app.simulation.city_graph import GraphModel
from app.simulation.model_config import battery_params

BASELINE_SOLVER_NAME = "baseline_greedy"


def _charging_economics(cfg: ConfigBundle, weights: CostWeights) -> ChargingEconomics:
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


def best_charger_for(
    scenario: Scenario,
    vehicle_id: str,
    target_soc: float,
    cfg: ConfigBundle,
    weights: CostWeights,
    hour_of_day: int,
) -> tuple[str, float] | None:
    """Earliest suitable charger by plug-in time; None when nothing qualifies."""
    vehicle = scenario.vehicle(vehicle_id)
    model = GraphModel(scenario.graph, min_speed_factor=float(cfg.op("traffic.min_speed_factor")))
    economics = _charging_economics(cfg, weights)
    best: tuple[str, float] | None = None
    for charger in scenario.chargers:
        try:
            path = model.shortest_path(
                vehicle.current_zone_id, charger.node_id, scenario.environment.traffic_factor
            )
        except Exception:  # noqa: BLE001 - unreachable charger sites are simply skipped
            continue
        estimate = evaluate_charger(
            charger=charger,
            battery=battery_params(vehicle.powertrain, cfg),
            economics=economics,
            soc_now=vehicle.soc,
            target_soc=target_soc,
            detour_km=path.distance_km,
            plug_count_in_use=charger.plugs_in_use,
            hour_of_day=hour_of_day,
            tariff_bands=(
                float(cfg.eco("electricity.peak_inr_per_kwh")),
                float(cfg.eco("electricity.offpeak_inr_per_kwh")),
            ),
            tariff_windows=(
                int(cfg.eco("electricity.peak_window_start_h")),
                int(cfg.eco("electricity.peak_window_end_h")),
                int(cfg.eco("electricity.offpeak_window_start_h")),
                int(cfg.eco("electricity.offpeak_window_end_h")),
            ),
        )
        if not estimate.feasible:
            continue
        plug_in_total_s = path.travel_time_min * 60.0 + estimate.plug_in_time_s
        if best is None or plug_in_total_s < best[1]:
            best = (charger.charger_id, plug_in_total_s)
    return best


def baseline_plan(
    scenario: Scenario,
    cfg: ConfigBundle | None = None,
    requests: list[TripRequest] | None = None,
):
    """Nearest-feasible-vehicle assignment; returns a dispatch-shaped outcome."""
    from app.optimization.dispatch import DispatchOutcome

    c = cfg or get_config()
    pending = sorted(requests or list(scenario.requests), key=lambda r: r.request_time_s)
    weights = cost_weights(c)
    model = GraphModel(scenario.graph, min_speed_factor=float(c.op("traffic.min_speed_factor")))
    hour = int(c.demo_v("world_defaults.horizon_start_hour"))

    assignments: list[tuple[Candidate, TripRequest]] = []
    unserved: list[str] = []
    rejected: list[tuple[str, str, tuple[str, ...]]] = []
    charging_backed: list[str] = []
    assigned_vehicles: set[str] = set()

    for request in pending:
        options: list[Candidate] = []
        for vehicle in scenario.vehicles:
            if vehicle.vehicle_id in assigned_vehicles:
                continue
            candidate = evaluate_candidate(scenario, vehicle, request, cfg=c, graph=model)
            if candidate.feasible:
                options.append(candidate)
            else:
                rejected.append(
                    (request.request_id, vehicle.vehicle_id, candidate.reason_codes)
                )

        if not options:
            # Charging fallback: give a positionally suitable vehicle the energy
            # it needs at the earliest suitable charger, then re-check.
            charged = _baseline_charge_then_serve(
                scenario, request, c, weights, model, hour, assigned_vehicles
            )
            if charged is not None:
                candidate, charger_id = charged
                assignments.append((candidate, request))
                assigned_vehicles.add(candidate.vehicle_id)
                charging_backed.append(f"{request.request_id}:{charger_id}")
                continue

            unserved.append(request.request_id)
            continue

        options.sort(key=lambda cand: (cand.deadhead_km, cand.pickup_eta_s))
        chosen = options[0]
        assignments.append((chosen, request))
        assigned_vehicles.add(chosen.vehicle_id)

    reason_codes = [ReasonCode.BASELINE_SELECTED_NEAREST.value]
    reason_codes.extend(f"{ReasonCode.CHARGER_CONNECTED.value}:{item}" for item in charging_backed)
    for request_id in unserved:
        reason_codes.append(
            f"{ReasonCode.REQUEST_UNASSIGNED_BASELINE.value}:{request_id}"
        )

    return DispatchOutcome(
        assignments=assignments,
        unserved=unserved,
        rejected=rejected,
        solver=SolverReport(
            name=BASELINE_SOLVER_NAME,
            version="1.0.0",
            status=SolverStatus.FEASIBLE,
            runtime_ms=0.0,
            message="transparent nearest-vehicle baseline",
        ),
        risk=None,
        reason_codes=reason_codes,
        status=SolverStatus.FEASIBLE,
    )


def _baseline_charge_then_serve(
    scenario: Scenario,
    request: TripRequest,
    cfg: ConfigBundle,
    weights: CostWeights,
    model: GraphModel,
    hour_of_day: int,
    assigned_vehicles: set[str],
) -> tuple[Candidate, str] | None:
    """Charge the closest capable vehicle, then retry the assignment."""
    operating_max = float(cfg.op("soc_policy.soc_operating_max"))

    best: tuple[Candidate, str] | None = None
    for vehicle in scenario.vehicles:
        if vehicle.vehicle_id in assigned_vehicles:
            continue
        battery = battery_params(vehicle.powertrain, cfg)
        target_soc, _reasons = mission_target_soc(
            battery,
            vehicle.soc,
            _estimate_mission_energy(scenario, vehicle, request, cfg, model),
            float(cfg.opt("charging.mission_uncertainty_reserve")),
            float(cfg.opt("charging.max_target_soc")),
            operating_max,
        )
        if target_soc <= vehicle.soc + 1e-6:
            continue
        charger_pick = best_charger_for(
            scenario, vehicle.vehicle_id, target_soc, cfg, weights, hour_of_day
        )
        if charger_pick is None:
            continue
        charger_id, _plug_in_s = charger_pick
        projected_vehicle = _vehicle_with_soc(vehicle, min(target_soc, vehicle.charge_limit_soc))
        candidate = evaluate_candidate(
            scenario.model_copy(update={"vehicles": _swap_vehicle(scenario, projected_vehicle)}),
            projected_vehicle,
            request,
            cfg=cfg,
            graph=model,
        )
        if candidate.feasible and (best is None or candidate.deadhead_km < best[0].deadhead_km):
            best = (candidate, charger_id)
    return best


def _estimate_mission_energy(
    scenario: Scenario,
    vehicle,
    request: TripRequest,
    cfg: ConfigBundle,
    model: GraphModel,
) -> float:
    """Mission energy via the candidate physics at the vehicle's current SOC."""
    probe = evaluate_candidate(scenario, vehicle, request, cfg=cfg, graph=model)
    return max(probe.route_energy_kwh, 0.0)


def _vehicle_with_soc(vehicle, soc: float):
    return vehicle.model_copy(update={"soc": round(min(soc, 1.0), 4)})


def _swap_vehicle(scenario: Scenario, vehicle):
    return [
        vehicle if existing.vehicle_id == vehicle.vehicle_id else existing
        for existing in scenario.vehicles
    ]
