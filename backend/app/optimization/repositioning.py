"""Proactive repositioning of idle vehicles toward forecast demand.

A move is recommended only when the expected value of reduced unserved demand
exceeds the deadhead energy, time, cost and risk by the configured margin.
The optimizer (and the baseline) may always choose to keep the vehicle where
it is.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.config.loader import ConfigBundle, get_config
from app.demand.demand_model import ZoneForecast, zone_demand_forecast
from app.optimization.dispatch import CostWeights, cost_weights
from app.schemas.common import VehicleAvailability
from app.schemas.decision import RepositioningAction
from app.schemas.reason_codes import ReasonCode
from app.schemas.scenario import DemandRecord, Scenario, TripRequest
from app.simulation.city_graph import GraphModel, RoutePlanningError
from app.simulation.model_config import battery_params, vehicle_model_params


@dataclass
class RepositioningPlan:
    actions: list[RepositioningAction] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)
    forecasts: dict[str, ZoneForecast] = field(default_factory=dict)
    reason_codes: list[str] = field(default_factory=list)


def _deadhead_cost_inr(
    km: float,
    energy_kwh: float,
    time_min: float,
    weights: CostWeights,
    energy_price_inr_per_kwh: float,
    pickup_time_weight_inr_per_min: float,
) -> float:
    return (
        km * weights.deadhead_inr_per_km
        + energy_kwh * energy_price_inr_per_kwh
        + time_min * pickup_time_weight_inr_per_min
    )


def plan_repositioning(
    scenario: Scenario,
    demand_records: list[DemandRecord],
    *,
    assigned_vehicle_ids: set[str] | None = None,
    now_s: float = 0.0,
    cfg: ConfigBundle | None = None,
) -> RepositioningPlan:
    """Forecast-driven moves for idle vehicles; may return no moves at all."""
    c = cfg or get_config()
    plan = RepositioningPlan()
    if not bool(c.opt("repositioning.enabled")):
        plan.reason_codes.append(ReasonCode.REPOSITION_SKIPPED_DISABLED.value)
        return plan

    weights = cost_weights(c)
    model = GraphModel(scenario.graph, min_speed_factor=float(c.op("traffic.min_speed_factor")))
    horizon_min = float(c.opt("repositioning.forecast_horizon_min"))
    threshold = float(c.opt("repositioning.benefit_margin_threshold"))
    max_deadhead_km = float(c.opt("repositioning.max_deadhead_km"))
    value_per_request = float(c.opt("repositioning.unserved_risk_inr_per_request"))
    pickup_time_weight = float(c.opt("dispatch_weights.pickup_time_inr_per_min"))
    energy_price = float(c.opt("dispatch_weights.energy_inr_per_kwh"))
    hour = int(
        (float(c.demo_v("world_defaults.horizon_start_hour")) + now_s / 3600.0) % 24.0
    )
    assigned = assigned_vehicle_ids or set()
    urban_zones = [z.zone_id for z in model.zones.values() if z.kind == "urban"]
    context = scenario.demand_context

    forecasts = {
        zone_id: zone_demand_forecast(
            demand_records, zone_id, hour, horizon_min / 60.0, context, c
        )
        for zone_id in urban_zones
    }
    plan.forecasts = forecasts

    for vehicle in scenario.vehicles:
        if vehicle.availability != VehicleAvailability.AVAILABLE:
            plan.skipped[vehicle.vehicle_id] = ReasonCode.VEHICLE_UNAVAILABLE.value
            continue
        if vehicle.vehicle_id in assigned:
            plan.skipped[vehicle.vehicle_id] = ReasonCode.VEHICLE_UNAVAILABLE.value
            continue

        best: tuple[float, RepositioningAction] | None = None
        for zone_id in urban_zones:
            if zone_id == vehicle.current_zone_id:
                continue
            forecast = forecasts[zone_id]
            if forecast.expected_requests <= 0.0:
                continue
            try:
                path = model.shortest_path(
                    vehicle.current_zone_id, zone_id, scenario.environment.traffic_factor
                )
            except RoutePlanningError:
                continue
            if path.distance_km > max_deadhead_km:
                continue
            energy_kwh = _deadhead_energy_kwh(scenario, vehicle, path, c)
            time_min = path.travel_time_min
            cost_inr = _deadhead_cost_inr(
                path.distance_km, energy_kwh, time_min, weights, energy_price, pickup_time_weight
            )
            benefit_inr = forecast.expected_requests * forecast.confidence * value_per_request
            if cost_inr <= 0.0:
                continue
            if benefit_inr < threshold * cost_inr:
                continue
            action = RepositioningAction(
                vehicle_id=vehicle.vehicle_id,
                from_zone_id=vehicle.current_zone_id,
                to_zone_id=zone_id,
                deadhead_km=round(path.distance_km, 3),
                deadhead_energy_kwh=round(energy_kwh, 3),
                deadhead_time_min=round(time_min, 2),
                forecast_confidence=forecast.confidence,
                expected_benefit_inr=round(benefit_inr, 2),
                cost_inr=round(cost_inr, 2),
                reason_codes=[ReasonCode.REPOSITION_BENEFIT_POSITIVE.value],
            )
            score = benefit_inr - cost_inr
            if best is None or score > best[0]:
                best = (score, action)

        if best is None:
            plan.skipped[vehicle.vehicle_id] = ReasonCode.REPOSITION_SKIPPED_NO_BENEFIT.value
        else:
            plan.actions.append(best[1])

    if plan.actions:
        plan.reason_codes.append(ReasonCode.REPOSITION_BENEFIT_POSITIVE.value)
    elif not plan.skipped:
        plan.reason_codes.append(ReasonCode.REPOSITION_SKIPPED_NO_BENEFIT.value)
    return plan


def _deadhead_energy_kwh(
    scenario: Scenario,
    vehicle,
    path,
    cfg: ConfigBundle,
) -> float:
    """Empty-leg energy via the same physics the dispatcher uses."""
    from app.optimization.dispatch import candidate_numerics
    from app.simulation.model_config import hvac_params, numerics_params
    from app.simulation.route_energy import TripInputs, build_segments, simulate_route

    model = GraphModel(scenario.graph, min_speed_factor=float(cfg.op("traffic.min_speed_factor")))
    segments = build_segments(
        [model.edges[e] for e in path.edge_ids], scenario.environment.traffic_factor
    )
    result = simulate_route(
        segments,
        vehicle_model_params(cfg),
        hvac_params(cfg),
        battery_params(vehicle.powertrain, cfg),
        candidate_numerics(cfg),
        TripInputs(
            vehicle_id=vehicle.vehicle_id,
            route_id=f"reposition-{vehicle.vehicle_id}",
            departure_soc=vehicle.soc,
            soh=vehicle.soh,
            ambient_temp_c=scenario.environment.ambient_temp_c,
            mode=vehicle.cabin_mode,
            occupied_zones=0,
            passenger_count=0,
            luggage_mass_kg=0.0,
            traffic_factor=scenario.environment.traffic_factor,
            soc_reserve_min=float(cfg.op("soc_policy.soc_reserve_min")),
            soc_operating_min=float(cfg.op("soc_policy.soc_operating_min")),
            config_hash=cfg.hash,
        ),
    )
    return result.route_energy_kwh
