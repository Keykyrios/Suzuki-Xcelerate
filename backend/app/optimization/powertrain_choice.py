"""Powertrain recommendation: NEXUS_E versus NEXUS_X on one mission.

Both configurations run the same route physics. The battery-electric option
pays for charging exposure when the mission exceeds usable battery energy;
the range-extender option pays fuel, generator maintenance, extra capital
allocation and extra maintenance per kilometre. NEXUS_X is recommended only
when its avoided service/charging risk exceeds those extra costs by the
configured confidence margin. This is a simulation decision, not evidence
that a physical range-extender system exists.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config.loader import ConfigBundle, get_config
from app.economics.model import fare_for
from app.optimization.dispatch import candidate_numerics, cost_weights
from app.schemas.common import CabinMode, PowertrainType
from app.schemas.decision import PowertrainComparison
from app.schemas.reason_codes import ReasonCode
from app.schemas.scenario import Scenario
from app.simulation.battery import wear_per_kwh
from app.simulation.charging import ChargingEconomics, evaluate_charger
from app.simulation.city_graph import GraphModel, RoutePlanningError
from app.simulation.model_config import (
    battery_params,
    hvac_params,
    vehicle_model_params,
)
from app.simulation.route_energy import TripInputs, build_segments, simulate_route


@dataclass(frozen=True)
class MissionSpec:
    """The mission a powertrain decision is made for."""

    mission_id: str
    origin_zone: str
    destination_zone: str
    service_class: CabinMode
    passenger_count: int
    luggage_mass_kg: float
    departure_soc: float = 0.8
    soh: float = 1.0


@dataclass(frozen=True)
class PowertrainCosting:
    powertrain: PowertrainType
    route_energy_kwh: float
    travel_time_min: float
    distance_km: float
    feasible_on_battery: bool
    deficit_kwh: float
    energy_cost_inr: float
    charging_risk_inr: float
    fuel_cost_inr: float
    generator_maintenance_inr: float
    capital_daily_inr: float
    maintenance_per_km_inr: float

    @property
    def mission_cost_inr(self) -> float:
        """Risk-adjusted mission cost in constant rupees."""
        km_cost = self.distance_km * self.maintenance_per_km_inr
        return (
            self.energy_cost_inr
            + self.charging_risk_inr
            + self.fuel_cost_inr
            + self.generator_maintenance_inr
            + self.capital_daily_inr
            + km_cost
        )


def _simulate(
    scenario: Scenario, mission: MissionSpec, powertrain: PowertrainType, cfg: ConfigBundle
) -> tuple[float, float, float, bool]:
    """(energy_kwh, travel_min, distance_km, feasible) for the mission."""
    model = GraphModel(scenario.graph, min_speed_factor=float(cfg.op("traffic.min_speed_factor")))
    try:
        path = model.shortest_path(
            mission.origin_zone, mission.destination_zone, scenario.environment.traffic_factor
        )
    except RoutePlanningError:
        return 0.0, 0.0, 0.0, False
    battery = battery_params(powertrain, cfg)
    result = simulate_route(
        build_segments(
            [model.edges[e] for e in path.edge_ids], scenario.environment.traffic_factor
        ),
        vehicle_model_params(cfg),
        hvac_params(cfg),
        battery,
        candidate_numerics(cfg),
        TripInputs(
            vehicle_id=f"probe-{powertrain.value}",
            route_id=mission.mission_id,
            departure_soc=mission.departure_soc,
            soh=mission.soh,
            ambient_temp_c=scenario.environment.ambient_temp_c,
            mode=mission.service_class,
            occupied_zones=mission.passenger_count,
            passenger_count=mission.passenger_count,
            luggage_mass_kg=mission.luggage_mass_kg,
            traffic_factor=scenario.environment.traffic_factor,
            soc_reserve_min=float(cfg.op("soc_policy.soc_reserve_min")),
            soc_operating_min=float(cfg.op("soc_policy.soc_operating_min")),
            config_hash=cfg.hash,
        ),
    )
    return (
        result.route_energy_kwh,
        result.travel_time_min,
        path.distance_km,
        result.reserve_margin >= 0.0,
    )


def cost_powertrain(
    scenario: Scenario,
    mission: MissionSpec,
    powertrain: PowertrainType,
    cfg: ConfigBundle | None = None,
) -> PowertrainCosting:
    """Risk-adjusted mission costing for one configuration."""
    c = cfg or get_config()
    weights = cost_weights(c)
    battery = battery_params(powertrain, c)
    energy_kwh, travel_min, distance_km, _ = _simulate(scenario, mission, powertrain, c)

    usable_at_reserve = battery.usable_energy_kwh * mission.soh * max(
        0.0, mission.departure_soc - float(c.op("soc_policy.soc_reserve_min"))
    )
    deficit_kwh = max(0.0, energy_kwh - usable_at_reserve)
    feasible_on_battery = deficit_kwh <= 1e-6

    energy_cost = energy_kwh * weights.energy_inr_per_kwh
    charging_risk = 0.0
    fuel_cost = 0.0
    generator_maintenance = 0.0

    if powertrain == PowertrainType.NEXUS_E:
        if not feasible_on_battery:
            charging_risk = float(c.opt("powertrain.bev_charging_risk_inr_per_mission"))
            charging_cost, reachable = _en_route_charging_cost(
                scenario, mission, deficit_kwh, battery, c, weights
            )
            energy_cost += charging_cost
            if not reachable:
                # No charger can cover the deficit: the mission fare itself is
                # exposed to a charging-driven service failure.
                fare = fare_for(
                    mission.service_class, mission.passenger_count, distance_km, c
                ).total_inr
                failure_probability = float(
                    c.opt("powertrain.service_failure_probability_when_no_charger")
                )
                charging_risk += failure_probability * fare
    else:
        if deficit_kwh > 0.0:
            generator_kw = float(c.v("range_extender.generator_usable_output_kw"))
            efficiency = float(c.v("range_extender.generator_efficiency"))
            fuel_price = float(c.v("range_extender.generator_fuel_price_inr_per_litre"))
            fuel_kwh = deficit_kwh / max(efficiency, 1e-6)
            fuel_litres = fuel_kwh / max(float(c.v("range_extender.generator_fuel_energy_kwh_per_litre")), 1e-6)
            fuel_cost = fuel_litres * fuel_price
            hours = deficit_kwh / max(generator_kw * efficiency, 1e-6)
            generator_maintenance = hours * float(
                c.v("range_extender.generator_maintenance_inr_per_hour")
            )

    capital_daily = float(c.opt("powertrain.rex_capital_allocation_inr_per_day")) if (
        powertrain == PowertrainType.NEXUS_X
    ) else 0.0
    maintenance_per_km = (
        float(c.opt("powertrain.rex_maintenance_delta_inr_per_km"))
        if powertrain == PowertrainType.NEXUS_X
        else 0.0
    )

    return PowertrainCosting(
        powertrain=powertrain,
        route_energy_kwh=round(energy_kwh, 3),
        travel_time_min=round(travel_min, 2),
        distance_km=round(distance_km, 3),
        feasible_on_battery=feasible_on_battery,
        deficit_kwh=round(deficit_kwh, 3),
        energy_cost_inr=round(energy_cost, 2),
        charging_risk_inr=round(charging_risk, 2),
        fuel_cost_inr=round(fuel_cost, 2),
        generator_maintenance_inr=round(generator_maintenance, 2),
        capital_daily_inr=round(capital_daily, 2),
        maintenance_per_km_inr=round(maintenance_per_km, 3),
    )


def _en_route_charging_cost(
    scenario: Scenario,
    mission: MissionSpec,
    deficit_kwh: float,
    battery,
    cfg: ConfigBundle,
    weights,
) -> tuple[float, bool]:
    """Cheapest en-route charge to cover the deficit; (cost_inr, reachable)."""
    model = GraphModel(scenario.graph, min_speed_factor=float(cfg.op("traffic.min_speed_factor")))
    economics = ChargingEconomics(
        queue_cost_inr_per_min=float(cfg.opt("charging.queue_cost_inr_per_min")),
        detour_cost_inr_per_km=float(cfg.opt("charging.detour_cost_inr_per_km")),
        lost_productive_inr_per_min=float(cfg.opt("charging.lost_productive_revenue_inr_per_min")),
        energy_inr_per_kwh=weights.energy_inr_per_kwh,
        dc_surcharge_inr_per_kwh=float(cfg.eco("electricity.dc_fast_surcharge_inr_per_kwh")),
        wear_inr_per_kwh=wear_per_kwh(
            battery,
            temperature_c=scenario.environment.ambient_temp_c,
            c_rate=1.0,
            depth_of_discharge=0.5,
            soh=mission.soh,
        )
        * weights.battery_replacement_inr_per_kwh,
    )
    tariff_windows = (
        int(cfg.eco("electricity.peak_window_start_h")),
        int(cfg.eco("electricity.peak_window_end_h")),
        int(cfg.eco("electricity.offpeak_window_start_h")),
        int(cfg.eco("electricity.offpeak_window_end_h")),
    )
    tariff_bands = (
        float(cfg.eco("electricity.peak_inr_per_kwh")),
        float(cfg.eco("electricity.offpeak_inr_per_kwh")),
    )
    hour = int(cfg.demo_v("world_defaults.horizon_start_hour"))
    target_soc = mission.departure_soc + deficit_kwh / max(battery.usable_energy_kwh, 1e-6)
    try:
        direct = model.shortest_path(
            mission.origin_zone, mission.destination_zone, scenario.environment.traffic_factor
        )
        direct_time_min = direct.travel_time_min
    except RoutePlanningError:
        return deficit_kwh * weights.energy_inr_per_kwh, False
    best_cost = float("inf")
    for charger in scenario.chargers:
        try:
            to_charger = model.shortest_path(
                mission.origin_zone, charger.node_id, scenario.environment.traffic_factor
            )
            from_charger = model.shortest_path(
                charger.node_id, mission.destination_zone, scenario.environment.traffic_factor
            )
        except RoutePlanningError:
            continue
        detour_time_min = max(
            0.0, to_charger.travel_time_min + from_charger.travel_time_min - direct_time_min
        )
        estimate = evaluate_charger(
            charger=charger,
            battery=battery,
            economics=economics,
            soc_now=mission.departure_soc,
            target_soc=min(target_soc, 1.0),
            detour_km=max(
                0.0, to_charger.distance_km + from_charger.distance_km - direct.distance_km
            ),
            detour_time_min=detour_time_min,
            plug_count_in_use=charger.plugs_in_use,
            hour_of_day=hour,
            tariff_bands=tariff_bands,
            tariff_windows=tariff_windows,
        )
        if estimate.feasible:
            best_cost = min(best_cost, estimate.cost.total_inr)
    if best_cost == float("inf"):
        return deficit_kwh * weights.energy_inr_per_kwh, False
    return best_cost, True


def recommend_powertrain(
    scenario: Scenario,
    mission: MissionSpec,
    cfg: ConfigBundle | None = None,
) -> PowertrainComparison:
    """Compare both configurations and explain what moved the recommendation."""
    c = cfg or get_config()
    cost_e = cost_powertrain(scenario, mission, PowertrainType.NEXUS_E, c)
    cost_x = cost_powertrain(scenario, mission, PowertrainType.NEXUS_X, c)
    margin = float(c.opt("powertrain.recommendation_confidence_margin"))

    ev_cost = cost_e.mission_cost_inr
    xv_cost = cost_x.mission_cost_inr
    difference = xv_cost - ev_cost
    switching: list[str] = []
    reasons: list[str] = []

    if cost_e.charging_risk_inr > 0:
        reasons.append(
            f"NEXUS_E carries {cost_e.charging_risk_inr:.0f} INR of charging and "
            f"service-failure exposure for a {cost_e.deficit_kwh:.1f} kWh deficit"
        )
        switching.extend(["charging_risk_inr", "service_failure_risk_inr"])
    if cost_x.fuel_cost_inr > 0 or cost_x.generator_maintenance_inr > 0:
        reasons.append(
            f"NEXUS_X spends {cost_x.fuel_cost_inr:.0f} INR fuel and "
            f"{cost_x.generator_maintenance_inr:.0f} INR generator maintenance on the deficit"
        )
        switching.extend(["fuel_cost_inr", "generator_maintenance_inr"])
    if cost_x.capital_daily_inr > 0:
        switching.append("capital_daily_inr")
    if cost_x.maintenance_per_km_inr > 0:
        switching.append("maintenance_per_km_inr")

    recommend_x = (
        cost_e.charging_risk_inr > 0.0
        and difference < 0.0
        and abs(difference) > margin * max(ev_cost, 1e-9)
    )
    if recommend_x:
        recommended = PowertrainType.NEXUS_X
        alternative = PowertrainType.NEXUS_E
        reasons.append(
            f"avoided charging risk outweighs the extra NEXUS_X cost by {-difference:.0f} INR"
        )
        reasons.append(ReasonCode.POWERTRAIN_X_RISK_OFFSET.value)
    else:
        recommended = PowertrainType.NEXUS_E
        alternative = PowertrainType.NEXUS_X
        reasons.append(
            f"NEXUS_E mission cost {ev_cost:.0f} INR stays below NEXUS_X {xv_cost:.0f} INR "
            f"once the extra configuration costs are included"
        )
        reasons.append(ReasonCode.POWERTRAIN_X_NOT_JUSTIFIED.value)

    explanation = (
        f"Mission {mission.mission_id} ({mission.origin_zone} to {mission.destination_zone}, "
        f"{cost_e.distance_km:.1f} km, {cost_e.route_energy_kwh:.1f} kWh) is compared on both "
        f"configurations. {recommended.value} has the lower risk-adjusted mission cost "
        f"({min(ev_cost, xv_cost):.0f} INR versus {max(ev_cost, xv_cost):.0f} INR). "
        + " ".join(reasons[:-1])
        + ". Simulation decision only; it is not evidence about a physical vehicle."
    )

    return PowertrainComparison(
        mission_id=mission.mission_id,
        recommended=recommended,
        alternative=alternative,
        cost_nexus_e_inr=round(ev_cost, 2),
        cost_nexus_x_inr=round(xv_cost, 2),
        cost_difference_inr=round(difference, 2),
        switching_inputs=sorted(set(switching)),
        reason_codes=[r for r in reasons if r.startswith("POWERTRAIN_")],
        explanation=explanation,
    )
