"""Independent hard-feasibility checks for assignments, charging, and plans.

The checker deliberately recomputes paths and route energy from the scenario
instead of trusting numbers cached by the optimizer: an accepted plan is only
displayed as feasible after passing these checks. Every failure carries a
structured reason code.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.config.loader import ConfigBundle, get_config
from app.schemas.charger import ChargerState
from app.schemas.common import (
    CabinMode,
    ChargerStatus,
    ConstraintStatus,
    VehicleAvailability,
)
from app.schemas.reason_codes import ReasonCode
from app.schemas.scenario import Scenario, TripRequest
from app.schemas.vehicle import RouteResult, VehicleState
from app.simulation.battery import charge_acceptance_kw
from app.simulation.city_graph import GraphModel, PlannedPath
from app.simulation.model_config import (
    battery_params,
    hvac_params,
    mode_layouts,
    numerics_params,
    vehicle_model_params,
)
from app.simulation.route_energy import (
    RouteSegment,
    TripInputs,
    build_segments,
    simulate_route,
)


@dataclass
class FeasibilityReport:
    checks: dict[str, ConstraintStatus] = field(default_factory=dict)
    reason_codes: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(
            status == ConstraintStatus.PASS for status in self.checks.values()
        )

    def to_dict(self) -> dict[str, str]:
        return {k: v.value for k, v in self.checks.items()}

    def merge(self, other: "FeasibilityReport") -> "FeasibilityReport":
        self.checks.update(other.checks)
        self.reason_codes.extend(other.reason_codes)
        return self


def check_capacity(
    vehicle: VehicleState,
    request: TripRequest,
    cfg: ConfigBundle | None = None,
) -> FeasibilityReport:
    """Passenger, luggage, and payload capacity for one request/vehicle pair."""
    c = cfg or get_config()
    layouts = mode_layouts(c)
    report = FeasibilityReport()
    layout = layouts[request.service_class]

    if request.passenger_count > layout.max_passengers:
        report.checks["passenger_capacity"] = ConstraintStatus.FAIL
        report.reason_codes.append(
            f"{ReasonCode.PASSENGER_CAPACITY_EXCEEDED.value}:{request.service_class.value}"
        )
    elif request.passenger_count > int(c.v("package.passenger_capacity")):
        report.checks["passenger_capacity"] = ConstraintStatus.FAIL
        report.reason_codes.append(ReasonCode.PASSENGER_CAPACITY_EXCEEDED.value)
    else:
        report.checks["passenger_capacity"] = ConstraintStatus.PASS

    if request.luggage_units > layout.max_luggage_units:
        report.checks["luggage_capacity"] = ConstraintStatus.FAIL
        report.reason_codes.append(
            f"{ReasonCode.LUGGAGE_CAPACITY_EXCEEDED.value}:{request.service_class.value}"
        )
    else:
        report.checks["luggage_capacity"] = ConstraintStatus.PASS

    unknown_classes = [
        name
        for name in request.luggage_classes
        if name.value not in layout.luggage_class_allowed
    ]
    if unknown_classes:
        report.checks["luggage_class"] = ConstraintStatus.FAIL
        report.reason_codes.append(ReasonCode.LUGGAGE_CAPACITY_EXCEEDED.value)
    else:
        report.checks["luggage_class"] = ConstraintStatus.PASS

    params = vehicle_model_params(c)
    payload_kg = (
        request.passenger_count * params.passenger_mass_kg
        + request.luggage_mass_kg
        + params.other_payload_kg
    )
    if payload_kg > params.max_payload_kg:
        report.checks["payload"] = ConstraintStatus.FAIL
        report.reason_codes.append(ReasonCode.PAYLOAD_EXCEEDED.value)
    else:
        report.checks["payload"] = ConstraintStatus.PASS
    return report


def check_availability(
    vehicle: VehicleState,
    request: TripRequest,
    pickup_eta_s: float,
) -> FeasibilityReport:
    """Vehicle availability and pickup-window feasibility."""
    report = FeasibilityReport()
    if vehicle.availability in (VehicleAvailability.OFFLINE, VehicleAvailability.MAINTENANCE):
        report.checks["availability"] = ConstraintStatus.FAIL
        report.reason_codes.append(ReasonCode.VEHICLE_UNAVAILABLE.value)
    elif vehicle.next_available_time_s > request.pickup_window_end_s:
        report.checks["availability"] = ConstraintStatus.FAIL
        report.reason_codes.append(ReasonCode.VEHICLE_UNAVAILABLE.value)
    else:
        report.checks["availability"] = ConstraintStatus.PASS

    if pickup_eta_s > request.pickup_window_end_s:
        report.checks["pickup_window"] = ConstraintStatus.FAIL
        report.reason_codes.append(ReasonCode.NO_FEASIBLE_VEHICLE_FOR_REQUEST.value)
    elif pickup_eta_s < request.pickup_window_start_s:
        report.checks["pickup_window"] = ConstraintStatus.PASS
        report.reason_codes.append("EARLY_ARRIVAL_WAIT_ACCEPTED")
    else:
        report.checks["pickup_window"] = ConstraintStatus.PASS
    return report


def evaluate_trip_energy(
    scenario: Scenario,
    vehicle: VehicleState,
    request: TripRequest,
    *,
    cfg: ConfigBundle | None = None,
) -> tuple[FeasibilityReport, RouteResult | None, PlannedPath | None, PlannedPath | None]:
    """Recompute deadhead + trip route energy from scratch and test the reserve."""
    c = cfg or get_config()
    model = GraphModel(scenario.graph, min_speed_factor=float(c.op("traffic.min_speed_factor")))
    report = FeasibilityReport()
    traffic = scenario.environment.traffic_factor
    try:
        deadhead = model.shortest_path(vehicle.current_zone_id, request.pickup_zone_id, traffic)
        trip = model.shortest_path(request.pickup_zone_id, request.dropoff_zone_id, traffic)
    except Exception as exc:  # noqa: BLE001 - route planning failure is a feasibility result
        report.checks["route_exists"] = ConstraintStatus.FAIL
        report.reason_codes.append(
            getattr(exc, "reason_code", ReasonCode.ROUTE_EDGE_MISSING).value
        )
        return report, None, None, None
    report.checks["route_exists"] = ConstraintStatus.PASS

    battery = battery_params(vehicle.powertrain, c)
    deadhead_segments = build_segments(
        [model.edges[e] for e in deadhead.edge_ids],
        traffic,
        float(c.op("traffic.min_speed_factor")),
    )
    trip_segments = build_segments(
        [model.edges[e] for e in trip.edge_ids],
        traffic,
        float(c.op("traffic.min_speed_factor")),
    )
    reserve = float(c.op("soc_policy.soc_reserve_min"))
    operating_min = float(c.op("soc_policy.soc_operating_min"))
    departure_soc_after_deadhead = _arrival_soc(
        deadhead_segments, vehicle, request, battery, c, reserve, operating_min
    )
    inputs = TripInputs(
        vehicle_id=vehicle.vehicle_id,
        route_id=f"{request.request_id}-trip",
        departure_soc=departure_soc_after_deadhead,
        soh=vehicle.soh,
        ambient_temp_c=scenario.environment.ambient_temp_c,
        mode=request.service_class,
        occupied_zones=request.passenger_count,
        passenger_count=request.passenger_count,
        luggage_mass_kg=request.luggage_mass_kg,
        traffic_factor=traffic,
        soc_reserve_min=reserve,
        soc_operating_min=operating_min,
        config_hash=c.hash,
    )
    result = simulate_route(
        trip_segments,
        vehicle_model_params(c),
        hvac_params(c),
        battery,
        numerics_params(c),
        inputs,
    )
    if result.constraint_checks.get("arrival_soc_reserve") != ConstraintStatus.PASS:
        report.checks["arrival_soc_reserve"] = ConstraintStatus.FAIL
        report.reason_codes.append(ReasonCode.ARRIVAL_SOC_BELOW_RESERVE.value)
    else:
        report.checks["arrival_soc_reserve"] = ConstraintStatus.PASS
        report.reason_codes.append(ReasonCode.ARRIVAL_SOC_ABOVE_RESERVE.value)
    return report, result, deadhead, trip


def _arrival_soc(
    segments: list[RouteSegment],
    vehicle: VehicleState,
    request: TripRequest,
    battery,
    cfg: ConfigBundle,
    reserve: float,
    operating_min: float,
) -> float:
    """SOC after the empty leg to the pickup zone."""
    inputs = TripInputs(
        vehicle_id=vehicle.vehicle_id,
        route_id=f"{request.request_id}-deadhead",
        departure_soc=vehicle.soc,
        soh=vehicle.soh,
        ambient_temp_c=cfg.demo_v("world_defaults.ambient_temp_c"),
        mode=CabinMode.COMMUTE,
        occupied_zones=0,
        passenger_count=0,
        luggage_mass_kg=0.0,
        traffic_factor=cfg.demo_v("world_defaults.traffic_factor"),
        soc_reserve_min=reserve,
        soc_operating_min=operating_min,
        config_hash=cfg.hash,
    )
    result = simulate_route(
        segments,
        vehicle_model_params(cfg),
        hvac_params(cfg),
        battery,
        numerics_params(cfg),
        inputs,
    )
    return result.arrival_soc


def recheck_assignment(
    scenario: Scenario,
    vehicle_id: str,
    request: TripRequest,
    pickup_eta_s: float,
    cfg: ConfigBundle | None = None,
) -> tuple[FeasibilityReport, RouteResult | None]:
    """Full independent recheck of one assignment; recomputes energy itself."""
    c = cfg or get_config()
    vehicle = scenario.vehicle(vehicle_id)
    report = FeasibilityReport()
    report.merge(check_capacity(vehicle, request, c))
    report.merge(check_availability(vehicle, request, pickup_eta_s))
    energy_report, result, deadhead, _trip = evaluate_trip_energy(
        scenario, vehicle, request, cfg=c
    )
    report.merge(energy_report)
    if deadhead is not None:
        pickup_time_s = deadhead.travel_time_min * 60.0 + vehicle.next_available_time_s
        window_report = check_availability(vehicle, request, pickup_time_s)
        report.checks["pickup_window"] = window_report.checks["pickup_window"]
    return report, result


def check_charger_usable(
    scenario: Scenario,
    charger: ChargerState,
    vehicle: VehicleState,
    target_soc: float,
) -> FeasibilityReport:
    """Charger online state, connector compatibility, and delivery capability."""
    report = FeasibilityReport()
    if charger.status != ChargerStatus.ONLINE:
        report.checks["charger_online"] = ConstraintStatus.FAIL
        report.reason_codes.append(
            f"{ReasonCode.CHARGER_OFFLINE.value}:{charger.charger_id}"
        )
        return report
    report.checks["charger_online"] = ConstraintStatus.PASS

    battery = battery_params(vehicle.powertrain)
    connector = charger.connectors[0] if charger.connectors else None
    if connector is None:
        report.checks["connector_compatible"] = ConstraintStatus.FAIL
        report.reason_codes.append(
            f"{ReasonCode.CHARGER_ENERGY_DELIVERY_CONSTRAINED.value}:{charger.charger_id}"
        )
        return report
    report.checks["connector_compatible"] = ConstraintStatus.PASS

    if target_soc <= vehicle.soc:
        report.checks["energy_delivery"] = ConstraintStatus.PASS
        return report
    power_kw = charge_acceptance_kw(battery, vehicle.soc, connector=connector)
    power_kw = min(power_kw, charger.rated_power_kw, charger.site_limit_kw / max(1, charger.plug_count))
    if power_kw <= 0.0:
        report.checks["energy_delivery"] = ConstraintStatus.FAIL
        report.reason_codes.append(
            f"{ReasonCode.CHARGER_ENERGY_DELIVERY_CONSTRAINED.value}:{charger.charger_id}"
        )
    else:
        report.checks["energy_delivery"] = ConstraintStatus.PASS
    return report


def check_plan(
    scenario: Scenario,
    assignments: list[tuple[str, str, float, TripRequest]],
    cfg: ConfigBundle | None = None,
) -> FeasibilityReport:
    """Aggregate recheck over a plan: (vehicle_id, request_id, pickup_eta, request)."""
    aggregate = FeasibilityReport()
    used_vehicles: dict[str, list[TripRequest]] = {}
    for vehicle_id, request_id, eta, request in assignments:
        report, _result = recheck_assignment(scenario, vehicle_id, request, eta, cfg)
        for name, status in report.checks.items():
            aggregate.checks[f"{vehicle_id}:{request_id}:{name}"] = status
        aggregate.reason_codes.extend(report.reason_codes)
        used_vehicles.setdefault(vehicle_id, []).append(request)

    # A vehicle can serve overlapping requests only if their windows do not clash.
    for vehicle_id, requests in used_vehicles.items():
        for i, first in enumerate(requests):
            for second in requests[i + 1 :]:
                if _windows_overlap(first, second):
                    aggregate.checks[f"{vehicle_id}:double_booking"] = ConstraintStatus.FAIL
                    aggregate.reason_codes.append(
                        f"{ReasonCode.VEHICLE_UNAVAILABLE.value}:{vehicle_id}"
                    )
    if not any(k.endswith("double_booking") for k in aggregate.checks):
        aggregate.checks["no_double_booking"] = ConstraintStatus.PASS
    return aggregate



def _windows_overlap(first: TripRequest, second: TripRequest) -> bool:
    return not (
        first.pickup_window_end_s < second.pickup_window_start_s
        or second.pickup_window_end_s < first.pickup_window_start_s
    )
