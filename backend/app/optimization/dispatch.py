"""Dispatch optimizer: candidate costs, scenario losses, and the CVaR MILP.

Assignments are binary decisions over (vehicle, request) pairs. Hard
constraints (capacity, availability, route feasibility, SOC reserve) remove
infeasible pairs before optimization rather than penalizing them. The
objective minimizes expected plan loss plus lambda * CVaR_alpha of the loss
across a batch of traffic-uncertainty scenarios; loss uses the fixed
convention loss = -profit (revenue reduces loss).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace as dc_replace

import numpy as np

from app.config.loader import ConfigBundle, get_config
from app.optimization import risk as risk_mod
from app.optimization.solver_adapter import LinearRow, MilpProblem, solve_milp
from app.schemas.common import CabinMode, ConstraintStatus, SolverStatus, VehicleAvailability
from app.schemas.decision import RiskMetrics, SolverReport
from app.schemas.reason_codes import ReasonCode
from app.schemas.scenario import Scenario, TripRequest
from app.schemas.vehicle import RouteResult, VehicleState

from app.simulation.battery import wear_per_kwh
from app.simulation.city_graph import GraphModel, RoutePlanningError
from app.simulation.model_config import (
    battery_params,
    hvac_params,
    mode_layouts,
    numerics_params,
    vehicle_model_params,
)
from app.simulation.route_energy import TripInputs, build_segments, simulate_route

INFINITY = 1e9


@dataclass(frozen=True)
class CostWeights:
    """Monetary weights of the assignment objective, all from configuration."""

    pickup_time_inr_per_min: float
    deadhead_inr_per_km: float
    late_pickup_penalty_inr: float
    unserved_penalty_inr: float
    energy_inr_per_kwh: float
    service_time_weight_inr_per_min: float
    battery_replacement_inr_per_kwh: float
    average_service_time_s: float


def cost_weights(cfg: ConfigBundle | None = None) -> CostWeights:
    c = cfg or get_config()
    return CostWeights(
        pickup_time_inr_per_min=float(c.opt("dispatch_weights.pickup_time_inr_per_min")),
        deadhead_inr_per_km=float(c.opt("dispatch_weights.deadhead_inr_per_km")),
        late_pickup_penalty_inr=float(c.opt("dispatch_weights.late_pickup_penalty_inr")),
        unserved_penalty_inr=float(c.opt("dispatch_weights.unserved_penalty_inr")),
        energy_inr_per_kwh=float(c.opt("dispatch_weights.energy_inr_per_kwh")),
        service_time_weight_inr_per_min=float(c.opt("dispatch_weights.service_time_weight_inr_per_min")),
        battery_replacement_inr_per_kwh=float(c.eco("capital.battery_replacement_inr_per_kwh")),
        average_service_time_s=float(c.op("service.average_service_time_s")),
    )


def candidate_numerics(cfg: ConfigBundle | None = None):
    """Physics parameters with the coarser screening step of the optimizer."""
    c = cfg or get_config()
    return dc_replace(
        numerics_params(c), integration_dt_s=float(c.opt("solver.candidate_dt_s"))
    )


@dataclass(frozen=True)
class Candidate:
    """One (vehicle, request) pairing with its recomputed physics and costs."""

    request_id: str
    vehicle_id: str
    deadhead_km: float
    deadhead_time_min: float
    pickup_eta_s: float
    trip_distance_km: float
    trip_time_min: float
    route_energy_kwh: float
    departure_soc: float
    arrival_soc: float
    wear_cost_inr: float
    feasible: bool
    checks: dict[str, str] = field(default_factory=dict)
    reason_codes: tuple[str, ...] = ()
    route_result: RouteResult | None = None

    @property
    def service_end_s(self) -> float:
        return self.pickup_eta_s + self.trip_time_min * 60.0

    def overlaps(self, other: "Candidate") -> bool:
        return not (
            self.service_end_s < other.pickup_eta_s
            or other.service_end_s < self.pickup_eta_s
        )

    def loss_inr(self, weights: CostWeights, request: TripRequest) -> float:
        """Plan loss of serving this request with this vehicle (lower is better)."""
        pickup_wait_min = max(0.0, self.pickup_eta_s - request.pickup_window_start_s) / 60.0
        trip_minutes = self.trip_time_min + weights.average_service_time_s / 60.0
        late = (
            weights.late_pickup_penalty_inr
            if self.pickup_eta_s > request.pickup_window_end_s
            else 0.0
        )
        cost = (
            self.deadhead_km * weights.deadhead_inr_per_km
            + pickup_wait_min * weights.pickup_time_inr_per_min
            + self.route_energy_kwh * weights.energy_inr_per_kwh
            + self.wear_cost_inr
            + trip_minutes * weights.service_time_weight_inr_per_min
            + late
        )
        fare = request.expected_fare_inr
        return cost - fare


def evaluate_candidate(
    scenario: Scenario,
    vehicle: VehicleState,
    request: TripRequest,
    *,
    cfg: ConfigBundle | None = None,
    graph: GraphModel | None = None,
    numerics=None,
) -> Candidate:
    """Recompute deadhead and trip physics for one pairing from the scenario."""
    c = cfg or get_config()
    model = graph or GraphModel(
        scenario.graph, min_speed_factor=float(c.op("traffic.min_speed_factor"))
    )
    traffic = scenario.environment.traffic_factor
    layouts = mode_layouts(c)
    layout = layouts[request.service_class]

    capacity_checks: dict[str, str] = {}
    reasons: list[str] = []
    params = vehicle_model_params(c)
    if request.passenger_count > layout.max_passengers:
        capacity_checks["passenger_capacity"] = ConstraintStatus.FAIL.value
        reasons.append(ReasonCode.PASSENGER_CAPACITY_EXCEEDED.value)
    else:
        capacity_checks["passenger_capacity"] = ConstraintStatus.PASS.value
    if request.luggage_units > layout.max_luggage_units:
        capacity_checks["luggage_capacity"] = ConstraintStatus.FAIL.value
        reasons.append(ReasonCode.LUGGAGE_CAPACITY_EXCEEDED.value)
    else:
        capacity_checks["luggage_capacity"] = ConstraintStatus.PASS.value
    if any(name.value not in layout.luggage_class_allowed for name in request.luggage_classes):
        capacity_checks["luggage_class"] = ConstraintStatus.FAIL.value
        reasons.append(ReasonCode.LUGGAGE_CAPACITY_EXCEEDED.value)
    else:
        capacity_checks["luggage_class"] = ConstraintStatus.PASS.value
    payload = (
        request.passenger_count * params.passenger_mass_kg
        + request.luggage_mass_kg
        + params.other_payload_kg
    )
    if payload > params.max_payload_kg:
        capacity_checks["payload"] = ConstraintStatus.FAIL.value
        reasons.append(ReasonCode.PAYLOAD_EXCEEDED.value)
    else:
        capacity_checks["payload"] = ConstraintStatus.PASS.value

    availability_checks: dict[str, str] = {}
    if vehicle.availability in (VehicleAvailability.OFFLINE, VehicleAvailability.MAINTENANCE):
        availability_checks["availability"] = ConstraintStatus.FAIL.value
        reasons.append(ReasonCode.VEHICLE_UNAVAILABLE.value)
    elif vehicle.next_available_time_s > request.pickup_window_end_s:
        availability_checks["availability"] = ConstraintStatus.FAIL.value
        reasons.append(ReasonCode.VEHICLE_UNAVAILABLE.value)
    else:
        availability_checks["availability"] = ConstraintStatus.PASS.value

    def failure(checks: dict[str, str]) -> Candidate:
        return Candidate(
            request_id=request.request_id,
            vehicle_id=vehicle.vehicle_id,
            deadhead_km=0.0,
            deadhead_time_min=0.0,
            pickup_eta_s=vehicle.next_available_time_s,
            trip_distance_km=0.0,
            trip_time_min=0.0,
            route_energy_kwh=0.0,
            departure_soc=vehicle.soc,
            arrival_soc=vehicle.soc,
            wear_cost_inr=0.0,
            feasible=False,
            checks=checks,
            reason_codes=tuple(dict.fromkeys(reasons)),
        )

    all_checks = {**capacity_checks, **availability_checks}
    if any(status == ConstraintStatus.FAIL.value for status in all_checks.values()):
        return failure(all_checks)

    try:
        deadhead = model.shortest_path(vehicle.current_zone_id, request.pickup_zone_id, traffic)
        trip = model.shortest_path(request.pickup_zone_id, request.dropoff_zone_id, traffic)
    except RoutePlanningError as exc:
        reasons.append(exc.reason_code.value)
        return failure({**all_checks, "route_exists": ConstraintStatus.FAIL.value})

    battery = battery_params(vehicle.powertrain, c)
    numerics_used = numerics or candidate_numerics(c)
    reserve = float(c.op("soc_policy.soc_reserve_min"))
    operating_min = float(c.op("soc_policy.soc_operating_min"))

    deadhead_result = simulate_route(
        build_segments([model.edges[e] for e in deadhead.edge_ids], traffic),
        params,
        hvac_params(c),
        battery,
        numerics_used,
        TripInputs(
            vehicle_id=vehicle.vehicle_id,
            route_id=f"{request.request_id}-deadhead",
            departure_soc=vehicle.soc,
            soh=vehicle.soh,
            ambient_temp_c=scenario.environment.ambient_temp_c,
            mode=CabinMode.COMMUTE,
            occupied_zones=0,
            passenger_count=0,
            luggage_mass_kg=0.0,
            traffic_factor=traffic,
            soc_reserve_min=reserve,
            soc_operating_min=operating_min,
            config_hash=c.hash,
        ),
    )
    trip_result = simulate_route(
        build_segments([model.edges[e] for e in trip.edge_ids], traffic),
        params,
        hvac_params(c),
        battery,
        numerics_used,
        TripInputs(
            vehicle_id=vehicle.vehicle_id,
            route_id=f"{request.request_id}-trip",
            departure_soc=deadhead_result.arrival_soc,
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
        ),
    )

    pickup_eta_s = vehicle.next_available_time_s + deadhead_result.travel_time_min * 60.0
    checks = dict(all_checks)
    checks["route_exists"] = ConstraintStatus.PASS.value
    if pickup_eta_s > request.pickup_window_end_s:
        checks["pickup_window"] = ConstraintStatus.FAIL.value
        reasons.append(ReasonCode.NO_FEASIBLE_VEHICLE_FOR_REQUEST.value)
    else:
        checks["pickup_window"] = ConstraintStatus.PASS.value
    if deadhead_result.constraint_checks.get("arrival_soc_reserve") != ConstraintStatus.PASS:
        checks["departure_soc_reserve"] = ConstraintStatus.FAIL.value
        reasons.append(ReasonCode.VEHICLE_BELOW_RESERVE_AT_START.value)
    else:
        checks["departure_soc_reserve"] = ConstraintStatus.PASS.value
    if trip_result.constraint_checks.get("arrival_soc_reserve") != ConstraintStatus.PASS:
        checks["arrival_soc_reserve"] = ConstraintStatus.FAIL.value
        reasons.append(ReasonCode.ARRIVAL_SOC_BELOW_RESERVE.value)
    else:
        checks["arrival_soc_reserve"] = ConstraintStatus.PASS.value
        reasons.append(ReasonCode.ARRIVAL_SOC_ABOVE_RESERVE.value)

    feasible = all(status == ConstraintStatus.PASS.value for status in checks.values())
    wear_rate = wear_per_kwh(
        battery,
        temperature_c=scenario.environment.ambient_temp_c,
        c_rate=1.0,
        depth_of_discharge=max(0.5, 1.0 - deadhead_result.arrival_soc),
        soh=vehicle.soh,
    )
    weights = cost_weights(c)
    wear_cost = wear_rate * trip_result.route_energy_kwh * weights.battery_replacement_inr_per_kwh

    return Candidate(
        request_id=request.request_id,
        vehicle_id=vehicle.vehicle_id,
        deadhead_km=round(deadhead.distance_km, 3),
        deadhead_time_min=round(deadhead_result.travel_time_min, 2),
        pickup_eta_s=round(pickup_eta_s, 1),
        trip_distance_km=round(trip.distance_km, 3),
        trip_time_min=round(trip_result.travel_time_min, 2),
        route_energy_kwh=round(trip_result.route_energy_kwh, 3),
        departure_soc=round(deadhead_result.arrival_soc, 4),
        arrival_soc=round(trip_result.arrival_soc, 4),
        wear_cost_inr=round(wear_cost, 2),
        feasible=feasible,
        checks=checks,
        reason_codes=tuple(dict.fromkeys(reasons)),
        route_result=trip_result,
    )


def scenario_traffic_draws(seed: int, count: int, sigma: float) -> np.ndarray:
    """Seeded traffic-factor draws for the risk objective."""
    if count <= 1:
        return np.array([1.0])
    rng = np.random.default_rng(seed)
    return np.clip(1.0 + rng.normal(0.0, sigma, count), 0.8, 2.5)


@dataclass
class DispatchOutcome:
    assignments: list[tuple[Candidate, TripRequest]]
    unserved: list[str]
    rejected: list[tuple[str, str, tuple[str, ...]]]
    solver: SolverReport
    risk: RiskMetrics | None
    reason_codes: list[str]
    status: SolverStatus
    scenario_losses_inr: np.ndarray | None = None


def solve_dispatch(
    scenario: Scenario,
    cfg: ConfigBundle | None = None,
    requests: list[TripRequest] | None = None,
) -> DispatchOutcome:
    """Risk-aware assignment solve for the scenario's pending requests."""
    c = cfg or get_config()
    pending = requests or [r for r in scenario.requests]
    weights = cost_weights(c)
    model = GraphModel(scenario.graph, min_speed_factor=float(c.op("traffic.min_speed_factor")))
    risk_cfg = risk_mod.risk_config(c)

    # Base-scenario screening; infeasible pairs are excluded, not penalized.
    candidates: list[Candidate] = []
    rejected: list[tuple[str, str, tuple[str, ...]]] = []
    request_by_id = {r.request_id: r for r in pending}
    for request in pending:
        for vehicle in scenario.vehicles:
            candidate = evaluate_candidate(
                scenario, vehicle, request, cfg=c, graph=model
            )
            if candidate.feasible:
                candidates.append(candidate)
            else:
                rejected.append(
                    (request.request_id, vehicle.vehicle_id, candidate.reason_codes)
                )

    if not candidates:
        return DispatchOutcome(
            assignments=[],
            unserved=[r.request_id for r in pending],
            rejected=rejected,
            solver=SolverReport(
                name="dispatch_milp",
                version="1.0.0",
                status=SolverStatus.NO_FEASIBLE_PLAN,
                runtime_ms=0.0,
                message="no feasible (vehicle, request) pairing exists",
            ),
            risk=None,
            reason_codes=[ReasonCode.SOLVER_NO_FEASIBLE_PLAN.value],
            status=SolverStatus.NO_FEASIBLE_PLAN,
        )

    # Scenario batch: traffic variation; a pair must stay feasible in every
    # scenario of the batch, so the risk term can never license a violation.
    scenario_count = risk_cfg.scenario_batch_size if risk_cfg.enabled else 1
    traffic_draws = scenario_traffic_draws(
        scenario.seed + 991, scenario_count, risk_cfg.traffic_sigma
    )
    feasible_matrix: list[list[float]] = []
    alive: list[int] = []
    for index, candidate in enumerate(candidates):
        request = request_by_id[candidate.request_id]
        losses: list[float] = []
        survives = True
        for traffic in traffic_draws:
            variant = scenario.model_copy(
                update={
                    "environment": scenario.environment.model_copy(
                        update={"traffic_factor": round(float(traffic), 3)}
                    )
                }
            )
            vehicle = scenario.vehicle(candidate.vehicle_id)
            variant_candidate = evaluate_candidate(
                variant, vehicle, request, cfg=c, graph=model
            )
            if not variant_candidate.feasible:
                survives = False
                break
            losses.append(variant_candidate.loss_inr(weights, request))
        if survives:
            feasible_matrix.append(losses)
            alive.append(index)

    candidates = [candidates[i] for i in alive]
    if not candidates:
        return DispatchOutcome(
            assignments=[],
            unserved=[r.request_id for r in pending],
            rejected=rejected,
            solver=SolverReport(
                name="dispatch_milp",
                version="1.0.0",
                status=SolverStatus.NO_FEASIBLE_PLAN,
                runtime_ms=0.0,
                message="every pairing failed in at least one risk scenario",
            ),
            risk=None,
            reason_codes=[ReasonCode.SOLVER_NO_FEASIBLE_PLAN.value],
            status=SolverStatus.NO_FEASIBLE_PLAN,
        )

    losses_by_decision = np.asarray(feasible_matrix, dtype=float).T
    scenario_total = losses_by_decision.shape[0]
    pair_count = len(candidates)
    request_ids = sorted(request_by_id)
    request_index = {rid: i for i, rid in enumerate(request_ids)}
    unserved_penalty = weights.unserved_penalty_inr

    # Decision vector: [pair x (D), unserved y (R), eta, u_0..u_{S-1}]
    decision_count = pair_count + len(request_ids)
    expanded = np.zeros((scenario_total, decision_count))
    expanded[:, :pair_count] = losses_by_decision
    expanded[:, pair_count:] = unserved_penalty

    cvar_objective = risk_mod.CvarObjective.build(
        expanded,
        decision_count=decision_count,
        constant_losses=np.zeros(scenario_total),
        alpha=risk_cfg.alpha,
        lambda_weight=risk_cfg.lambda_weight if risk_cfg.enabled else 0.0,
    )
    total_vars = decision_count + 1 + scenario_total

    lower = np.zeros(total_vars)
    upper = np.ones(total_vars)
    lower[decision_count] = -INFINITY
    upper[decision_count] = INFINITY
    upper[decision_count + 1 :] = INFINITY
    integrality = np.zeros(total_vars)
    integrality[: pair_count + len(request_ids)] = 1

    rows: list[LinearRow] = []

    def add_row(coefficients: np.ndarray, lb: float, ub: float) -> None:
        rows.append(LinearRow(coefficients=coefficients, lb=lb, ub=ub))

    # Each request is served exactly once or marked unserved.
    for request_id, r_index in request_index.items():
        row = np.zeros(total_vars)
        for pair_index, candidate in enumerate(candidates):
            if candidate.request_id == request_id:
                row[pair_index] = 1.0
        row[pair_count + r_index] = 1.0
        add_row(row, 1.0, 1.0)

    # One active assignment per vehicle in the horizon.
    vehicles_seen: dict[str, list[int]] = {}
    for pair_index, candidate in enumerate(candidates):
        vehicles_seen.setdefault(candidate.vehicle_id, []).append(pair_index)
    for pair_indices in vehicles_seen.values():
        row = np.zeros(total_vars)
        for pair_index in pair_indices:
            row[pair_index] = 1.0
        add_row(row, 0.0, 1.0)

    # Overlapping missions on the same vehicle are mutually exclusive.
    for pair_indices in vehicles_seen.values():
        for position, first in enumerate(pair_indices):
            for second in pair_indices[position + 1 :]:
                if candidates[first].overlaps(candidates[second]):
                    row = np.zeros(total_vars)
                    row[first] = 1.0
                    row[second] = 1.0
                    add_row(row, 0.0, 1.0)

    if risk_cfg.enabled:
        for coefficients, _bound in cvar_objective.cvar_rows(expanded):
            add_row(np.asarray(coefficients, dtype=float), -INFINITY, 0.0)

    problem = MilpProblem(
        objective=cvar_objective.objective,
        integrality=integrality,
        lower_bounds=lower,
        upper_bounds=upper,
        rows=rows,
    )
    solution = solve_milp(
        problem,
        timeout_s=float(c.opt("solver.timeout_s")),
        mip_gap=float(c.opt("solver.mip_relative_gap")),
        preferred=str(c.opt("solver.primary")),
        allow_fallback=str(c.opt("solver.fallback")) != str(c.opt("solver.primary")),
    )

    if not solution.solved:
        status = (
            SolverStatus.TIMEOUT
            if solution.status == "TIMEOUT"
            else SolverStatus.ERROR
            if solution.status == "ERROR"
            else SolverStatus.NO_FEASIBLE_PLAN
        )
        return DispatchOutcome(
            assignments=[],
            unserved=[r.request_id for r in pending],
            rejected=rejected,
            solver=SolverReport(
                name=solution.solver_name,
                version=solution.solver_version,
                status=status,
                runtime_ms=round(solution.runtime_s * 1000.0, 1),
                message=solution.message,
            ),
            risk=None,
            reason_codes=[ReasonCode.SOLVER_NO_FEASIBLE_PLAN.value],
            status=status,
        )

    x = solution.x
    assignments: list[tuple[Candidate, TripRequest]] = []
    chosen_pairs = [i for i in range(pair_count) if x[i] > 0.5]
    for pair_index in chosen_pairs:
        candidate = candidates[pair_index]
        assignments.append((candidate, request_by_id[candidate.request_id]))
    unserved = [
        request_ids[i]
        for i in range(len(request_ids))
        if x[pair_count + i] > 0.5
    ]

    plan_losses = np.zeros(scenario_total)
    for pair_index in chosen_pairs:
        plan_losses += losses_by_decision[:, pair_index]
    for request_id in unserved:
        plan_losses += unserved_penalty

    metrics = risk_mod.build_risk_metrics(plan_losses, risk_cfg)
    reason_codes = [ReasonCode.OPTIMIZER_SELECTED_LOWER_TOTAL_COST.value]
    if risk_cfg.enabled:
        reason_codes.append(
            ReasonCode.CVAR_RISK_ACTIVE.value
            if metrics.enabled
            else ReasonCode.CVAR_DISABLED.value
        )
    for request_id in unserved:
        reason_codes.append(f"{ReasonCode.NO_FEASIBLE_VEHICLE_FOR_REQUEST.value}:{request_id}")

    return DispatchOutcome(
        assignments=assignments,
        unserved=unserved,
        rejected=rejected,
        solver=SolverReport(
            name=solution.solver_name,
            version=solution.solver_version,
            status=SolverStatus(solution.status) if solution.status in SolverStatus._value2member_map_ else SolverStatus.FEASIBLE,
            runtime_ms=round(solution.runtime_s * 1000.0, 1),
            message=solution.message,
        ),
        risk=metrics,
        reason_codes=reason_codes,
        status=SolverStatus.FEASIBLE,
        scenario_losses_inr=plan_losses,
    )


def total_loss_inr(
    assignments: list[tuple[Candidate, TripRequest]],
    unserved: list[str],
    weights: CostWeights,
    unserved_penalty: float | None = None,
) -> float:
    """Plan loss of a concrete assignment set under one scenario's candidates."""
    penalty = weights.unserved_penalty_inr if unserved_penalty is None else unserved_penalty
    total = sum(candidate.loss_inr(weights, request) for candidate, request in assignments)
    return total + penalty * len(unserved)
