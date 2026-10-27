"""Optimization orchestration: challenger versus baseline on identical inputs.

Builds the risk-aware plan, re-checks it with the independent feasibility
checker at display-grade physics, computes paired baseline metrics, and
assembles the full Decision - including the explanation card generated from
structured reason codes and numeric fields, never from free-form generation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace as dc_replace

from app.config.loader import ConfigBundle, get_config
from app.data_access.manifests import ManifestStore
from app.optimization.baseline import baseline_plan
from app.optimization.charging_schedule import MissionNeed, plan_charging
from app.optimization.dispatch import (
    DispatchOutcome,
    cost_weights,
    solve_dispatch,
)
from app.optimization.feasibility import recheck_assignment
from app.optimization.powertrain_choice import MissionSpec, recommend_powertrain
from app.optimization.repositioning import plan_repositioning
from app.schemas.common import (
    CabinMode,
    ConstraintStatus,
    EvidenceLabel,
    SolverStatus,
    utcnow,
)
from app.schemas.decision import (
    Assignment,
    Decision,
    DecisionPayload,
    FleetMetrics,
    RejectedAlternative,
    RiskMetrics,
    SolverReport,
)
from app.schemas.reason_codes import ReasonCode
from app.schemas.scenario import Scenario
from app.schemas.vehicle import VehicleState
from app.services.state_store import StateStore, get_state_store
from app.simulation.city_graph import GraphModel
from app.simulation.model_config import battery_params, hvac_params, vehicle_model_params
from app.simulation.route_energy import TripInputs, build_segments, simulate_route
from app.simulation.scenario_generator import demand_records_for

SOLVER_ADAPTER_VERSION = "1.0.0"


@dataclass
class OptimizationResult:
    decision: Decision
    baseline_outcome: DispatchOutcome
    optimized_outcome: DispatchOutcome


def _deadhead_energy_kwh(
    scenario: Scenario,
    vehicle: VehicleState,
    pickup_zone: str,
    cfg: ConfigBundle,
    model: GraphModel,
) -> float:
    """Empty-leg energy for metrics via the optimizer's screening physics."""
    from app.optimization.dispatch import candidate_numerics

    traffic = scenario.environment.traffic_factor
    try:
        path = model.shortest_path(vehicle.current_zone_id, pickup_zone, traffic)
    except Exception:  # noqa: BLE001 - metrics never fail on route lookup
        return 0.0
    result = simulate_route(
        build_segments([model.edges[e] for e in path.edge_ids], traffic),
        vehicle_model_params(cfg),
        hvac_params(cfg),
        battery_params(vehicle.powertrain, cfg),
        candidate_numerics(cfg),
        TripInputs(
            vehicle_id=vehicle.vehicle_id,
            route_id="metrics-deadhead",
            departure_soc=vehicle.soc,
            soh=vehicle.soh,
            ambient_temp_c=scenario.environment.ambient_temp_c,
            mode=CabinMode.COMMUTE,
            occupied_zones=0,
            passenger_count=0,
            luggage_mass_kg=0.0,
            traffic_factor=traffic,
            soc_reserve_min=float(cfg.op("soc_policy.soc_reserve_min")),
            soc_operating_min=float(cfg.op("soc_policy.soc_operating_min")),
            config_hash=cfg.hash,
        ),
    )
    return result.route_energy_kwh


def _fleet_metrics(
    scenario: Scenario,
    outcome: DispatchOutcome,
    charging_actions: list,
    cfg: ConfigBundle,
) -> FleetMetrics:
    """Paired benchmark metrics for one plan on the same scenario."""
    weights = cost_weights(cfg)
    model = GraphModel(scenario.graph, min_speed_factor=float(cfg.op("traffic.min_speed_factor")))

    deadhead_km = 0.0
    productive_km = 0.0
    total_energy = 0.0
    gross_revenue = 0.0
    wear_cost = 0.0
    min_soc = 1.0
    reserve_violations = 0

    for candidate, request in outcome.assignments:
        vehicle = scenario.vehicle(candidate.vehicle_id)
        deadhead_km += candidate.deadhead_km
        productive_km += candidate.trip_distance_km
        deadhead_energy = _deadhead_energy_kwh(
            scenario, vehicle, request.pickup_zone_id, cfg, model
        )
        total_energy += candidate.route_energy_kwh + deadhead_energy
        gross_revenue += request.expected_fare_inr
        wear_cost += candidate.wear_cost_inr
        min_soc = min(min_soc, candidate.arrival_soc)
        if candidate.arrival_soc < float(cfg.op("soc_policy.soc_reserve_min")):
            reserve_violations += 1

    for vehicle in scenario.vehicles:
        if vehicle.vehicle_id not in {c.vehicle_id for c, _ in outcome.assignments}:
            min_soc = min(min_soc, vehicle.soc)

    total_km = deadhead_km + productive_km
    charging_wait = sum(a.queue_wait_min for a in charging_actions)
    charging_detour = sum(a.detour_km for a in charging_actions)
    lost_productive = sum(a.cost.lost_productive_inr for a in charging_actions)
    energy_cost = total_energy * weights.energy_inr_per_kwh
    missed_service_cost = weights.unserved_penalty_inr * len(outcome.unserved)
    gross = gross_revenue
    contribution = (
        gross
        - energy_cost
        - wear_cost
        - lost_productive
        - missed_service_cost
        - deadhead_km * (float(cfg.eco("operating_costs.maintenance_reserve_inr_per_km"))
                         + float(cfg.eco("operating_costs.tyre_reserve_inr_per_km")))
    )

    return FleetMetrics(
        completed_requests=len(outcome.assignments),
        unserved_requests=len(outcome.unserved),
        deadhead_km=round(deadhead_km, 2),
        productive_km=round(productive_km, 2),
        total_km=round(total_km, 2),
        productive_km_ratio=round(productive_km / total_km, 4) if total_km > 0 else 0.0,
        total_energy_kwh=round(total_energy, 3),
        energy_per_vehicle_km_kwh=round(total_energy / total_km, 4) if total_km > 0 else 0.0,
        charging_wait_min=round(charging_wait, 2),
        charging_detour_km=round(charging_detour, 3),
        missed_productive_min=round(
            sum(a.charge_duration_min + a.queue_wait_min for a in charging_actions), 2
        ),
        min_soc=round(min_soc, 4),
        reserve_violations=reserve_violations,
        gross_revenue_inr=round(gross, 2),
        energy_cost_inr=round(energy_cost, 2),
        wear_cost_inr=round(wear_cost, 2),
        charging_opportunity_cost_inr=round(lost_productive, 2),
        missed_service_cost_inr=round(missed_service_cost, 2),
        contribution_inr=round(contribution, 2),
    )


def _risk_level(risk: RiskMetrics | None) -> str:
    if risk is None or not risk.enabled:
        return "LOW"
    if risk.cvar_loss_inr <= 0.0 and risk.expected_profit_inr > 0.0:
        return "LOW"
    if risk.cvar_loss_inr <= max(risk.expected_profit_inr, 0.0):
        return "MEDIUM"
    return "HIGH"


def _explanation_card(
    decision_payload: DecisionPayload,
    outcome: DispatchOutcome,
    baseline: DispatchOutcome,
    baseline_metrics: FleetMetrics | None,
    optimized_metrics: FleetMetrics | None,
    rejected: list[RejectedAlternative],
    risk: RiskMetrics | None,
    run_id: str,
    scenario_id: str,
    evidence_label: EvidenceLabel,
    stale: bool,
) -> str:
    """Render the operator explanation from codes and numbers only."""
    lines = [
        f"Selected plan: {len(outcome.assignments)} assignment(s), "
        f"{len(outcome.unserved)} unserved.",
    ]
    decisive = [code for code in outcome.reason_codes if not code.startswith("NO_FEASIBLE")][:3]
    if decisive:
        lines.append("Decisive reasons: " + ", ".join(decisive) + ".")
    if baseline_metrics is not None and optimized_metrics is not None:
        delta = optimized_metrics.contribution_inr - baseline_metrics.contribution_inr
        lines.append(
            f"Contribution versus baseline: {delta:+.0f} INR "
            f"({optimized_metrics.contribution_inr:.0f} vs "
            f"{baseline_metrics.contribution_inr:.0f})."
        )
    if rejected:
        first = rejected[0]
        lines.append(
            f"Rejected alternative: {first.kind} {first.subject_id} "
            f"({', '.join(first.reason_codes) or 'dominated on cost'})."
        )
    if risk is not None and risk.enabled:
        lines.append(
            f"Risk: expected profit {risk.expected_profit_inr:.0f} INR, "
            f"CVaR({risk.alpha}) of loss {risk.cvar_loss_inr:.0f} INR "
            f"over {risk.scenario_count} scenarios."
        )
    lines.append(
        f"Arrival SOC {decision_payload.arrival_soc_pct:.1f}% versus configured reserve "
        f"{decision_payload.reserve_soc_pct:.1f}%."
    )
    lines.append(f"Evidence: {evidence_label.value}; run {run_id}; scenario {scenario_id}.")
    if stale:
        lines.append("This recommendation is stale: the world changed after it was computed.")
    lines.append("Simulation estimates; not vehicle-calibrated.")
    return " ".join(lines)


class OptimizationService:
    """Solves the decision problem and assembles the operator-facing Decision."""

    def __init__(
        self,
        store: StateStore | None = None,
        cfg: ConfigBundle | None = None,
        manifests: ManifestStore | None = None,
    ) -> None:
        self.store = store or get_state_store()
        self.cfg = cfg or get_config()
        self.manifests = manifests or ManifestStore(self.cfg.root)
        self._explanations: dict[str, str] = {}

    def optimize(self, run_id: str | None = None) -> OptimizationResult:
        """Full optimization pass for the active scenario."""
        scenario = self.store.require_scenario()
        run_id = run_id or self.manifests.next_run_id()
        weights = cost_weights(self.cfg)

        try:
            optimized = solve_dispatch(scenario, self.cfg)
        except Exception as exc:  # noqa: BLE001 - a failed solve is reported, never faked
            optimized = DispatchOutcome(
                assignments=[],
                unserved=[request.request_id for request in scenario.requests],
                rejected=[],
                solver=SolverReport(
                    name=str(self.cfg.opt("solver.primary")),
                    version="unknown",
                    status=SolverStatus.ERROR,
                    runtime_ms=0.0,
                    fallback_used=False,
                    message=f"solver raised {type(exc).__name__}: {exc}",
                ),
                risk=None,
                reason_codes=[ReasonCode.SOLVER_ERROR_BASELINE_FALLBACK.value],
                status=SolverStatus.ERROR,
            )
        baseline = baseline_plan(scenario, self.cfg)

        fallback_policy = str(self.cfg.opt("solver.on_failure"))
        fallback_used = False
        solver_status = optimized.solver.status
        if optimized.status in (SolverStatus.TIMEOUT, SolverStatus.ERROR):
            if fallback_policy == "fall_back_to_baseline":
                optimized = dc_replace(
                    baseline,
                    solver=baseline.solver.model_copy(
                        update={
                            "fallback_used": True,
                            "status": SolverStatus.BASELINE_FALLBACK,
                            "message": optimized.solver.message
                            or "solver failed; baseline returned",
                        }
                    ),
                    status=SolverStatus.BASELINE_FALLBACK,
                    reason_codes=[
                        ReasonCode.SOLVER_TIMEOUT_BASELINE_FALLBACK.value
                        if solver_status == SolverStatus.TIMEOUT
                        else ReasonCode.SOLVER_ERROR_BASELINE_FALLBACK.value,
                        *baseline.reason_codes,
                    ],
                )
                fallback_used = True
            else:
                optimized = dc_replace(
                    optimized,
                    reason_codes=[
                        ReasonCode.SOLVER_FALLBACK_DISABLED.value,
                        *optimized.reason_codes,
                    ],
                )

        # Independent recheck at display-grade physics.
        assignments: list[Assignment] = []
        rejected: list[RejectedAlternative] = []
        recheck_passed = True
        for candidate, request in optimized.assignments:
            report, result = recheck_assignment(
                scenario, candidate.vehicle_id, request, candidate.pickup_eta_s, self.cfg
            )
            if not report.passed or result is None:
                recheck_passed = False
                rejected.append(
                    RejectedAlternative(
                        kind="assignment",
                        subject_id=f"{candidate.vehicle_id}:{request.request_id}",
                        reason_codes=[c for c in report.reason_codes],
                    )
                )
                continue
            charger_id = self._charger_for_vehicle(scenario, candidate.vehicle_id)
            assignments.append(
                Assignment(
                    request_id=request.request_id,
                    vehicle_id=candidate.vehicle_id,
                    pickup_zone_id=request.pickup_zone_id,
                    dropoff_zone_id=request.dropoff_zone_id,
                    pickup_eta_s=candidate.pickup_eta_s,
                    pickup_deadline_s=request.pickup_window_end_s,
                    deadhead_km=candidate.deadhead_km,
                    trip_distance_km=candidate.trip_distance_km,
                    route_energy_kwh=result.route_energy_kwh,
                    departure_soc=result.departure_soc,
                    arrival_soc=result.arrival_soc,
                    expected_fare_inr=request.expected_fare_inr,
                    contribution_inr=round(
                        request.expected_fare_inr
                        - result.route_energy_kwh * weights.energy_inr_per_kwh
                        - candidate.deadhead_km * weights.deadhead_inr_per_km
                        - candidate.wear_cost_inr,
                        2,
                    ),
                    selected_charger_id=charger_id,
                )
            )

        for request_id, vehicle_id, reasons in optimized.rejected:
            rejected.append(
                RejectedAlternative(
                    kind="assignment",
                    subject_id=f"{vehicle_id}:{request_id}",
                    reason_codes=list(reasons),
                )
            )

        # Charging, repositioning, and powertrain around the accepted plan.
        missions = [
            MissionNeed(
                vehicle_id=assignment.vehicle_id,
                start_s=assignment.pickup_eta_s,
                deadline_s=assignment.pickup_deadline_s,
                energy_kwh=assignment.route_energy_kwh
                * (1.0 + float(self.cfg.opt("charging.mission_uncertainty_reserve"))),
            )
            for assignment in assignments
        ]
        charging = plan_charging(scenario, missions, mode="day_ahead", cfg=self.cfg)
        records = demand_records_for(scenario, self.cfg)
        repositioning = plan_repositioning(
            scenario,
            records,
            assigned_vehicle_ids={a.vehicle_id for a in assignments},
            cfg=self.cfg,
        )
        powertrain = self._powertrain_for_plan(scenario, assignments)

        baseline_metrics = _fleet_metrics(scenario, baseline, [], self.cfg)
        optimized_metrics = _fleet_metrics(scenario, optimized, charging.actions, self.cfg)

        evidence = EvidenceLabel.MODEL_ESTIMATE
        risk = optimized.risk
        scenario_id = scenario.scenario_id
        focus_mode = self._focus_cabin_mode(scenario, assignments)
        focus_assignment = next(
            (a for a in assignments if a.vehicle_id == self._focus_vehicle_id(scenario)),
            assignments[0] if assignments else None,
        )
        reserve_pct = float(self.cfg.op("soc_policy.soc_reserve_min")) * 100.0
        payload = DecisionPayload(
            cabin_mode=focus_mode,
            powertrain_recommendation=powertrain.recommended,
            route_energy_kwh=focus_assignment.route_energy_kwh if focus_assignment else 0.0,
            energy_kwh_per_100km=(
                focus_assignment.route_energy_kwh / focus_assignment.trip_distance_km * 100.0
                if focus_assignment and focus_assignment.trip_distance_km > 0
                else 0.0
            ),
            departure_soc_pct=(
                focus_assignment.departure_soc * 100.0 if focus_assignment else 0.0
            ),
            arrival_soc_pct=focus_assignment.arrival_soc * 100.0 if focus_assignment else 0.0,
            reserve_soc_pct=reserve_pct,
            reserve_margin_pct=(
                (focus_assignment.arrival_soc * 100.0 - reserve_pct) if focus_assignment else 0.0
            ),
            selected_charger_id=charging.actions[0].charger_id if charging.actions else None,
            estimated_charge_wait_min=(
                charging.actions[0].queue_wait_min if charging.actions else None
            ),
            expected_contribution_inr=optimized_metrics.contribution_inr,
            risk_level=_risk_level(risk),
        )

        reason_codes = list(optimized.reason_codes)
        reason_codes.extend(repositioning.reason_codes)
        reason_codes.append(ReasonCode.CVAR_RISK_ACTIVE.value if risk and risk.enabled else ReasonCode.CVAR_DISABLED.value)
        if powertrain.reason_codes:
            reason_codes.extend(powertrain.reason_codes)
        reason_codes.append(ReasonCode.PLAN_RECHECKED_INDEPENDENTLY.value)

        stale = False
        decision = Decision(
            run_id=run_id,
            scenario_id=scenario_id,
            status=optimized.status,
            created_utc=utcnow(),
            decision=payload,
            assignments=assignments,
            charging_actions=charging.actions,
            repositioning_actions=repositioning.actions,
            powertrain=powertrain,
            constraints={
                "passenger_capacity": ConstraintStatus.PASS,
                "luggage_capacity": ConstraintStatus.PASS,
                "route_energy": ConstraintStatus.PASS,
                "arrival_soc_reserve": (
                    ConstraintStatus.PASS if recheck_passed else ConstraintStatus.FAIL
                ),
                "charger_available": (
                    ConstraintStatus.PASS
                    if all(a.effective_power_kw > 0.0 for a in charging.actions)
                    else ConstraintStatus.FAIL
                ),
            },
            reason_codes=reason_codes,
            rejected_alternatives=rejected[:20],
            risk=risk,
            solver=optimized.solver,
            baseline_metrics=baseline_metrics,
            optimized_metrics=optimized_metrics,
            warnings=[
                f"{vehicle_id}: {code}"
                for vehicle_id, code in list(charging.skipped.items())[:3]
            ],
            evidence_status=evidence,
            stale=stale,
            recheck_passed=recheck_passed,
        )
        explanation = _explanation_card(
            payload,
            optimized,
            baseline,
            baseline_metrics,
            optimized_metrics,
            rejected,
            risk,
            run_id,
            scenario_id,
            evidence,
            stale,
        )
        self._explanations[run_id] = explanation
        self.store.set_explanation(run_id, explanation)
        self.store.apply_decision(decision)
        self.manifests.record(
            run_id=run_id,
            seed=scenario.seed,
            scenario_id=scenario_id,
            solver_name=decision.solver.name,
            solver_status=decision.status.value,
            headline_metrics={
                "completed_requests": float(len(assignments)),
                "unserved_requests": float(len(optimized.unserved)),
                "contribution_inr": optimized_metrics.contribution_inr,
                "baseline_contribution_inr": baseline_metrics.contribution_inr,
                "cvar_loss_inr": risk.cvar_loss_inr if risk else 0.0,
                "deadhead_km": optimized_metrics.deadhead_km,
            },
            evidence_status=evidence.value,
            notes=("fallback to baseline" if fallback_used else ""),
            cfg=self.cfg,
        )
        return OptimizationResult(
            decision=decision,
            baseline_outcome=baseline,
            optimized_outcome=optimized,
        )

    def explanation_for(self, run_id: str) -> str:
        """Stored explanation text; shared through the canonical store so a fresh
        service instance can still explain a recorded run."""
        stored = self.store.get_explanation(run_id) or self._explanations.get(run_id)
        if stored is None:
            return "No explanation is stored for this run; re-run the optimizer."
        return stored

    def repositioning_for_current_state(self):
        scenario = self.store.require_scenario()
        records = demand_records_for(scenario, self.cfg)
        return plan_repositioning(scenario, records, cfg=self.cfg)

    # -- helpers -----------------------------------------------------------
    def _charger_for_vehicle(self, scenario: Scenario, vehicle_id: str) -> str | None:
        for charger in scenario.chargers:
            if charger.plugs_available > 0 and charger.is_online:
                return charger.charger_id
        return None

    def _focus_vehicle_id(self, scenario: Scenario) -> str:
        return str(self.cfg.demo_v("demo_script.focus_vehicle_id"))

    def _focus_cabin_mode(self, scenario: Scenario, assignments: list[Assignment]) -> CabinMode:
        cabin = self.store.cabin_states.get(self._focus_vehicle_id(scenario))
        if cabin is not None:
            return cabin.mode
        vehicle = next(
            (v for v in scenario.vehicles if v.vehicle_id == self._focus_vehicle_id(scenario)),
            scenario.vehicles[0],
        )
        return vehicle.cabin_mode

    def _powertrain_for_plan(self, scenario: Scenario, assignments: list[Assignment]):
        """Recommend for the longest accepted mission (or the demo mission)."""
        if assignments:
            longest = max(assignments, key=lambda a: a.trip_distance_km)
            request = scenario.request(longest.request_id)
            mission = MissionSpec(
                mission_id=f"{scenario.scenario_id}:{longest.request_id}",
                origin_zone=longest.pickup_zone_id,
                destination_zone=longest.dropoff_zone_id,
                service_class=request.service_class,
                passenger_count=request.passenger_count,
                luggage_mass_kg=request.luggage_mass_kg,
                departure_soc=longest.departure_soc,
            )
        else:
            mission = MissionSpec(
                mission_id=f"{scenario.scenario_id}:demo-intercity",
                origin_zone=str(self.cfg.demo_v("demo_script.intercity_origin_zone")),
                destination_zone=str(self.cfg.demo_v("demo_script.intercity_destination_zone")),
                service_class=CabinMode.INTERCITY_PREMIUM,
                passenger_count=4,
                luggage_mass_kg=40.0,
            )
        return recommend_powertrain(scenario, mission, self.cfg)
