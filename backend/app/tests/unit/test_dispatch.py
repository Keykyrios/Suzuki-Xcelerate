"""Dispatch optimizer: cost weights, candidate evaluation, solve, determinism."""

from __future__ import annotations

import math

import pytest

from app.optimization.dispatch import (
    cost_weights,
    evaluate_candidate,
    scenario_traffic_draws,
    solve_dispatch,
    total_loss_inr,
)
from app.schemas.common import SolverStatus


def test_cost_weights_come_from_configuration(cfg):
    weights = cost_weights(cfg)
    assert weights.pickup_time_inr_per_min == pytest.approx(
        float(cfg.opt("dispatch_weights.pickup_time_inr_per_min"))
    )
    assert weights.deadhead_inr_per_km == pytest.approx(
        float(cfg.opt("dispatch_weights.deadhead_inr_per_km"))
    )
    assert weights.energy_inr_per_kwh == pytest.approx(
        float(cfg.opt("dispatch_weights.energy_inr_per_kwh"))
    )
    assert weights.unserved_penalty_inr == pytest.approx(
        float(cfg.opt("dispatch_weights.unserved_penalty_inr"))
    )
    assert weights.energy_inr_per_kwh > 0.0
    assert weights.unserved_penalty_inr > 0.0
    assert weights.average_service_time_s > 0.0


def test_candidate_evaluation_is_physically_consistent(demo_scenario, cfg):
    vehicle = demo_scenario.vehicles[0]
    request = demo_scenario.requests[0]
    candidate = evaluate_candidate(demo_scenario, vehicle, request, cfg=cfg)
    assert candidate.request_id == request.request_id
    assert candidate.vehicle_id == vehicle.vehicle_id
    assert candidate.deadhead_km >= 0.0
    assert candidate.trip_distance_km > 0.0
    assert candidate.route_energy_kwh >= 0.0
    assert 0.0 <= candidate.departure_soc <= 1.0
    assert 0.0 <= candidate.arrival_soc <= 1.0
    assert candidate.arrival_soc <= candidate.departure_soc + 1e-6
    assert candidate.wear_cost_inr >= 0.0
    assert candidate.pickup_eta_s >= 0.0
    assert set(candidate.checks) <= {
        "passenger_capacity",
        "luggage_capacity",
        "luggage_class",
        "payload",
        "availability",
        "route_exists",
        "pickup_window",
        "departure_soc_reserve",
        "arrival_soc_reserve",
    }


def test_infeasible_pairings_are_marked_not_hidden(demo_scenario, cfg):
    request = max(demo_scenario.requests, key=lambda r: r.passenger_count)
    crowded = request.model_copy(update={"passenger_count": 8, "service_class": request.service_class})
    vehicle = demo_scenario.vehicles[0].model_copy(update={"soc": 0.05})
    candidate = evaluate_candidate(demo_scenario, vehicle, crowded, cfg=cfg)
    assert candidate.feasible is False
    assert candidate.checks
    assert candidate.reason_codes


def test_solve_dispatch_produces_consistent_partitions(demo_scenario, cfg):
    outcome = solve_dispatch(demo_scenario, cfg=cfg)
    solved_ids = {r.request_id for _c, r in outcome.assignments}
    assert solved_ids & set(outcome.unserved) == set()
    assert solved_ids | set(outcome.unserved) | {
        entry[0] for entry in outcome.rejected
    } >= {r.request_id for r in demo_scenario.requests}
    assert outcome.status in {status.value for status in SolverStatus} or isinstance(
        outcome.status, SolverStatus
    )


def test_every_solved_request_is_unique_by_vehicle(demo_scenario, cfg):
    outcome = solve_dispatch(demo_scenario, cfg=cfg)
    vehicle_ids = [c.vehicle_id for c, _r in outcome.assignments]
    request_ids = [r.request_id for _c, r in outcome.assignments]
    assert len(vehicle_ids) == len(set(vehicle_ids))
    assert len(request_ids) == len(set(request_ids))


def test_solve_is_deterministic_for_one_scenario(demo_scenario, cfg):
    first = solve_dispatch(demo_scenario, cfg=cfg)
    second = solve_dispatch(demo_scenario, cfg=cfg)
    keyed = lambda o: sorted((c.vehicle_id, r.request_id) for c, r in o.assignments)
    assert keyed(first) == keyed(second)
    assert first.status == second.status
    assert list(first.unserved) == list(second.unserved)


def test_solver_report_is_truthful(demo_scenario, cfg):
    outcome = solve_dispatch(demo_scenario, cfg=cfg)
    report = outcome.solver
    assert report.status in {
        "OPTIMAL", "FEASIBLE", "NO_FEASIBLE_PLAN", "TIMEOUT", "ERROR", "BASELINE_FALLBACK",
    }
    assert report.runtime_ms >= 0.0
    assert report.name
    assert report.version
    if report.fallback_used:
        assert report.status != "OPTIMAL"


def test_loss_function_penalises_unserved_requests(demo_scenario, cfg):
    outcome = solve_dispatch(demo_scenario, cfg=cfg)
    weights = cost_weights(cfg)
    loss_with_unserved = total_loss_inr(outcome.assignments, outcome.unserved, weights)
    loss_if_served = total_loss_inr(outcome.assignments, [], weights)
    assert math.isfinite(loss_with_unserved)
    if outcome.unserved:
        # loss = -profit, so the penalty appears as an exact additive term.
        assert loss_with_unserved - loss_if_served == pytest.approx(
            weights.unserved_penalty_inr * len(outcome.unserved), rel=1e-9
        )
        assert loss_with_unserved > loss_if_served


def test_risk_metrics_are_exposed_when_enabled(demo_scenario, cfg):
    outcome = solve_dispatch(demo_scenario, cfg=cfg)
    if outcome.risk is not None:
        assert outcome.risk.scenario_count > 0
        assert outcome.risk.loss_definition == "loss_equals_negative_profit"
        assert math.isfinite(outcome.risk.cvar_loss_inr)


def test_scenario_losses_are_reported_per_scenario(demo_scenario, cfg):
    outcome = solve_dispatch(demo_scenario, cfg=cfg)
    losses = outcome.scenario_losses_inr
    if losses is not None:
        count = int(cfg.opt("risk.scenario_batch_size"))
        assert losses is not None and len(losses) >= 1
        assert len(losses) in {1, count}


def test_traffic_draws_are_seeded_and_bounded():
    draws = scenario_traffic_draws(seed=3, count=25, sigma=0.1)
    same = scenario_traffic_draws(seed=3, count=25, sigma=0.1)
    assert len(draws) == 25
    assert (draws == same).all()
    assert all(0.5 <= value <= 2.0 for value in draws)
    assert len(scenario_traffic_draws(1, 0, 0.1)) == 1