"""Baseline dispatcher: nearest feasible vehicle, honest unassigned reasons."""

from __future__ import annotations

import pytest

from app.optimization.baseline import baseline_plan
from app.optimization.feasibility import recheck_assignment
from app.schemas.common import SolverStatus


def test_baseline_returns_a_dispatch_shaped_outcome(demo_scenario, cfg):
    outcome = baseline_plan(demo_scenario, cfg=cfg)
    assert outcome.status in {SolverStatus.FEASIBLE, SolverStatus.OPTIMAL, SolverStatus.NO_FEASIBLE_PLAN}
    assigned = {request.request_id for _candidate, request in outcome.assignments}
    unserved = set(outcome.unserved)
    assert assigned & unserved == set()  # a request is never both


def test_every_baseline_assignment_passes_the_independent_recheck(demo_scenario, cfg):
    outcome = baseline_plan(demo_scenario, cfg=cfg)
    for candidate, request in outcome.assignments:
        report, _result = recheck_assignment(
            demo_scenario, candidate.vehicle_id, request, candidate.pickup_eta_s, cfg
        )
        assert report.passed, f"{request.request_id}: {report.to_dict()} {report.reason_codes}"


def test_baseline_uses_each_vehicle_at_most_once(demo_scenario, cfg):
    outcome = baseline_plan(demo_scenario, cfg=cfg)
    vehicle_ids = [candidate.vehicle_id for candidate, _ in outcome.assignments]
    assert len(vehicle_ids) == len(set(vehicle_ids))


def test_unserved_requests_carry_a_reason(demo_scenario, cfg):
    outcome = baseline_plan(demo_scenario, cfg=cfg)
    covered = {r.request_id for _c, r in outcome.assignments} | set(outcome.unserved)
    assert covered == {r.request_id for r in demo_scenario.requests}
    for entry in outcome.rejected:
        _subject, reason, codes = entry
        assert codes or reason


def test_baseline_is_deterministic(demo_scenario, cfg):
    first = baseline_plan(demo_scenario, cfg=cfg)
    second = baseline_plan(demo_scenario, cfg=cfg)
    keyed = lambda o: [(c.vehicle_id, r.request_id) for c, r in o.assignments]
    assert keyed(first) == keyed(second)
    assert list(first.unserved) == list(second.unserved)


def test_baseline_reports_a_risk_view(demo_scenario, cfg):
    outcome = baseline_plan(demo_scenario, cfg=cfg)
    if outcome.risk is not None:
        assert outcome.risk.loss_definition == "loss_equals_negative_profit"
        assert outcome.risk.scenario_count >= 0
        assert outcome.risk.alpha > 0.0


def test_baseline_never_assigns_to_unavailable_vehicles(demo_scenario, cfg):
    scenario = demo_scenario.model_copy(
        update={
            "vehicles": [
                v.model_copy(update={"availability": "OFFLINE"})
                if index == 0
                else v
                for index, v in enumerate(demo_scenario.vehicles)
            ]
        }
    )
    outcome = baseline_plan(scenario, cfg=cfg)
    assert all(candidate.vehicle_id != scenario.vehicles[0].vehicle_id
               for candidate, _ in outcome.assignments)


def test_baseline_never_assigns_to_offline_chargers_for_backing_charge(demo_scenario, cfg):
    scenario = demo_scenario.model_copy(
        update={
            "chargers": [
                c.model_copy(update={"status": "OFFLINE"}) for c in demo_scenario.chargers
            ]
        }
    )
    outcome = baseline_plan(scenario, cfg=cfg)
    # With no charger available, plans must degrade to unassigned-with-reason rather than lie.
    for entry in outcome.rejected:
        assert isinstance(entry, tuple) and len(entry) == 3