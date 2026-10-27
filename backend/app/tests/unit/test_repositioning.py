"""Repositioning planner: benefit threshold, deadhead limit, honest skipping."""

from __future__ import annotations

import pytest

from app.optimization.repositioning import plan_repositioning
from app.schemas.reason_codes import ReasonCode


def test_disabled_repositioning_reports_the_reason(demo_scenario, cfg, monkeypatch):
    parameter = cfg.optimizer._parameters["repositioning.enabled"]
    original = parameter.value
    try:
        parameter.value = False
        plan = plan_repositioning(demo_scenario, records_for(demo_scenario, cfg), cfg=cfg)
        assert plan.actions == []
        assert ReasonCode.REPOSITION_SKIPPED_DISABLED.value in plan.reason_codes
    finally:
        parameter.value = original


def records_for(scenario, cfg) -> list:
    from app.simulation.scenario_generator import demand_records_for as build

    return build(scenario, cfg)


def test_every_action_pays_for_itself(demo_scenario, cfg):
    plan = plan_repositioning(demo_scenario, records_for(demo_scenario, cfg), cfg=cfg)
    threshold = float(cfg.opt("repositioning.benefit_margin_threshold"))
    max_deadhead = float(cfg.opt("repositioning.max_deadhead_km"))
    for action in plan.actions:
        assert action.deadhead_km <= max_deadhead
        assert action.deadhead_km > 0.0
        assert action.deadhead_energy_kwh >= 0.0
        assert action.cost_inr > 0.0
        assert action.expected_benefit_inr >= threshold * action.cost_inr
        assert 0.0 <= action.forecast_confidence <= 1.0
        assert ReasonCode.REPOSITION_BENEFIT_POSITIVE.value in action.reason_codes


def test_skipped_vehicles_carry_a_reason(demo_scenario, cfg):
    plan = plan_repositioning(demo_scenario, records_for(demo_scenario, cfg), cfg=cfg)
    for vehicle_id, reason in plan.skipped.items():
        assert vehicle_id
        assert reason in {code.value for code in ReasonCode}


def test_assigned_and_unavailable_vehicles_are_never_moved(demo_scenario, cfg):
    busy = demo_scenario.vehicles[0].vehicle_id
    plan = plan_repositioning(
        demo_scenario, records_for(demo_scenario, cfg),
        assigned_vehicle_ids={busy}, cfg=cfg,
    )
    assert all(action.vehicle_id != busy for action in plan.actions)
    assert plan.skipped.get(busy) == ReasonCode.VEHICLE_UNAVAILABLE.value


def test_no_move_is_a_valid_outcome(demo_scenario, cfg):
    plan = plan_repositioning(demo_scenario, records_for(demo_scenario, cfg), now_s=3600.0, cfg=cfg)
    assert plan.reason_codes or plan.actions or plan.skipped
    assert set(plan.forecasts) == {
        z.zone_id for z in demo_scenario.graph.zones if z.kind == "urban"
    }


def test_planner_is_deterministic(demo_scenario, cfg):
    data = records_for(demo_scenario, cfg)
    first = plan_repositioning(demo_scenario, data, cfg=cfg)
    second = plan_repositioning(demo_scenario, data, cfg=cfg)
    keyed = lambda p: [(a.vehicle_id, a.to_zone_id) for a in p.actions]
    assert keyed(first) == keyed(second)


def test_forecast_is_available_for_urban_zones(demo_scenario, cfg):
    plan = plan_repositioning(demo_scenario, records_for(demo_scenario, cfg), cfg=cfg)
    urban = [z.zone_id for z in demo_scenario.graph.zones if z.kind == "urban"]
    assert set(plan.forecasts) == set(urban)
    for forecast in plan.forecasts.values():
        assert forecast.expected_requests >= 0.0
        assert 0.0 <= forecast.confidence <= 1.0