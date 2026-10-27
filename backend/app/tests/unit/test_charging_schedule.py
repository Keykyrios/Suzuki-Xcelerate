"""Charging scheduler: day-ahead plan, rolling correction, mission targeting."""

from __future__ import annotations

import pytest

from app.optimization.charging_schedule import (
    DAY_AHEAD,
    ROLLING,
    MissionNeed,
    invalidate_plans,
    plan_charging,
)
from app.schemas.common import ChargerStatus
from app.schemas.reason_codes import ReasonCode


def test_unknown_mode_is_rejected(demo_scenario, cfg):
    with pytest.raises(ValueError):
        plan_charging(demo_scenario, [], mode="whenever", cfg=cfg)


def test_no_missions_means_no_actions(demo_scenario, cfg):
    result = plan_charging(demo_scenario, [], mode=DAY_AHEAD, cfg=cfg)
    assert result.actions == []
    assert result.mode == DAY_AHEAD


def test_vehicle_needing_charge_gets_a_plan(demo_scenario, cfg):
    vehicle = next(v for v in demo_scenario.vehicles if v.soc < 0.6)
    mission = MissionNeed(
        vehicle_id=vehicle.vehicle_id, start_s=0.0, deadline_s=8 * 3600.0, energy_kwh=18.0
    )
    result = plan_charging(demo_scenario, [mission], mode=DAY_AHEAD, now_s=0.0, cfg=cfg)
    assert result.actions, f"expected a plan, skipped={result.skipped}"
    action = result.actions[0]
    assert action.vehicle_id == vehicle.vehicle_id
    assert action.target_soc <= float(cfg.opt("charging.max_target_soc"))
    assert action.target_soc <= float(cfg.op("soc_policy.soc_operating_max"))
    assert action.energy_kwh > 0.0
    assert ReasonCode.CHARGE_TARGET_ONLY_MISSION_ENERGY.value in action.reason_codes
    assert ReasonCode.DAY_AHEAD_SCHEDULE_REPLANNED.value in result.reason_codes


def test_target_only_covers_mission_energy_plus_reserve(demo_scenario, cfg):
    vehicle = next(v for v in demo_scenario.vehicles if v.soc < 0.6)
    battery_kwh = float(
        cfg.v(
            "battery.bev_usable_energy_kwh"
            if vehicle.powertrain.value == "NEXUS_E"
            else "battery.rex_usable_energy_kwh"
        )
    )
    mission = MissionNeed(
        vehicle_id=vehicle.vehicle_id, start_s=0.0, deadline_s=6 * 3600.0, energy_kwh=10.0
    )
    result = plan_charging(demo_scenario, [mission], mode=DAY_AHEAD, now_s=0.0, cfg=cfg)
    assert result.actions
    added_energy = (result.actions[0].target_soc - vehicle.soc) * battery_kwh
    reserve = float(cfg.opt("charging.mission_uncertainty_reserve"))
    assert added_energy <= 10.0 * (1.0 + reserve) + 0.5


def test_vehicle_already_above_its_mission_needs_is_skipped(demo_scenario, cfg):
    vehicle = max(demo_scenario.vehicles, key=lambda v: v.soc)
    mission = MissionNeed(
        vehicle_id=vehicle.vehicle_id, start_s=0.0, deadline_s=3600.0, energy_kwh=0.0
    )
    result = plan_charging(demo_scenario, [mission], mode=DAY_AHEAD, now_s=0.0, cfg=cfg)
    assert result.actions == []
    assert vehicle.vehicle_id in result.skipped
    assert (
        result.skipped[vehicle.vehicle_id]
        == ReasonCode.CHARGE_TARGET_ONLY_MISSION_ENERGY.value
    )


def test_impossible_deadline_is_skipped_not_faked(demo_scenario, cfg):
    vehicle = next(v for v in demo_scenario.vehicles if v.soc < 0.7)
    mission = MissionNeed(
        vehicle_id=vehicle.vehicle_id, start_s=0.0, deadline_s=1.0, energy_kwh=40.0
    )
    result = plan_charging(demo_scenario, [mission], mode=DAY_AHEAD, now_s=0.0, cfg=cfg)
    assert result.actions == []
    assert vehicle.vehicle_id in result.skipped


def test_offline_charger_is_never_scheduled_on(demo_scenario, cfg):
    online_ids = {c.charger_id for c in demo_scenario.chargers}
    scenario = demo_scenario.model_copy(
        update={
            "chargers": [
                c.model_copy(update={"status": ChargerStatus.OFFLINE.value})
                for c in demo_scenario.chargers
            ]
        }
    )
    vehicle = next(v for v in demo_scenario.vehicles if v.soc < 0.6)
    mission = MissionNeed(
        vehicle_id=vehicle.vehicle_id, start_s=0.0, deadline_s=8 * 3600.0, energy_kwh=20.0
    )
    result = plan_charging(scenario, [mission], mode=DAY_AHEAD, now_s=0.0, cfg=cfg)
    assert all(action.charger_id in online_ids for action in result.actions)


def test_rolling_mode_shares_the_logic_and_labels_itself(demo_scenario, cfg):
    vehicle = next(v for v in demo_scenario.vehicles if v.soc < 0.6)
    mission = MissionNeed(
        vehicle_id=vehicle.vehicle_id, start_s=0.0, deadline_s=6 * 3600.0, energy_kwh=15.0
    )
    rolling = plan_charging(demo_scenario, [mission], mode=ROLLING, now_s=0.0, cfg=cfg)
    assert rolling.mode == ROLLING
    assert rolling.actions
    assert ReasonCode.ROLLING_CORRECTION_APPLIED.value in rolling.reason_codes


def test_rolling_window_is_shorter_than_the_day_ahead_horizon(cfg):
    assert float(cfg.opt("charging.rolling_lookahead_min")) < float(
        cfg.opt("charging.day_ahead_horizon_h")
    ) * 60.0
    assert float(cfg.opt("charging.day_ahead_slot_minutes")) in {5.0, 10.0, 15.0, 30.0}


def test_schedule_costs_are_the_five_term_breakdown(demo_scenario, cfg):
    vehicle = next(v for v in demo_scenario.vehicles if v.soc < 0.6)
    mission = MissionNeed(
        vehicle_id=vehicle.vehicle_id, start_s=0.0, deadline_s=6 * 3600.0, energy_kwh=20.0
    )
    result = plan_charging(demo_scenario, [mission], mode=DAY_AHEAD, now_s=0.0, cfg=cfg)
    assert result.actions
    cost = result.actions[0].cost
    total = cost.total_inr
    assert cost.electricity_inr > 0.0
    assert total > 0.0


def test_invalidated_plans_are_reported(demo_scenario):
    scenario = demo_scenario.model_copy(
        update={
            "chargers": [
                c.model_copy(update={"status": ChargerStatus.OFFLINE.value})
                for c in demo_scenario.chargers[:1]
            ]
            + list(demo_scenario.chargers[1:])
        }
    )
    offline_ids = [scenario.chargers[0].charger_id]
    assert invalidate_plans(scenario, offline_ids) == offline_ids
    assert invalidate_plans(demo_scenario, []) == []
    assert invalidate_plans(demo_scenario, ["NOT_THE_CHARGER"]) == []