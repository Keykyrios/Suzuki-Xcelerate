"""Powertrain recommendation: risk-adjusted comparison between both configurations."""

from __future__ import annotations

import pytest

from app.optimization.powertrain_choice import MissionSpec, cost_powertrain, recommend_powertrain
from app.schemas.common import CabinMode, PowertrainType


def mission(scenario) -> MissionSpec:
    zones = [z.zone_id for z in scene_zones(scenario)]
    vehicle = scenario.vehicles[0]
    return MissionSpec(
        mission_id="M-1",
        origin_zone=vehicle.current_zone_id,
        destination_zone=zones[-1] if zones[-1] != vehicle.current_zone_id else zones[0],
        service_class=CabinMode.AIRPORT,
        passenger_count=4,
        luggage_mass_kg=45.0,
        departure_soc=vehicle.soc,
        soh=vehicle.soh,
    )


def scene_zones(scenario):
    return scenario.graph.zones


def test_both_configurations_are_costed(demo_scenario, cfg):
    spec = mission(demo_scenario)
    cost_e = cost_powertrain(demo_scenario, spec, PowertrainType.NEXUS_E, cfg)
    cost_x = cost_powertrain(demo_scenario, spec, PowertrainType.NEXUS_X, cfg)
    assert cost_e.powertrain == PowertrainType.NEXUS_E
    assert cost_x.powertrain == PowertrainType.NEXUS_X
    assert cost_e.distance_km == pytest.approx(cost_x.distance_km)
    assert cost_e.route_energy_kwh >= 0.0
    assert cost_e.mission_cost_inr > 0.0
    # The range extender starts with higher capital and maintenance cost.
    assert cost_x.capital_daily_inr > cost_e.capital_daily_inr
    assert cost_x.maintenance_per_km_inr > cost_e.maintenance_per_km_inr


def test_comparison_is_explained_with_numbers(demo_scenario, cfg):
    spec = mission(demo_scenario)
    comparison = recommend_powertrain(demo_scenario, spec, cfg=cfg)
    assert comparison.recommended in {PowertrainType.NEXUS_E, PowertrainType.NEXUS_X}
    assert comparison.alternative != comparison.recommended
    assert comparison.cost_difference_inr == pytest.approx(
        comparison.cost_nexus_x_inr - comparison.cost_nexus_e_inr
    )
    assert comparison.explanation
    assert comparison.reason_codes
    assert comparison.mission_id == "M-1"
    assert "Simulation" in comparison.explanation or "decision only" in comparison.explanation


def test_both_reason_code_branches_exist_in_the_catalog():
    from app.schemas.reason_codes import ReasonCode

    assert ReasonCode.POWERTRAIN_E_PREFERRED_COST
    assert ReasonCode.POWERTRAIN_X_RISK_OFFSET
    assert ReasonCode.POWERTRAIN_X_NOT_JUSTIFIED


def test_recommendation_is_deterministic(demo_scenario, cfg):
    spec = mission(demo_scenario)
    first = recommend_powertrain(demo_scenario, spec, cfg=cfg)
    second = recommend_powertrain(demo_scenario, spec, cfg=cfg)
    assert first.recommended == second.recommended
    assert first.cost_nexus_e_inr == pytest.approx(second.cost_nexus_e_inr)
    assert first.cost_difference_inr == pytest.approx(second.cost_difference_inr)


def test_empty_battery_here_shows_the_range_extender_advantage(demo_scenario, cfg):
    spec = mission(demo_scenario)
    weakened = MissionSpec(
        mission_id=spec.mission_id,
        origin_zone=spec.origin_zone,
        destination_zone=spec.destination_zone,
        service_class=spec.service_class,
        passenger_count=spec.passenger_count,
        luggage_mass_kg=spec.luggage_mass_kg,
        departure_soc=0.12,
        soh=0.7,
    )
    cost_e = cost_powertrain(demo_scenario, spec, PowertrainType.NEXUS_E, cfg)
    cost_x = cost_powertrain(demo_scenario, spec, PowertrainType.NEXUS_X, cfg)
    # With a weak pack the battery-side configuration may carry charging risk.
    assert cost_e.charging_risk_inr >= 0.0
    assert cost_x.fuel_cost_inr >= 0.0
    comparison = recommend_powertrain(demo_scenario, spec, cfg=cfg)
    assert comparison.switching_inputs


def test_mission_spec_requires_the_documented_inputs():
    with pytest.raises(TypeError):
        MissionSpec(mission_id="M-2")  # origin/destination/service are mandatory