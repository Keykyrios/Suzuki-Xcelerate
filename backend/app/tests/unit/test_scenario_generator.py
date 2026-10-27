"""Scenario generation: repeatability, faults, uncertainty families, replay."""

from __future__ import annotations

import pytest

from app.schemas.scenario import Scenario, ScenarioSpec
from app.simulation.scenario_generator import (
    apply_faults,
    demo_spec,
    faults_from_spec,
    generate_scenario,
    generate_scenario_batch,
    named_fault,
    sample_uncertainty_families,
)

FAULT_NAMES = (
    "charger_offline",
    "traffic_spike",
    "heat_wave",
    "low_soh_vehicle",
    "demand_surge",
)


def strip_timestamps(node):
    """Drop wall-clock fields so two generations of the same seed compare equal."""
    if isinstance(node, dict):
        return {
            key: strip_timestamps(value)
            for key, value in node.items()
            if not key.endswith("_utc")
        }
    if isinstance(node, list):
        return [strip_timestamps(item) for item in node]
    return node


def test_demo_spec_is_the_configured_scenario(cfg):
    spec = demo_spec(cfg)
    assert isinstance(spec, ScenarioSpec)
    assert spec.scenario_id
    assert spec.seed == int(cfg.demo_v("seeds.master_seed"))


def test_generation_is_deterministic_for_a_spec(cfg):
    spec = demo_spec(cfg)
    first = generate_scenario(spec, cfg)
    second = generate_scenario(spec, cfg)
    payload = strip_timestamps(first.model_dump(mode="json"))
    second_payload = strip_timestamps(second.model_dump(mode="json"))
    assert payload == second_payload


def test_scenario_survives_a_json_round_trip(cfg):
    scenario = generate_scenario(demo_spec(cfg), cfg)
    restored = Scenario.model_validate(scenario.model_dump(mode="json"))
    assert restored.scenario_id == scenario.scenario_id
    assert len(restored.vehicles) == len(scenario.vehicles)
    assert len(restored.chargers) == len(scenario.chargers)
    assert len(restored.requests) == len(scenario.requests)
    assert restored.seed == scenario.seed


def test_scenario_shape_matches_the_demonstrator_requirements(cfg):
    scenario = generate_scenario(demo_spec(cfg), cfg)
    assert len(scenario.graph.zones) >= 12
    assert len(scenario.chargers) >= 3
    assert len(scenario.vehicles) == 10
    assert any(z.kind == "airport" for z in scenario.graph.zones)
    assert any(z.is_intercity_terminal for z in scenario.graph.zones)
    assert all(charge.tariff_inr_per_kwh >= 0 for charge in scenario.chargers)


def test_every_named_fault_exists_in_the_configured_catalog(cfg):
    for name in FAULT_NAMES:
        fault = named_fault(name, cfg=cfg)
        assert fault.fault_id == f"F-{name}"
        assert fault.target
        assert fault.parameter


def test_unknown_fault_is_rejected(cfg):
    with pytest.raises(Exception):
        named_fault("ghost_fault", cfg=cfg)


def test_heat_wave_raises_the_ambient_temperature(cfg):
    scenario = generate_scenario(demo_spec(cfg), cfg)
    faults = faults_from_spec(
        ScenarioSpec(
            scenario_id="t",
            seed=scenario.seed,
            fault_injection={"name": "heat_wave", "target_id": scenario.vehicles[0].vehicle_id},
        ),
        cfg,
    )
    hotter = apply_faults(scenario, faults)
    assert hotter.environment.ambient_temp_c >= scenario.environment.ambient_temp_c


def test_charger_offline_fault_takes_a_charger_offline(cfg):
    scenario = generate_scenario(demo_spec(cfg), cfg)
    charger = scenario.chargers[0]
    faults = faults_from_spec(
        ScenarioSpec(
            scenario_id="t",
            seed=scenario.seed,
            fault_injection={"charger_id": charger.charger_id, "status": "OFFLINE"},
        ),
        cfg,
    )
    affected = apply_faults(scenario, faults)
    statuses = {c.charger_id: c.status.value for c in affected.chargers}
    assert statuses[charger.charger_id] == "OFFLINE"


def test_traffic_and_demand_faults_raise_their_factors(cfg):
    scenario = generate_scenario(demo_spec(cfg), cfg)
    faults = faults_from_spec(
        ScenarioSpec(
            scenario_id="t",
            seed=scenario.seed,
            fault_injection={"traffic_factor": 1.9, "demand_multiplier": 2.0},
        ),
        cfg,
    )
    assert len(faults) == 2
    spiked = apply_faults(scenario, faults)
    assert spiked.environment.traffic_factor == pytest.approx(1.9)
    assert spiked.demand_context.demand_multiplier == pytest.approx(2.0)


def test_low_soh_fault_degrades_one_vehicle(cfg):
    scenario = generate_scenario(demo_spec(cfg), cfg)
    target = scenario.vehicles[3]
    faults = faults_from_spec(
        ScenarioSpec(
            scenario_id="t",
            seed=scenario.seed,
            fault_injection={"vehicle_id": target.vehicle_id, "soh": 0.72},
        ),
        cfg,
    )
    degraded = apply_faults(scenario, faults)
    by_id = {v.vehicle_id: v.soh for v in degraded.vehicles}
    assert by_id[target.vehicle_id] == pytest.approx(0.72)


def test_unsupported_fault_payload_is_rejected(cfg):
    with pytest.raises(ValueError):
        faults_from_spec(
            ScenarioSpec(scenario_id="t", seed=1, fault_injection={"invented": True}), cfg
        )


def test_uncertainty_families_cover_the_configured_axes(cfg):
    draw = sample_uncertainty_families(5, cfg)
    for key in ("demand_multiplier", "traffic_factor", "charger_outages", "queue_delta",
                "ambient_temp_c", "vehicle_soh"):
        assert key in draw
    assert draw["demand_multiplier"] > 0.0
    assert draw["traffic_factor"] > 0.0
    assert len(draw["vehicle_soh"]) == 10
    assert draw["ambient_temp_c"] >= float(cfg.demo_v("uncertainty_families.ambient_temp_range_c")[0])


def test_uncertainty_draws_are_seeded(cfg):
    import numpy as np

    first = sample_uncertainty_families(17, cfg)
    second = sample_uncertainty_families(17, cfg)
    assert {key: list(np.atleast_1d(value)) for key, value in first.items()} == {
        key: list(np.atleast_1d(value)) for key, value in second.items()
    }
    third = sample_uncertainty_families(18, cfg)
    assert third["demand_multiplier"] != first["demand_multiplier"] or (
        third["ambient_temp_c"] != first["ambient_temp_c"]
    )


def test_regression_and_robustness_seed_sets_exist_and_are_disjoint(cfg):
    regression = list(cfg.demo_v("seeds.regression_seeds"))
    robustness = list(cfg.demo_v("seeds.robustness_seeds"))
    assert len(regression) >= 3
    assert len(robustness) >= 3
    assert not set(regression) & set(robustness)


def test_scenario_batch_varies_across_seeds(cfg):
    seeds = list(cfg.demo_v("seeds.regression_seeds"))[:3]
    batch = generate_scenario_batch(demo_spec(cfg), seeds, cfg)
    assert len(batch) == len(seeds)
    temperatures = {s.environment.ambient_temp_c for s in batch}
    assert len(temperatures) >= 1
    assert len({s.scenario_id for s in batch}) == len(batch)


def test_scenario_carries_provenance_and_version(cfg):
    scenario = generate_scenario(demo_spec(cfg), cfg)
    assert scenario.provenance.value == "SYNTHETIC"
    assert scenario.scenario_version
    assert scenario.demand_context.provenance.value in {"SYNTHETIC", "SCENARIO_ASSUMPTION"}


def test_requests_have_service_classes_fares_and_windows(cfg):
    scenario = generate_scenario(demo_spec(cfg), cfg)
    for request in scenario.requests:
        assert request.expected_fare_inr >= 0.0
        assert request.pickup_window_end_s >= request.pickup_window_start_s
        assert request.service_class.value in {"POOL", "COMMUTE", "AIRPORT", "INTERCITY_PREMIUM"}
        assert request.passenger_count >= 1