"""Synthetic demand: seeded counts, gravity destinations, provenance labels."""

from __future__ import annotations

import pytest

from app.demand.demand_model import (
    day_type_weights,
    destination_probabilities,
    diurnal_multipliers,
    generate_demand_records,
    negative_binomial_sample,
    service_weights,
)
from app.schemas.common import CabinMode, DayType, EvidenceLabel
from app.simulation.scenario_generator import demand_records_for


def test_configuration_shapes_are_complete(cfg):
    assert len(diurnal_multipliers(cfg)) == 24
    weights = day_type_weights(cfg)
    assert set(weights) == set(DayType)
    assert abs(sum(weights.values()) - 1.0) < 1e-6
    assert set(service_weights(cfg)) == set(CabinMode)
    assert all(w > 0 for w in service_weights(cfg).values())


def test_generation_is_reproducible_for_a_seed(demo_scenario, cfg):
    zones = demo_scenario.graph.zones
    first = generate_demand_records(zones, 40, seed=42, cfg=cfg)
    second = generate_demand_records(zones, 40, seed=42, cfg=cfg)
    ids = lambda records: [(r.hour_of_day, r.day_type, r.request_count, r.origin_zone_id) for r in records]
    assert ids(first) == ids(second)


def test_generation_differs_across_seeds(demo_scenario, cfg):
    zones = demo_scenario.graph.zones
    first = generate_demand_records(zones, 40, seed=1, cfg=cfg)
    second = generate_demand_records(zones, 40, seed=2, cfg=cfg)
    assert [r.origin_zone_id for r in first] != [r.origin_zone_id for r in second]


def test_every_record_carries_a_provenance_label(demo_scenario, cfg):
    zones = demo_scenario.graph.zones
    for record in generate_demand_records(zones, 60, seed=7, cfg=cfg):
        assert isinstance(record.source_status, EvidenceLabel)
        assert record.source_status == EvidenceLabel.SYNTHETIC
        assert 0 <= record.hour_of_day <= 23
        assert record.request_count >= 0
        assert record.origin_zone_id != record.destination_zone_id or record.request_count >= 0


def test_negative_binomial_is_overdispersed_and_non_negative(cfg):
    import numpy as np

    rng = np.random.default_rng(11)
    dispersion = float(cfg.demo_v("demand.negative_binomial_dispersion"))
    samples = [
        negative_binomial_sample(rng, 20.0, dispersion) for _ in range(4000)
    ]
    mean = sum(samples) / len(samples)
    variance = sum((s - mean) ** 2 for s in samples) / len(samples)
    assert all(s >= 0 for s in samples)
    assert mean == pytest.approx(20.0, rel=0.15)
    assert variance > 20.0  # over-dispersed relative to a Poisson
    assert negative_binomial_sample(rng, 0.0, dispersion) == 0


def test_destinations_form_a_probability_distribution(demo_scenario, cfg):
    zones = demo_scenario.graph.zones
    origin = zones[0].zone_id
    probs = destination_probabilities(origin, zones, CabinMode.COMMUTE, cfg)
    assert origin not in probs
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-9)
    assert all(0.0 <= p <= 1.0 for p in probs.values())


def test_airport_prefers_airport_consumers(demo_scenario, cfg):
    zones = demo_scenario.graph.zones
    airport = next(z for z in zones if z.kind == "airport")
    airport_zone = next(z for z in zones if z.kind == "urban")
    air = destination_probabilities(airport_zone.zone_id, zones, CabinMode.AIRPORT, cfg)
    plain = destination_probabilities(airport_zone.zone_id, zones, CabinMode.COMMUTE, cfg)
    assert air[airport.zone_id] > plain[airport.zone_id]


def test_unknown_origin_zone_raises(demo_scenario, cfg):
    with pytest.raises(KeyError):
        destination_probabilities("MISSING", demo_scenario.graph.zones, CabinMode.COMMUTE, cfg)


def test_demand_records_for_is_deterministic(demo_scenario, cfg):
    first = demand_records_for(demo_scenario, cfg)
    second = demand_records_for(demo_scenario, cfg)
    assert len(first) == len(second)
    assert [r.request_count for r in first] == [r.request_count for r in second]
    assert all(r.source_status == EvidenceLabel.SYNTHETIC for r in first)