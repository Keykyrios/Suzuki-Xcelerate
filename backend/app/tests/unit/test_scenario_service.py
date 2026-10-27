"""Scenario service: deterministic creation, fault catalog, guarded updates."""

from __future__ import annotations

import pytest

from app.schemas.common import CabinMode, ChargerStatus
from app.schemas.scenario import ScenarioSpec
from app.services.scenario_service import ScenarioService
from app.services.state_store import reset_state_store


@pytest.fixture
def service():
    return ScenarioService(store=reset_state_store())


def test_reset_demo_is_deterministic(service):
    first = service.reset_demo()
    second = service.reset_demo()
    assert first.scenario_id == second.scenario_id
    assert [v.vehicle_id for v in first.vehicles] == [v.vehicle_id for v in second.vehicles]
    assert [r.request_id for r in first.requests] == [r.request_id for r in second.requests]
    assert first.seed == second.seed


def test_create_accepts_a_validated_spec_with_overrides(service):
    spec = ScenarioSpec(
        scenario_id="scenario-unit-create",
        seed=7,
        cabin={"mode": "AIRPORT", "passenger_count": 4, "luggage_count": 2, "luggage_mass_kg": 30},
        environment={"ambient_temp_c": 40.0},
    )
    scenario = service.create(spec)
    assert scenario.seed == 7
    assert scenario.environment.ambient_temp_c == 40.0
    assert service.store.require_scenario().scenario_id == "scenario-unit-create"


def test_named_faults_are_applied_from_the_configured_catalog(service):
    scenario = service.reset_demo()
    heated = service.apply_fault("heat_wave")
    assert heated.environment.ambient_temp_c == 45.0
    assert [f.name for f in heated.faults] == ["heat_wave"]
    charger_id = scenario.chargers[0].charger_id
    outage = service.apply_fault("charger_offline", target_id=charger_id)
    assert outage.charger(charger_id).status is ChargerStatus.OFFLINE
    assert service.store.recommendation_stale is True


def test_vehicle_updates_are_allowlisted(service):
    scenario = service.reset_demo()
    vehicle_id = scenario.vehicles[0].vehicle_id
    updated = service.update_vehicle_state(vehicle_id, {"cabin_mode": CabinMode.AIRPORT.value})
    assert updated.cabin_mode is CabinMode.AIRPORT
    with pytest.raises(ValueError):
        service.update_vehicle_state(vehicle_id, {"vehicle_id": "rewritten"})
    with pytest.raises(KeyError):
        service.update_vehicle_state("ghost", {"soc": 0.5})
