"""Canonical state store: versioning, events, staleness, guarded updates."""

from __future__ import annotations

import pytest

from app.schemas.common import ChargerStatus
from app.schemas.events import EventType
from app.services.scenario_service import ScenarioService
from app.services.state_store import reset_state_store


@pytest.fixture
def store():
    return reset_state_store()


@pytest.fixture
def scenario(store):
    return ScenarioService(store=store).reset_demo()


def test_mutations_bump_monotonic_state_versions(store, scenario):
    start = store.state_version
    vehicle_id = scenario.vehicles[0].vehicle_id
    first = store.update_vehicle(vehicle_id, {"passenger_count": 2})
    assert store.state_version > start
    seen = store.state_version
    charger_id = scenario.chargers[0].charger_id
    store.update_charger(charger_id, {"status": ChargerStatus.MAINTENANCE})
    assert store.state_version > seen
    assert first.passenger_count == 2


def test_listeners_receive_versioned_events(store, scenario):
    received = []
    unsubscribe = store.subscribe(received.append)
    store.update_vehicle(scenario.vehicles[0].vehicle_id, {"soc": 0.5})
    unsubscribe()
    store.update_vehicle(scenario.vehicles[0].vehicle_id, {"soc": 0.4})
    assert len(received) == 1
    event = received[0]
    assert event.event_type is EventType.VEHICLE_UPDATED
    assert event.state_version == store.state_version
    assert event.scenario_id == scenario.scenario_id
    assert event.event_id


def test_staleness_flag_only_publishes_on_change(store, scenario):
    received = []
    store.subscribe(received.append)
    store.set_stale(True, reason="charger changed")
    store.set_stale(True, reason="repeated")
    flags = [e for e in received if e.event_type is EventType.SOLVER_STATUS]
    assert store.recommendation_stale is True
    assert len(flags) == 1
    assert flags[0].payload["stale"] is True


def test_unknown_entities_are_rejected_explicitly(store, scenario):
    with pytest.raises(KeyError):
        store.update_vehicle("no-such-vehicle", {"soc": 0.5})
    with pytest.raises(KeyError):
        store.update_charger("no-such-charger", {"queue_length": 1})
    with pytest.raises(RuntimeError):
        reset_state_store().require_scenario()


def test_snapshot_reflects_canonical_state(store, scenario):
    snapshot = store.snapshot()
    payload = snapshot.model_dump(mode="json")
    assert payload["scenario_id"] == scenario.scenario_id
    assert payload["state_version"] == store.state_version
    assert len(payload["vehicles"]) == len(scenario.vehicles)
    assert len(payload["chargers"]) == len(scenario.chargers)
    assert payload["recommendation_stale"] is False
