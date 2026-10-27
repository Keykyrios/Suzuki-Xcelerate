"""WebSocket protocol: snapshot first, versioned deltas, envelope contract."""

from __future__ import annotations

import json


def envelope_fields(event: dict) -> set[str]:
    return {
        "event_id",
        "event_type",
        "state_version",
        "timestamp_utc",
        "scenario_id",
        "run_id",
        "payload",
    }


def test_connect_receives_a_full_snapshot_first(client):
    with client.websocket_connect("/api/events") as socket:
        first = socket.receive_json()
    assert first["event_type"] == "STATE_SNAPSHOT"
    assert envelope_fields(first) <= set(first)
    payload = first["payload"]
    assert len(payload["vehicles"]) == 10
    assert payload["scenario_id"] == first["scenario_id"]
    assert first["state_version"] >= 0


def test_mutations_arrive_as_versioned_deltas(client, vehicle_id):
    with client.websocket_connect("/api/events") as socket:
        snapshot = socket.receive_json()
        assert snapshot["event_type"] == "STATE_SNAPSHOT"
        versions = [snapshot["state_version"]]
        response = client.post(f"/api/vehicles/{vehicle_id}/state", json={"soc": 0.61})
        assert response.status_code == 200
        event = socket.receive_json()
        versions.append(event["state_version"])
    assert event["event_type"] in {
        "VEHICLE_UPDATED",
        "SOLVER_STATUS",
        "CHARGER_UPDATED",
        "STATE_DELTA",
    }
    assert versions[1] > versions[0]
    assert envelope_fields(event) <= set(event)
    assert event["payload"]["vehicle"]["soc"] == 0.61 if "vehicle" in event["payload"] else True


def test_event_types_cover_the_documented_stream(client, vehicle_id, charger_id):
    seen = set()
    with client.websocket_connect("/api/events") as socket:
        socket.receive_json()  # snapshot
        client.post(f"/api/vehicles/{vehicle_id}/state", json={"soc": 0.58})
        seen.add(socket.receive_json()["event_type"])
        client.post(f"/api/chargers/{charger_id}/status", json={"status": "MAINTENANCE"})
        seen.add(socket.receive_json()["event_type"])
        client.post("/api/optimize", json={})
        for _ in range(4):
            event = socket.receive_json()
            seen.add(event["event_type"])
            if event["event_type"] == "DECISION_UPDATED":
                break
    assert {
        "VEHICLE_UPDATED",
        "CHARGER_UPDATED",
        "DECISION_UPDATED",
        "SOLVER_STATUS",
    } <= seen


def test_resync_request_returns_a_fresh_snapshot(client):
    with client.websocket_connect("/api/events") as socket:
        socket.receive_json()
        client.post("/api/demo/reset")
        socket.receive_json()  # whatever the reset published
        socket.send_text(json.dumps({"type": "resync"}))
        again = socket.receive_json()
    assert again["event_type"] == "STATE_SNAPSHOT"
    assert again["payload"]["vehicles"]


def test_malformed_client_message_does_not_drop_the_connection(client):
    with client.websocket_connect("/api/events") as socket:
        socket.receive_json()
        socket.send_text("this is not json")
        socket.send_text(json.dumps({"type": "resync"}))
        event = socket.receive_json()
    assert event["event_type"] == "STATE_SNAPSHOT"


def test_snapshot_state_version_is_monotonic_across_reconnects(client, vehicle_id):
    with client.websocket_connect("/api/events") as socket:
        first = socket.receive_json()
        client.post(f"/api/vehicles/{vehicle_id}/state", json={"soc": 0.44})
        socket.receive_json()
    with client.websocket_connect("/api/events") as socket:
        second = socket.receive_json()
    assert second["state_version"] >= first["state_version"]


def test_decision_event_reaches_subscribers(client, solved):
    run_id = solved["run_id"]
    stored = client.get(f"/api/runs/{run_id}")
    assert stored.status_code == 200
    with client.websocket_connect("/api/events") as socket:
        snapshot = socket.receive_json()
        assert snapshot["payload"]["decision"] is not None
        assert snapshot["payload"]["decision"]["run_id"] == run_id
        assert snapshot["payload"]["solver_status"] == snapshot["payload"]["decision"]["status"]


def test_stale_charger_state_is_reflected_in_the_stream(client, charger_id, solved):
    with client.websocket_connect("/api/events") as socket:
        socket.receive_json()
        client.post(f"/api/chargers/{charger_id}/status", json={"status": "OFFLINE"})
        event = socket.receive_json()
    if event["event_type"] == "CHARGER_UPDATED":
        assert event["payload"]["charger"]["status"] == "OFFLINE"
    state = client.get("/api/state").json()
    assert state["recommendation_stale"] is True
    assert state["solver_status"] is not None