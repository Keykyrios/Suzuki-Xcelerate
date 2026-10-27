"""State synchronisation: every mutation bumps the version and reaches clients."""

from __future__ import annotations


def test_state_version_advances_on_every_mutation(client, vehicle_id, charger_id):
    start = client.get("/api/state").json()["state_version"]
    client.post(f"/api/vehicles/{vehicle_id}/state", json={"soc": 0.5})
    after_vehicle = client.get("/api/state").json()["state_version"]
    client.post(f"/api/chargers/{charger_id}/status", json={"status": "MAINTENANCE"})
    after_charger = client.get("/api/state").json()["state_version"]
    assert after_vehicle > start
    assert after_charger > after_vehicle


def test_rest_endpoint_and_stream_report_the_same_version(client, vehicle_id):
    with client.websocket_connect("/api/events") as socket:
        socket.receive_json()  # snapshot
        response = client.post(f"/api/vehicles/{vehicle_id}/state", json={"soc": 0.52})
        reported = response.json()["state_version"]
        versions = []
        for _ in range(4):
            event = socket.receive_json()
            versions.append(event["state_version"])
            if event["state_version"] >= reported:
                break
        rest_state = client.get("/api/state").json()
    # Every mutation publishes an ordered, gap-free chain up to the reported version.
    assert versions == sorted(versions)
    assert versions[-1] == reported
    assert rest_state["state_version"] == reported


def test_snapshot_after_a_mutation_matches_the_stored_state(client, vehicle_id):
    client.post(f"/api/vehicles/{vehicle_id}/state", json={"soc": 0.47})
    snapshot = client.get("/api/state").json()
    vehicle = next(v for v in snapshot["vehicles"] if v["vehicle_id"] == vehicle_id)
    assert vehicle["soc"] == 0.47
    assert vehicle["state_version"] >= 1


def test_mutation_marks_the_recommendation_stale(client, vehicle_id, solved):
    assert client.get("/api/state").json()["recommendation_stale"] is False
    client.post(f"/api/vehicles/{vehicle_id}/state", json={"soc": 0.6})
    state = client.get("/api/state").json()
    assert state["recommendation_stale"] is True
    assert client.get("/api/health").json()["recommendation_stale"] is True


def test_state_timestamps_move_forward(client, vehicle_id):
    first = client.get("/api/state").json()["timestamp_utc"]
    client.post(f"/api/vehicles/{vehicle_id}/state", json={"odometer_km": 1500.0})
    second = client.get("/api/state").json()["timestamp_utc"]
    assert second >= first


def test_cabin_actions_reach_the_state_snapshot(client, vehicle_id):
    response = client.post(
        "/api/simulation/actions",
        json={
            "action_type": "CABIN_MODE_CHANGE",
            "target_type": "vehicle",
            "target_id": vehicle_id,
            "requested_state": {"mode": "AIRPORT"},
        },
    )
    assert response.status_code == 200, response.text
    snapshot = client.get("/api/state").json()
    vehicle = next(v for v in snapshot["vehicles"] if v["vehicle_id"] == vehicle_id)
    assert vehicle["cabin_mode"] == "AIRPORT"
    assert any(
        action["action_type"] == "CABIN_MODE_CHANGE"
        for action in snapshot["recent_actions"]
    )


def test_session_history_is_recorded_in_the_event_log(client, vehicle_id):
    from app.services.state_store import get_state_store

    client.post(f"/api/vehicles/{vehicle_id}/state", json={"soc": 0.41})
    client.post(
        "/api/simulation/actions",
        json={
            "action_type": "HVAC_ZONE_CHANGE",
            "target_type": "vehicle",
            "target_id": vehicle_id,
            "requested_state": {"zones": ["row1"]},
        },
    )
    kinds = [event.event_type.value for event in get_state_store().recent_events(50)]
    assert "VEHICLE_UPDATED" in kinds
    assert "SIMULATION_ACTION" in kinds
    versions = [event.state_version for event in get_state_store().recent_events(50)]
    assert versions == sorted(versions)