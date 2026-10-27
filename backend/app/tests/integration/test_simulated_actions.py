"""Virtual actions: validated, recorded, and never touching hardware."""

from __future__ import annotations

import pytest

from app.schemas.events import SimulatedActionType

ACTION_SAMPLES = {
    SimulatedActionType.CABIN_MODE_CHANGE: ("vehicle", {"mode": "AIRPORT"}),
    SimulatedActionType.SEAT_LAYOUT_CHANGE: ("vehicle", {"passenger_count": 4}),
    SimulatedActionType.LUGGAGE_LAYOUT_CHANGE: (
        "vehicle",
        {"luggage_units": 3, "luggage_mass_kg": 55.0},
    ),
    SimulatedActionType.VEHICLE_MOVE: (
        "vehicle",
        {"zone_id": "Z02", "distance_km": 4.5, "energy_kwh": 1.2, "duration_s": 600.0},
    ),
    SimulatedActionType.CHARGER_CONNECT: ("charger", {}),
    SimulatedActionType.CHARGER_DISCONNECT: ("charger", {}),
    SimulatedActionType.HVAC_ZONE_CHANGE: ("vehicle", {"zones": ["row1", "row2"]}),
    SimulatedActionType.POWERTRAIN_SELECTION: ("vehicle", {"recommended": "NEXUS_X"}),
    SimulatedActionType.MAINTENANCE_FLAG: ("vehicle", {"flag": "tyre_check"}),
}


def apply(client, action_type, target_type, target_id, state):
    return client.post(
        "/api/simulation/actions",
        json={
            "action_type": action_type.value,
            "target_type": target_type,
            "target_id": target_id,
            "requested_state": state,
        },
    )


@pytest.mark.parametrize("action_type", list(SimulatedActionType))
def test_every_documented_action_type_is_a_simulated_record(
    client, action_type, vehicle_id, charger_id
):
    target_type, state = ACTION_SAMPLES[action_type]
    target_id = charger_id if target_type == "charger" else vehicle_id
    response = apply(client, action_type, target_type, target_id, state)
    assert response.status_code == 200, f"{action_type}: {response.text}"
    payload = response.json()
    action = payload["action"]
    assert payload["hardware_commanded"] is False
    assert action["status"] == "SIMULATED"
    assert action["hardware_commanded"] is False
    assert action["action_type"] == action_type.value
    assert action["target_id"] == target_id
    assert action["previous_state"] != action["resulting_state"] or action_type in {
        SimulatedActionType.POWERTRAIN_SELECTION,
        SimulatedActionType.CHARGER_CONNECT,
        SimulatedActionType.CHARGER_DISCONNECT,
    }
    assert action["simulated_at_utc"]
    assert "SIMULATED_ACTION_APPLIED" in action["reason_codes"]


def test_action_count_matches_the_nine_required_types():
    assert len(SimulatedActionType) == 9


def test_actions_reach_the_state_and_the_event_log(client, vehicle_id):
    response = apply(
        client,
        SimulatedActionType.SEAT_LAYOUT_CHANGE,
        "vehicle",
        vehicle_id,
        {"passenger_count": 5},
    )
    assert response.status_code == 200
    state = client.get("/api/state").json()
    vehicle = next(v for v in state["vehicles"] if v["vehicle_id"] == vehicle_id)
    assert vehicle["passenger_count"] == 5
    assert any(
        action["action_type"] == "SEAT_LAYOUT_CHANGE" for action in state["recent_actions"]
    )


def test_actions_stream_over_the_websocket(client, vehicle_id):
    with client.websocket_connect("/api/events") as socket:
        socket.receive_json()
        apply(
            client,
            SimulatedActionType.CABIN_MODE_CHANGE,
            "vehicle",
            vehicle_id,
            {"mode": "POOL"},
        )
        seen = set()
        for _ in range(4):
            event = socket.receive_json()
            seen.add(event["event_type"])
            if event["event_type"] == "SIMULATION_ACTION":
                assert event["payload"]["action"]["hardware_commanded"] is False
                break
    assert "SIMULATION_ACTION" in seen


def test_powertrain_selection_is_a_recommendation_not_a_configuration_change(
    client, vehicle_id
):
    before = next(
        v for v in client.get("/api/state").json()["vehicles"] if v["vehicle_id"] == vehicle_id
    )
    response = apply(
        client,
        SimulatedActionType.POWERTRAIN_SELECTION,
        "vehicle",
        vehicle_id,
        {"recommended": "NEXUS_X"},
    )
    after = next(
        v for v in client.get("/api/state").json()["vehicles"] if v["vehicle_id"] == vehicle_id
    )
    assert response.status_code == 200
    assert response.json()["action"]["resulting_state"]["applied_to_fleet"] is False
    assert after["powertrain"] == before["powertrain"]


def test_invalid_action_payload_and_unknown_targets_are_rejected(client, vehicle_id):
    response = client.post(
        "/api/simulation/actions",
        json={
            "action_type": "LAUNCH_ROCKET",
            "target_type": "vehicle",
            "target_id": vehicle_id,
            "requested_state": {},
        },
    )
    assert response.status_code == 422

    response = apply(
        client,
        SimulatedActionType.CABIN_MODE_CHANGE,
        "vehicle",
        "GHOST",
        {"mode": "POOL"},
    )
    assert response.status_code == 404

    response = apply(
        client,
        SimulatedActionType.CABIN_MODE_CHANGE,
        "vehicle",
        vehicle_id,
        {"mode": "TELEPORT"},
    )
    assert response.status_code in (404, 422)


def test_seat_layout_change_rejects_out_of_range_passengers(client, vehicle_id):
    response = apply(
        client,
        SimulatedActionType.SEAT_LAYOUT_CHANGE,
        "vehicle",
        vehicle_id,
        {"passenger_count": 12},
    )
    assert response.status_code in (422, 404)
    vehicle = next(
        v for v in client.get("/api/state").json()["vehicles"] if v["vehicle_id"] == vehicle_id
    )
    assert vehicle["passenger_count"] <= 8


def test_charger_connect_and_disconnect_track_plug_state(client, charger_id):
    apply(client, SimulatedActionType.CHARGER_CONNECT, "charger", charger_id, {})
    after_connect = next(
        c for c in client.get("/api/state").json()["chargers"] if c["charger_id"] == charger_id
    )
    apply(client, SimulatedActionType.CHARGER_DISCONNECT, "charger", charger_id, {})
    after_disconnect = next(
        c for c in client.get("/api/state").json()["chargers"] if c["charger_id"] == charger_id
    )
    assert after_connect["plugs_in_use"] >= 0
    assert after_disconnect["plugs_in_use"] <= after_connect["plugs_in_use"]


def test_vehicle_move_updates_position_and_battery(client, vehicle_id):
    before = next(
        v for v in client.get("/api/state").json()["vehicles"] if v["vehicle_id"] == vehicle_id
    )
    response = apply(
        client,
        SimulatedActionType.VEHICLE_MOVE,
        "vehicle",
        vehicle_id,
        {"zone_id": "Z03", "distance_km": 6.0, "energy_kwh": 1.5, "duration_s": 900.0},
    )
    after = next(
        v for v in client.get("/api/state").json()["vehicles"] if v["vehicle_id"] == vehicle_id
    )
    assert response.status_code == 200
    assert after["current_zone_id"] == "Z03"
    assert after["odometer_km"] == pytest.approx(before["odometer_km"] + 6.0, rel=1e-6)
    assert after["soc"] <= before["soc"] + 1e-9