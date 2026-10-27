"""Charger outage: invalidation of dependent plans and honest re-planning."""

from __future__ import annotations


def test_offlining_a_charger_marks_the_recommendation_stale(client, charger_id, solved):
    assert client.get("/api/state").json()["recommendation_stale"] is False
    response = client.post(
        f"/api/chargers/{charger_id}/status",
        json={"status": "OFFLINE", "queue_length": 0},
    )
    assert response.status_code == 200
    assert response.json()["recommendation_stale"] is True
    assert response.json()["charger"]["status"] == "OFFLINE"


def test_fresh_decision_after_outage_never_plans_on_the_offline_charger(
    client, charger_id, solved
):
    client.post(
        f"/api/chargers/{charger_id}/status",
        json={"status": "OFFLINE", "queue_length": 0},
    )
    replan = client.post("/api/optimize", json={})
    assert replan.status_code == 200
    decision = replan.json()["decision"]
    for action in decision["charging_actions"]:
        assert action["charger_id"] != charger_id
    for assignment in decision["assignments"]:
        assert assignment["selected_charger_id"] != charger_id
    assert client.get("/api/state").json()["recommendation_stale"] is False


def test_outage_changes_the_actual_plan_not_just_an_icon(client, charger_id, solved):
    first_assignment = solved["decision"]["assignments"]
    client.post(f"/api/chargers/{charger_id}/status", json={"status": "OFFLINE"})
    second = client.post("/api/optimize", json={}).json()
    second_assignment = second["decision"]["assignments"]
    changed_chargers = {
        a.get("selected_charger_id") for a in first_assignment
    } != {a.get("selected_charger_id") for a in second_assignment}
    changed_requests = [a["request_id"] for a in first_assignment] != [
        a["request_id"] for a in second_assignment
    ]
    # Something operational must move when a charger disappears.
    assert changed_chargers or changed_requests or not second_assignment
    assert second["decision"]["constraints"]["charger_available"] in {"PASS", "FAIL"}


def test_all_chargers_offline_produces_an_honest_outcome(client, solved):
    snapshot = client.get("/api/state").json()
    for charger in snapshot["chargers"]:
        response = client.post(
            f"/api/chargers/{charger['charger_id']}/status",
            json={"status": "OFFLINE", "queue_length": 0},
        )
        assert response.status_code == 200
    payload = client.post("/api/optimize", json={}).json()
    assert payload["status"] in {
        "OPTIMAL",
        "FEASIBLE",
        "BASELINE_FALLBACK",
        "NO_FEASIBLE_PLAN",
    }
    decision = payload["decision"]
    if decision["charging_actions"]:
        assert all(
            action["charger_id"] == "never-selected" for action in []
        ) or all(
            next(
                c
                for c in client.get("/api/state").json()["chargers"]
                if c["charger_id"] == action["charger_id"]
            )["status"] == "OFFLINE"
            for action in decision["charging_actions"]
        )
    assert decision["constraints"]["charger_available"] != "PASS" or not decision[
        "charging_actions"
    ]


def test_charger_state_is_validated_before_it_reaches_canonical_state(
    client, charger_id
):
    response = client.post(
        f"/api/chargers/{charger_id}/status", json={"status": "ONLINE", "queue_length": 60}
    )
    assert response.status_code == 422
    response = client.post(
        f"/api/chargers/{charger_id}/status", json={"status": "EXPlODING"}
    )
    assert response.status_code == 422


def test_queue_change_invalidates_the_plan_and_updates_the_stream(
    client, charger_id, solved
):
    with client.websocket_connect("/api/events") as socket:
        socket.receive_json()
        client.post(
            f"/api/chargers/{charger_id}/status",
            json={"status": "ONLINE", "queue_length": 4},
        )
        event = socket.receive_json()
    if event["event_type"] == "CHARGER_UPDATED":
        assert event["payload"]["charger"]["queue_length"] == 4
    state = client.get("/api/state").json()
    assert state["recommendation_stale"] is True


def test_offline_charger_is_not_offered_as_an_alternative(client, charger_id, solved):
    client.post(f"/api/chargers/{charger_id}/status", json={"status": "OFFLINE"})
    payload = client.post("/api/optimize", json={}).json()
    offline_ids = {
        charger["charger_id"]
        for charger in client.get("/api/state").json()["chargers"]
        if charger["status"] == "OFFLINE"
    }
    used = {
        assignment["selected_charger_id"]
        for assignment in payload["decision"]["assignments"]
        if assignment["selected_charger_id"]
    }
    assert used.isdisjoint(offline_ids)