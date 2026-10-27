"""API surface: required endpoints, validation, and explicit error behaviour."""

from __future__ import annotations

import numpy as np

from app import __version__

REQUIRED_GET = {"/api/health", "/api/config", "/api/state"}
REQUIRED_POST = {
    "/api/scenarios",
    "/api/simulate",
    "/api/optimize",
    "/api/chargers/{charger_id}/status",
    "/api/vehicles/{vehicle_id}/state",
    "/api/simulation/actions",
    "/api/runs/{run_id}/replay",
    "/api/demo/reset",
}


def test_every_required_endpoint_is_registered(client):
    paths = set(client.get("/openapi.json").json()["paths"])
    assert REQUIRED_GET <= paths
    assert REQUIRED_POST <= paths
    assert "/api/events" not in paths  # the event stream is not an OpenAPI route


def test_health_reports_version_solver_and_state(client):
    payload = client.get("/api/health").json()
    assert payload["status"] == "ok"
    assert payload["version"] == __version__
    assert len(payload["config_hash"]) == 64
    assert payload["solver"]["available"] is True
    assert payload["scenario_loaded"] is True
    assert payload["recommendation_stale"] is False
    assert set(payload["config_versions"]) == {
        "vehicle",
        "operating_policy",
        "optimizer",
        "economics",
        "demo_scenarios",
    }
    assert {"physics", "scenario"} <= set(payload["model_versions"])


def test_config_exposes_units_and_status_labels(client):
    payload = client.get("/api/config").json()
    parameters = payload["parameters"]
    assert parameters
    for parameter in parameters:
        assert parameter["unit"], f"{parameter['path']} has no unit"
        assert parameter["status"], f"{parameter['path']} has no status"
        assert parameter["source_or_rationale"]
        assert parameter["file"]
    assert payload["config_hash"] == client.get("/api/health").json()["config_hash"]


def test_state_snapshot_carries_the_expected_entities(client):
    snapshot = client.get("/api/state").json()
    assert len(snapshot["vehicles"]) == 10
    assert len(snapshot["chargers"]) >= 3
    assert len(snapshot["graph"]["zones"]) >= 12
    assert snapshot["state_version"] >= 1
    assert snapshot["scenario_id"]
    assert snapshot["decision"] is None
    assert snapshot["solver_status"] is None
    assert snapshot["recommendation_stale"] is False


def test_unknown_endpoint_returns_404(client):
    assert client.get("/api/nothing_here").status_code == 404


def test_invalid_json_body_is_rejected(client):
    response = client.post(
        "/api/simulate", content=b"{not json", headers={"content-type": "application/json"}
    )
    assert response.status_code == 422


def test_unknown_fields_are_rejected_everywhere(client):
    response = client.post(
        "/api/scenarios",
        json={"scenario_id": "s", "seed": 1, "invented_field": True},
    )
    assert response.status_code == 422


def test_out_of_range_values_are_rejected(client, vehicle_id, zones):
    response = client.post(f"/api/vehicles/{vehicle_id}/state", json={"soc": 1.5})
    assert response.status_code == 422
    response = client.post(
        "/api/simulate",
        json={
            "vehicle_id": vehicle_id,
            "route": {"origin_zone": zones[0]["zone_id"], "destination_zone": zones[-1]["zone_id"]},
            "ambient_temp_c": 99.0,
        },
    )
    assert response.status_code == 422


def test_unknown_vehicle_and_charger_return_404(client, zones):
    response = client.post(
        "/api/simulate",
        json={
            "vehicle_id": "GHOST",
            "route": {"origin_zone": zones[0]["zone_id"], "destination_zone": zones[-1]["zone_id"]},
        },
    )
    assert response.status_code == 404
    response = client.post(
        "/api/chargers/GHOST/status", json={"status": "OFFLINE", "queue_length": 0}
    )
    assert response.status_code == 404
    response = client.post("/api/vehicles/GHOST/state", json={"soc": 0.5})
    assert response.status_code == 404


def test_empty_vehicle_update_is_rejected(client, vehicle_id):
    response = client.post(f"/api/vehicles/{vehicle_id}/state", json={})
    assert response.status_code == 422


def test_disallowed_vehicle_fields_are_rejected(client, vehicle_id):
    response = client.post(f"/api/vehicles/{vehicle_id}/state", json={"soh": 0.5, "powertrain": "NEXUS_X"})
    # soh is permitted; powertrain is not part of the mutable surface.
    assert response.status_code == 422
    assert "powertrain" in response.text


def test_unknown_run_id_is_an_explicit_not_found(client):
    assert client.get("/api/runs/run_zzzzzz").status_code == 404
    replay = client.post("/api/runs/run_zzzzzz/replay")
    assert replay.status_code == 404


def test_route_response_types_are_json_serialisable(client, zones, vehicle_id):
    response = client.post(
        "/api/simulate",
        json={
            "vehicle_id": vehicle_id,
            "route": {"origin_zone": zones[0]["zone_id"], "destination_zone": zones[-1]["zone_id"]},
        },
    )
    payload = response.json()
    assert response.headers["content-type"].startswith("application/json")
    # NumPy containers must never reach the encoder.
    text = response.text
    assert "ndarray" not in text
    assert isinstance(payload["route_energy_kwh"], float)
    assert not isinstance(payload["route_energy_kwh"], bool)
    assert isinstance(payload["constraint_checks"], dict)


def test_optimize_response_is_plain_json_numbers(client):
    response = client.post("/api/optimize", json={})
    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload["run_id"], str)
    assert isinstance(payload["recheck_passed"], bool)
    for value in payload["baseline"]["loss_inr"]:
        assert isinstance(value, float)
        assert np.isfinite(value)