"""Run manifests, retrieval, and deterministic replay."""

from __future__ import annotations


def test_optimize_records_a_run_that_can_be_listed_and_read(client, solved):
    run_id = solved["run_id"]
    listing = client.get("/api/runs").json()
    assert run_id in listing["runs"]
    response = client.get(f"/api/runs/{run_id}")
    assert response.status_code == 200
    payload = response.json()
    manifest = payload["manifest"]
    assert manifest["run_id"] == run_id
    assert manifest["scenario_id"] == solved["scenario_id"]
    assert manifest["seed"] >= 0
    assert len(manifest["config_hash"]) == 64
    assert manifest["git_commit"]
    assert manifest["model_versions"]["physics"]
    assert manifest["solver_status"] == solved["status"]
    assert {"completed_requests", "contribution_inr", "cvar_loss_inr"} <= set(
        manifest["headline_metrics"]
    )
    assert payload["explanation"]
    assert run_id in payload["explanation"]


def test_replay_reproduces_the_decision_for_deterministic_settings(client, solved):
    run_id = solved["run_id"]
    replay = client.post(f"/api/runs/{run_id}/replay")
    assert replay.status_code == 200, replay.text
    payload = replay.json()
    original_pairs = sorted(
        (a["vehicle_id"], a["request_id"]) for a in solved["decision"]["assignments"]
    )
    replay_pairs = sorted(
        (a["vehicle_id"], a["request_id"]) for a in payload["decision"]["assignments"]
    )
    assert payload["scenario_id"] == solved["scenario_id"]
    assert replay_pairs == original_pairs
    assert payload["decision"]["status"] == solved["status"]


def test_reproduced_numbers_match_within_documented_rounding(client, solved):
    run_id = solved["run_id"]
    replay = client.post(f"/api/runs/{run_id}/replay").json()
    original = solved["decision"]
    repeated = replay["decision"]
    assert repeated["decision"]["departure_soc_pct"] == original["decision"]["departure_soc_pct"]
    assert repeated["decision"]["arrival_soc_pct"] == original["decision"]["arrival_soc_pct"]
    assert repeated["decision"]["route_energy_kwh"] == original["decision"]["route_energy_kwh"]
    assert repeated["optimized_metrics"] == original["optimized_metrics"]
    assert repeated["baseline_metrics"] == original["baseline_metrics"]


def test_replay_restores_the_run_scenario_into_state(client, solved):
    run_id = solved["run_id"]
    response = client.post(f"/api/runs/{run_id}/replay")
    assert response.status_code == 200
    state = client.get("/api/state").json()
    assert state["scenario_id"] == solved["scenario_id"]
    assert state["run_id"] == run_id


def test_unknown_run_is_rejected_explicitly(client):
    assert client.get("/api/runs/run_missing").status_code == 404
    assert client.post("/api/runs/run_missing/replay").status_code == 404


def test_replay_restores_the_scenario_from_stored_inputs_not_memory(client, solved):
    run_id = solved["run_id"]
    client.post("/api/demo/reset")
    assert client.get("/api/state").json()["scenario_id"] != "replayed-other"
    replay = client.post(f"/api/runs/{run_id}/replay")
    assert replay.status_code == 200
    assert replay.json()["manifest"]["run_id"] == run_id


def test_repeated_optimize_runs_get_distinct_run_ids(client):
    first = client.post("/api/optimize", json={}).json()["run_id"]
    second = client.post("/api/optimize", json={}).json()["run_id"]
    assert first != second
    runs = client.get("/api/runs").json()["runs"]
    assert {first, second} <= set(runs)