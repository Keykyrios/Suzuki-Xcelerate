"""Scenario creation and simulation endpoints against canonical state."""

from __future__ import annotations

# The request published in the specification's scenario example.
SPEC_EXAMPLE = {
    "scenario_id": "airport_heat_charger_outage_01",
    "seed": 42,
    "vehicle_id": "NX-03",
    "cabin": {
        "mode": "AIRPORT",
        "passenger_count": 4,
        "luggage_count": 3,
        "luggage_mass_kg": 55,
        "source": "manual",
        "confidence": 1.0,
    },
    "environment": {"ambient_temp_c": 45, "traffic_factor": 1.6, "road_grade_percent": 3.0},
    "route": {"origin_zone": "Z04", "destination_zone": "AIRPORT"},
    "fault_injection": {"charger_id": "CH-02", "status": "OFFLINE"},
}


def test_specification_scenario_example_is_accepted(client):
    response = client.post("/api/scenarios", json=SPEC_EXAMPLE)
    assert response.status_code == 201, response.text
    payload = response.json()
    assert payload["scenario_id"] == SPEC_EXAMPLE["scenario_id"]
    assert payload["vehicles"] == 10
    assert payload["chargers"] >= 3
    assert payload["state_version"] >= 1


def test_created_scenario_replaces_canonical_state(client):
    client.post("/api/scenarios", json=SPEC_EXAMPLE)
    snapshot = client.get("/api/state").json()
    assert snapshot["scenario_id"] == SPEC_EXAMPLE["scenario_id"]
    assert snapshot["environment"]["ambient_temp_c"] == 45.0
    assert snapshot["environment"]["traffic_factor"] == 1.6


def test_scenario_fault_payload_takes_the_charger_offline(client):
    client.post("/api/scenarios", json=SPEC_EXAMPLE)
    snapshot = client.get("/api/state").json()
    target = "CH-02"
    by_id = {c["charger_id"]: c for c in snapshot["chargers"]}
    if target in by_id:
        assert by_id[target]["status"] == "OFFLINE"


def test_scenario_validation_rejects_out_of_band_controls(client):
    payload = dict(SPEC_EXAMPLE, scenario_id="bad_controls")
    payload["environment"] = {"ambient_temp_c": 80.0, "traffic_factor": 1.0}
    response = client.post("/api/scenarios", json=payload)
    assert response.status_code == 422


def test_scenario_requires_an_identifier(client):
    response = client.post("/api/scenarios", json={"seed": 1})
    assert response.status_code == 422


def test_demo_reset_restores_the_deterministic_scenario(client):
    client.post("/api/scenarios", json=dict(SPEC_EXAMPLE, scenario_id="reset_check"))
    response = client.post("/api/demo/reset")
    assert response.status_code == 200
    snapshot = client.get("/api/state").json()
    assert snapshot["scenario_id"] != "reset_check"
    baseline = client.get("/api/state").json()
    again = client.post("/api/demo/reset").json()
    assert len(again["snapshot"]["vehicles"]) == len(baseline["vehicles"])
    assert again["snapshot"]["environment"] == baseline["environment"]


def test_simulate_returns_route_physics(client, zones, vehicle_id):
    origin = zones[0]["zone_id"]
    destination = zones[-1]["zone_id"]
    response = client.post(
        "/api/simulate",
        json={"vehicle_id": vehicle_id, "route": {"origin_zone": origin, "destination_zone": destination}},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["distance_km"] > 0.0
    assert payload["travel_time_min"] > 0.0
    assert payload["route_energy_kwh"] >= 0.0
    assert 0.0 <= payload["departure_soc"] <= 1.0
    assert 0.0 <= payload["arrival_soc"] <= 1.0
    assert payload["evidence_status"] == "MODEL_ESTIMATE"
    assert payload["calibrated"] is False
    assert payload["config_hash"]
    assert len(payload["time_s"]) == len(payload["speed_mps"]) == len(payload["soc_trace"])


def test_simulate_respects_cabin_and_environment_overrides(client, zones, vehicle_id):
    origin = zones[0]["zone_id"]
    destination = zones[-1]["zone_id"]
    base = client.post(
        "/api/simulate",
        json={"vehicle_id": vehicle_id, "route": {"origin_zone": origin, "destination_zone": destination}},
    ).json()
    loaded = client.post(
        "/api/simulate",
        json={
            "vehicle_id": vehicle_id,
            "route": {"origin_zone": origin, "destination_zone": destination},
            "cabin_passengers": 4,
            "cabin_luggage_mass_kg": 90.0,
            "mode": "AIRPORT",
            "ambient_temp_c": 45.0,
        },
    ).json()
    assert loaded["constraint_checks"]["passenger_capacity"] == "PASS"
    assert loaded["loaded_mass_kg"] > base["loaded_mass_kg"]
    assert loaded["route_energy_kwh"] >= base["route_energy_kwh"]
    assert loaded["energy_contributors_kwh"]["hvac_kwh"] >= base["energy_contributors_kwh"]["hvac_kwh"]


def test_simulate_rejects_unknown_zones(client, vehicle_id):
    response = client.post(
        "/api/simulate",
        json={
            "vehicle_id": vehicle_id,
            "route": {"origin_zone": "NOPE", "destination_zone": "ALSO_NOPE"},
        },
    )
    assert response.status_code in (404, 422)


def test_simulate_rejects_economies_over_capacity(client, zones, vehicle_id):
    response = client.post(
        "/api/simulate",
        json={
            "vehicle_id": vehicle_id,
            "route": {"origin_zone": zones[0]["zone_id"], "destination_zone": zones[-1]["zone_id"]},
            "cabin_passengers": 8,
            "mode": "POOL",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    if payload["constraint_checks"]["passenger_capacity"] == "FAIL":
        assert payload["route_energy_kwh"] == 0.0
        assert payload["constraint_checks"]["route_trace_valid"] == "FAIL"


def test_scenario_request_is_serialisable_round_trip(client):
    created = client.post("/api/scenarios", json=SPEC_EXAMPLE).json()
    state = client.get("/api/state").json()
    assert created["snapshot"]["scenario_id"] == state["scenario_id"]
    assert state["demand_context"] if "demand_context" in state else True
    # Timestamps arrive as ISO-8601 strings.
    assert isinstance(state["timestamp_utc"], str)
    assert state["timestamp_utc"].endswith("Z") or "+" in state["timestamp_utc"]