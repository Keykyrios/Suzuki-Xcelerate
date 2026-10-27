"""Throwaway smoke probe for the API layer; not part of the test suite."""
from fastapi.testclient import TestClient

from app.main import app

with TestClient(app) as client:
    for path in ["/api/health", "/api/config", "/api/state"]:
        r = client.get(path)
        print(path, r.status_code, list(r.json())[:6] if r.status_code == 200 else r.text[:300])
        assert r.status_code == 200, path

    r = client.post("/api/demo/reset")
    print("/api/demo/reset", r.status_code)
    assert r.status_code == 200, r.text[:500]

    s = client.get("/api/state").json()
    vid = s["vehicles"][0]["vehicle_id"]
    zone = s["graph"]["zones"][0]["zone_id"]
    dest = s["graph"]["zones"][-1]["zone_id"]

    r = client.post("/api/simulate", json={"vehicle_id": vid,
                                           "route": {"origin_zone": zone, "destination_zone": dest}})
    print("/api/simulate", r.status_code)
    assert r.status_code == 200, r.text[:500]

    r = client.post("/api/optimize", json={})
    print("/api/optimize", r.status_code, r.json().get("status") if r.status_code == 200 else r.text[:800])
    assert r.status_code == 200, r.text[:800]

    run_id = r.json()["run_id"]
    r = client.get(f"/api/runs/{run_id}")
    print("GET /api/runs", r.status_code)
    assert r.status_code == 200, r.text[:400]
    r = client.post(f"/api/runs/{run_id}/replay")
    print("POST replay", r.status_code)
    assert r.status_code == 200, r.text[:500]

    cid = s["chargers"][0]["charger_id"]
    r = client.post(f"/api/chargers/{cid}/status", json={"status": "OFFLINE", "queue_length": 0})
    print("charger status", r.status_code)
    assert r.status_code == 200, r.text[:400]

    r = client.post(f"/api/vehicles/{vid}/state", json={"soc": 0.55})
    print("vehicle state", r.status_code)
    assert r.status_code == 200, r.text[:400]

    r = client.post("/api/simulate", json={"vehicle_id": "NOPE",
                                           "route": {"origin_zone": zone, "destination_zone": dest}})
    print("bad simulate ->", r.status_code)
    assert r.status_code in (404, 422), r.text[:300]

    r = client.post("/api/simulation/actions", json={
        "action_type": "CABIN_MODE_CHANGE", "target_type": "vehicle",
        "target_id": vid, "requested_state": {"mode": "AIRPORT"}})
    print("sim action", r.status_code, r.json().get("hardware_commanded") if r.status_code == 200 else r.text[:400])
    assert r.status_code == 200 and r.json()["hardware_commanded"] is False

    # WebSocket: snapshot first, then a delta after a mutation.
    with client.websocket_connect("/api/events") as ws:
        first = ws.receive_json()
        assert first["event_type"] == "STATE_SNAPSHOT", first["event_type"]
        client.post(f"/api/vehicles/{vid}/state", json={"soc": 0.5})
        import json as _json
        raw = ws.receive_text()
        delta = _json.loads(raw)
        print("ws first:", first["event_type"], "then:", delta["event_type"], "v", delta["state_version"])
        ws.send_text(_json.dumps({"type": "resync"}))
        again = ws.receive_json()
        assert again["event_type"] == "STATE_SNAPSHOT"

print("API smoke OK")