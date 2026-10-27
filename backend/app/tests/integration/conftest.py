"""Shared client fixture for API-level tests."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """A TestClient with the deterministic demo scenario loaded."""
    from app.main import app

    with TestClient(app) as test_client:
        response = test_client.post("/api/demo/reset")
        assert response.status_code == 200, response.text
        yield test_client


@pytest.fixture
def zones(client) -> list[dict]:
    snapshot = client.get("/api/state").json()
    return snapshot["graph"]["zones"]


@pytest.fixture
def vehicle_id(client) -> str:
    return client.get("/api/state").json()["vehicles"][0]["vehicle_id"]


@pytest.fixture
def charger_id(client) -> str:
    return client.get("/api/state").json()["chargers"][0]["charger_id"]


@pytest.fixture
def solved(client):
    """One optimization run on the freshly reset demo scenario."""
    response = client.post("/api/optimize", json={})
    assert response.status_code == 200, response.text
    return response.json()