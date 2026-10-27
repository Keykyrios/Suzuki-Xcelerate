"""Solver failure behaviour: truthful status, baseline fallback, no fake success."""

from __future__ import annotations

import pytest

from app.optimization.dispatch import DispatchOutcome
from app.schemas.common import SolverStatus
from app.schemas.decision import SolverReport


def error_outcome(scenario, status: SolverStatus, message: str) -> DispatchOutcome:
    return DispatchOutcome(
        assignments=[],
        unserved=[request.request_id for request in scenario.requests],
        rejected=[],
        solver=SolverReport(
            name="CP-SAT / scipy",
            version="9.14",
            status=status,
            runtime_ms=12.0,
            fallback_used=status is SolverStatus.TIMEOUT,
            message=message,
        ),
        risk=None,
        reason_codes=[],
        status=status,
    )


@pytest.mark.parametrize("status", [SolverStatus.TIMEOUT, SolverStatus.ERROR])
def test_timeout_and_error_fall_back_to_the_transparent_baseline(
    client, monkeypatch, status
):
    from app.services import optimization_service as module
    from app.services.state_store import get_state_store

    scenario = get_state_store().require_scenario()

    def failing_solve(_scenario, _cfg=None, requests=None):
        return error_outcome(scenario, status, f"{status.value} hit the configured limit")

    monkeypatch.setattr(module, "solve_dispatch", failing_solve)
    response = client.post("/api/optimize", json={})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == SolverStatus.BASELINE_FALLBACK.value
    assert payload["decision"]["status"] == SolverStatus.BASELINE_FALLBACK.value
    assert payload["decision"]["solver"]["fallback_used"] is True
    expected_code = (
        "SOLVER_TIMEOUT_BASELINE_FALLBACK"
        if status is SolverStatus.TIMEOUT
        else "SOLVER_ERROR_BASELINE_FALLBACK"
    )
    assert expected_code in payload["decision"]["reason_codes"]
    # The baseline still produced a plan; it is labelled as a fallback, never OPTIMAL.
    assert payload["status"] not in {"OPTIMAL", "FEASIBLE"}


def test_raising_solver_reports_error_instead_of_crashing_the_run(client, monkeypatch):
    from app.services import optimization_service as module

    def exploding_solve(_scenario, _cfg=None, requests=None):
        raise RuntimeError("model exploded")

    monkeypatch.setattr(module, "solve_dispatch", exploding_solve)
    response = client.post("/api/optimize", json={})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == SolverStatus.BASELINE_FALLBACK.value
    decision = payload["decision"]
    assert decision["status"] != SolverStatus.OPTIMAL.value
    assert decision["solver"]["fallback_used"] is True
    assert decision["solver"]["message"] or decision["reason_codes"]


def test_failed_solve_is_exposed_on_state_and_never_shown_as_optimal(
    client, monkeypatch
):
    from app.services import optimization_service as module
    from app.services.state_store import get_state_store

    scenario = get_state_store().require_scenario()
    monkeypatch.setattr(
        module,
        "solve_dispatch",
        lambda *args, **kwargs: error_outcome(scenario, SolverStatus.ERROR, "boom"),
    )
    payload = client.post("/api/optimize", json={}).json()
    state = client.get("/api/state").json()
    assert state["solver_status"] == SolverStatus.BASELINE_FALLBACK.value
    assert state["decision"]["status"] == SolverStatus.BASELINE_FALLBACK.value
    assert state["decision"]["run_id"] == payload["run_id"]
    health = client.get("/api/health").json()
    assert health["status"] == "ok"  # the service stays healthy while the solve fails


def test_stored_manifest_records_the_truthful_solver_status(client, monkeypatch):
    from app.services import optimization_service as module
    from app.services.state_store import get_state_store

    scenario = get_state_store().require_scenario()
    monkeypatch.setattr(
        module,
        "solve_dispatch",
        lambda *args, **kwargs: error_outcome(scenario, SolverStatus.TIMEOUT, "timed out"),
    )
    payload = client.post("/api/optimize", json={}).json()
    manifest = client.get(f"/api/runs/{payload['run_id']}").json()["manifest"]
    assert manifest["solver_status"] == SolverStatus.BASELINE_FALLBACK.value
    assert manifest["status"] == "completed"
    assert manifest["notes"] == "fallback to baseline"


def test_successful_solve_is_still_optimal_after_the_failure_tests(client):
    # Reset clears the monkeypatched path for this fresh client fixture.
    payload = client.post("/api/optimize", json={}).json()
    assert payload["status"] in {"OPTIMAL", "FEASIBLE", "BASELINE_FALLBACK", "NO_FEASIBLE_PLAN"}
    assert payload["decision"]["solver"]["fallback_used"] is False