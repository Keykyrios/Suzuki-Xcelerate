"""Decision routes: optimize, run manifests, and run replay."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import Field
from starlette.concurrency import run_in_threadpool

from app.api.deps import (
    get_optimization_service,
    get_scenario_service,
    require_operator,
)
from app.data_access.manifests import ManifestStore
from app.data_access.repository import Repository
from app.config.loader import get_config
from app.schemas.common import StrictModel
from app.schemas.scenario import Scenario

router = APIRouter(prefix="/api", tags=["decisions"])


def _losses(values) -> list[float] | None:
    """Plain-JSON copy of the per-scenario loss vector."""
    if values is None:
        return None
    return [round(float(v), 2) for v in values]


def _baseline_losses(outcome) -> list[float]:
    """Baseline plan loss on the base scenario; no uncertainty draws are solved."""
    from app.optimization.dispatch import cost_weights, total_loss_inr

    weights = cost_weights(get_config())
    return [round(float(total_loss_inr(outcome.assignments, outcome.unserved, weights)), 2)]


class OptimizeRequest(StrictModel):
    """Optional body for POST /api/optimize."""

    run_id: str | None = Field(default=None, min_length=1, max_length=64)


@router.post("/optimize", dependencies=[Depends(require_operator)])
async def optimize(
    request: OptimizeRequest | None = None,
    optimization_service=Depends(get_optimization_service),
) -> dict[str, Any]:
    """Solve the dispatch/charging/powertrain decision problem.

    The solve runs in the threadpool so the request never blocks the event
    loop that serves camera/state ingestion and WebSocket streaming.
    """
    run_id = request.run_id if request else None
    result = await run_in_threadpool(lambda: optimization_service.optimize(run_id))
    decision = result.decision
    return {
        "run_id": decision.run_id,
        "scenario_id": decision.scenario_id,
        "status": decision.status.value,
        "decision": decision.model_dump(mode="json"),
        "explanation": optimization_service.explanation_for(decision.run_id),
        "baseline": {
            "metrics": decision.baseline_metrics.model_dump(mode="json"),
            "loss_inr": _baseline_losses(result.baseline_outcome),
            "unserved": list(result.baseline_outcome.unserved),
        },
        "optimized": {
            "metrics": decision.optimized_metrics.model_dump(mode="json"),
            "loss_inr": _losses(result.optimized_outcome.scenario_losses_inr),
            "unserved": list(result.optimized_outcome.unserved),
        },
        "recheck_passed": decision.recheck_passed,
    }


@router.get("/runs")
def list_runs() -> dict:
    """All recorded run ids."""
    return {"runs": ManifestStore().list_runs()}


@router.get("/runs/{run_id}")
def get_run(run_id: str, optimization_service=Depends(get_optimization_service)) -> dict:
    """Manifest, headline metrics, and decision explanation for one run."""
    try:
        manifest = ManifestStore().load(run_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown run id {run_id}") from None
    return {
        "manifest": manifest.model_dump(mode="json"),
        "explanation": optimization_service.explanation_for(run_id),
    }


@router.post("/runs/{run_id}/replay", dependencies=[Depends(require_operator)])
async def replay_run(
    run_id: str,
    optimization_service=Depends(get_optimization_service),
    scenario_service=Depends(get_scenario_service),
) -> dict:
    """Replay a stored run: reload its scenario and reproduce its decision."""
    try:
        manifest = ManifestStore().load(run_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown run id {run_id}") from None
    cfg = get_config()
    try:
        document = Repository(cfg.root).read_json_document(f"scenario_{manifest.scenario_id}")
    except (FileNotFoundError, ValueError):
        raise HTTPException(
            status_code=404,
            detail=f"stored scenario for run {run_id} not found",
        ) from None
    scenario = Scenario.model_validate(document)
    scenario_service.store.load_scenario(scenario, run_id=run_id)
    result = await run_in_threadpool(lambda: optimization_service.optimize(run_id))
    return {
        "run_id": run_id,
        "scenario_id": scenario.scenario_id,
        "manifest": manifest.model_dump(mode="json"),
        "decision": result.decision.model_dump(mode="json"),
        "explanation": optimization_service.explanation_for(run_id),
    }