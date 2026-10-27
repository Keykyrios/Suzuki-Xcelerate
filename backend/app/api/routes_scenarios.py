"""Meta and scenario routes: health, config, state, scenarios, simulate, reset."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import Field
from starlette.concurrency import run_in_threadpool

from app import __version__
from app.api.deps import (
    current_store,
    get_economics_service,
    get_scenario_service,
    get_simulation_service,
    require_operator,
)
from app.config.loader import get_config
from app.data_access.manifests import model_versions
from app.schemas.common import CabinMode, StrictModel
from app.schemas.scenario import RouteSpec, ScenarioSpec
from app.services.state_store import StateStore

router = APIRouter(prefix="/api", tags=["meta", "scenarios"])


class SimulateRequest(StrictModel):
    """Body for POST /api/simulate: one vehicle mission at display-grade physics."""

    vehicle_id: str = Field(min_length=1)
    route: RouteSpec
    cabin_passengers: int | None = Field(default=None, ge=0, le=8)
    cabin_luggage_mass_kg: float | None = Field(default=None, ge=0.0, le=400.0)
    mode: CabinMode | None = None
    ambient_temp_c: float | None = Field(default=None, ge=-10.0, le=55.0)
    traffic_factor: float | None = Field(default=None, gt=0.0, le=3.0)
    road_grade_percent: float | None = Field(default=None, ge=-12.0, le=12.0)


@router.get("/health")
def health() -> dict:
    """Service health, version, and model/solver availability."""
    cfg = get_config()
    store = current_store()
    try:
        import ortools

        solver = {"name": "CP-SAT / scipy", "version": getattr(ortools, "__version__", "?"), "available": True}
    except Exception:  # noqa: BLE001 - solver absence must be reported, not raised
        solver = {"name": "CP-SAT / scipy", "version": "unavailable", "available": False}
    return {
        "status": "ok",
        "version": __version__,
        "config_hash": cfg.hash,
        "config_versions": cfg.versions(),
        "model_versions": model_versions(),
        "solver": solver,
        "state_version": store.state_version,
        "scenario_loaded": store.scenario is not None,
        "recommendation_stale": store.recommendation_stale,
    }


@router.get("/config")
def config() -> dict:
    """Active configurable parameters with their evidence-status labels."""
    cfg = get_config()
    return {
        "config_hash": cfg.hash,
        "versions": cfg.versions(),
        "parameters": cfg.snapshot(),
    }


@router.get("/state")
def state(store: StateStore = Depends(current_store)) -> dict:
    """Full canonical current state snapshot."""
    return store.snapshot().model_dump(mode="json")


@router.post("/scenarios", status_code=201, dependencies=[Depends(require_operator)])
def create_scenario(spec: ScenarioSpec, scenario_service=Depends(get_scenario_service)) -> dict:
    """Validate and create/load a scenario."""
    scenario = scenario_service.create(spec)
    store = scenario_service.store
    return {
        "scenario_id": scenario.scenario_id,
        "state_version": store.state_version,
        "vehicles": len(scenario.vehicles),
        "chargers": len(scenario.chargers),
        "requests": len(scenario.requests),
        "snapshot": store.snapshot().model_dump(mode="json"),
    }


@router.post("/simulate")
async def simulate(
    request: SimulateRequest,
    simulation_service=Depends(get_simulation_service),
) -> dict:
    """Run vehicle/route physics for a mission; pure computation, no state change."""
    result = await run_in_threadpool(
        lambda: simulation_service.simulate_route(
            request.vehicle_id,
            request.route,
            cabin_passengers=request.cabin_passengers,
            cabin_luggage_mass_kg=request.cabin_luggage_mass_kg,
            mode=request.mode,
            ambient_temp_c=request.ambient_temp_c,
            traffic_factor=request.traffic_factor,
            road_grade_override_percent=request.road_grade_percent,
        )
    )
    return result.model_dump(mode="json")


@router.get("/economics")
def economics(
    hour_of_day: int = 12,
    economics_service=Depends(get_economics_service),
) -> dict:
    """Configured economics with provenance: cost structure, tariffs, TCO, day.

    The fleet-day report only appears once a decision has been recorded; until
    then the response states why rather than inventing a result.
    """
    service = economics_service
    report: dict = {
        "currency": service.currency(),
        "cost_structure": service.cost_structure_report(),
        "tariff": service.tariff_report(hour_of_day=hour_of_day),
        "reference_case": service.reference_case_report(),
    }
    try:
        report["tco"] = service.tco_report(hour_of_day=hour_of_day)
    except ValueError as exc:
        report["tco"] = {"available": False, "reason": str(exc)}
    report["fleet_day"] = service.fleet_day_report()
    return report


@router.post("/demo/reset", dependencies=[Depends(require_operator)])
def reset_demo(scenario_service=Depends(get_scenario_service)) -> dict:
    """Restore the configured deterministic demo state."""
    scenario = scenario_service.reset_demo()
    store = scenario_service.store
    return {
        "scenario_id": scenario.scenario_id,
        "state_version": store.state_version,
        "snapshot": store.snapshot().model_dump(mode="json"),
    }