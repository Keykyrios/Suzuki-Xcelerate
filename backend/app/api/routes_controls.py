"""Control routes: charger status, vehicle state, and simulated actions.

Every endpoint here mutates only canonical simulated state. Nothing in this
module (or anywhere in the API layer) can command vehicle hardware: each
applied action returns a SimulatedAction record with hardware_commanded=False.
"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Body, Depends, HTTPException
from pydantic import Field

from app.api.deps import require_operator, get_scenario_service, get_simulation_service
from app.schemas.common import ChargerStatus, StrictModel
from app.schemas.events import SimulatedActionType
from app.schemas.vehicle import VehicleState

router = APIRouter(prefix="/api", tags=["controls"])


class ChargerStatusRequest(StrictModel):
    status: ChargerStatus
    queue_length: int | None = Field(default=None, ge=0, le=50)


class SimulatedActionRequest(StrictModel):
    action_type: SimulatedActionType
    target_type: Literal["vehicle", "charger", "cabin"]
    target_id: str = Field(min_length=1)
    requested_state: dict[str, Any] = Field(default_factory=dict)
    duration_s: float = Field(default=0.0, ge=0.0)
    run_id: str | None = None
    originating_decision_id: str | None = None


@router.post("/chargers/{charger_id}/status", dependencies=[Depends(require_operator)])
def set_charger_status(
    charger_id: str,
    request: ChargerStatusRequest,
    scenario_service=Depends(get_scenario_service),
) -> dict:
    """Change simulated charger availability/queue state; invalidates decisions."""
    try:
        scenario = scenario_service.set_charger_status(
            charger_id, request.status, queue_length=request.queue_length
        )
    except StopIteration:
        raise HTTPException(status_code=404, detail=f"unknown charger {charger_id}") from None
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    charger = scenario.charger(charger_id)
    return {
        "charger": charger.model_dump(mode="json"),
        "state_version": scenario_service.store.state_version,
        "recommendation_stale": scenario_service.store.recommendation_stale,
    }


@router.post("/vehicles/{vehicle_id}/state", dependencies=[Depends(require_operator)])
def set_vehicle_state(
    vehicle_id: str,
    updates: dict[str, Any] = Body(...),
    scenario_service=Depends(get_scenario_service),
) -> dict:
    """Update permitted simulated vehicle state; values are validated first."""
    if not updates:
        raise HTTPException(status_code=422, detail="no updates supplied")
    try:
        current = scenario_service.store.require_scenario().vehicle(vehicle_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown vehicle {vehicle_id}") from None
    # Merge and re-validate the full vehicle so field constraints (units, ranges,
    # enums) are enforced before anything reaches canonical state.
    candidate = current.model_dump(mode="json")
    candidate.update(updates)
    if "state_version" in candidate:
        candidate["state_version"] = current.state_version + 1
    try:
        VehicleState.model_validate(candidate)
        vehicle = scenario_service.update_vehicle_state(vehicle_id, updates)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown vehicle {vehicle_id}") from None
    return {
        "vehicle": vehicle.model_dump(mode="json"),
        "state_version": scenario_service.store.state_version,
        "recommendation_stale": scenario_service.store.recommendation_stale,
    }


@router.post("/simulation/actions", dependencies=[Depends(require_operator)])
def apply_simulation_action(
    request: SimulatedActionRequest,
    simulation_service=Depends(get_simulation_service),
) -> dict:
    """Apply a validated virtual action to simulated state.

    This endpoint never touches hardware: the emitted SimulatedAction always
    carries hardware_commanded=False and status=SIMULATED.
    """
    try:
        action = simulation_service.apply_action(
            action_type=request.action_type,
            target_type=request.target_type,
            target_id=request.target_id,
            requested_state=request.requested_state,
            run_id=request.run_id,
            duration_s=request.duration_s,
            originating_decision_id=request.originating_decision_id,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"unknown target {exc.args[0]}") from None
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    return {
        "action": action.model_dump(mode="json"),
        "hardware_commanded": False,
        "state_version": simulation_service.store.state_version,
    }