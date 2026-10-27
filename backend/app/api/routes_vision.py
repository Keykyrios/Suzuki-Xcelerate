"""Vision routes: cabin capture, manual override, and camera availability."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import Field
from starlette.concurrency import run_in_threadpool

from app.api.deps import get_vision_service, require_operator
from app.schemas.common import CabinMode, StrictModel

router = APIRouter(prefix="/api/vision", tags=["vision"])


class ManualOverrideRequest(StrictModel):
    mode: CabinMode
    passenger_count: int = Field(ge=0, le=8)
    small_luggage: int = Field(default=0, ge=0, le=12)
    large_luggage: int = Field(default=0, ge=0, le=12)


class CameraToggleRequest(StrictModel):
    enabled: bool


class StreamFrameRequest(StrictModel):
    frame_jpeg_base64: str = Field(min_length=8, max_length=8_000_000)


@router.get("/state")
def vision_state(vision_service=Depends(get_vision_service)) -> dict:
    """Camera availability, detector status, and per-vehicle cabin states."""
    return vision_service.status()


@router.post("/capture/{vehicle_id}", dependencies=[Depends(require_operator)])
def capture_cabin(vehicle_id: str, vision_service=Depends(get_vision_service)) -> dict:
    """Run one debounced camera capture for a vehicle; frames stay in memory."""
    try:
        return vision_service.capture(vehicle_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown vehicle {vehicle_id}") from None
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None


@router.post("/override/{vehicle_id}", dependencies=[Depends(require_operator)])
def override_cabin(
    vehicle_id: str,
    request: ManualOverrideRequest,
    vision_service=Depends(get_vision_service),
) -> dict:
    """Operator-entered cabin state; supersedes camera readings."""
    try:
        return vision_service.manual_override(
            vehicle_id,
            mode=request.mode,
            passenger_count=request.passenger_count,
            small_luggage=request.small_luggage,
            large_luggage=request.large_luggage,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown vehicle {vehicle_id}") from None


@router.post("/stream-frame/{vehicle_id}", dependencies=[Depends(require_operator)])
async def stream_frame(
    vehicle_id: str,
    request: StreamFrameRequest,
    vision_service=Depends(get_vision_service),
) -> dict:
    """One live camera frame through detection; returned with boxes and state."""
    raw = request.frame_jpeg_base64.encode("ascii")
    try:
        return await run_in_threadpool(
            lambda: vision_service.process_frame(vehicle_id, raw)
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown vehicle {vehicle_id}") from None
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None


@router.post("/camera", dependencies=[Depends(require_operator)])
def set_camera(
    request: CameraToggleRequest, vision_service=Depends(get_vision_service)
) -> dict:
    """Enable or disable the camera path; the last state is retained when off."""
    return vision_service.set_camera_enabled(request.enabled)
