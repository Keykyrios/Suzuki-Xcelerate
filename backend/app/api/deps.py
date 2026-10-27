"""Shared FastAPI dependencies: service access and the local operator gate.

State-changing endpoints check an optional operator token configured via the
operating policy (environment variable name in state_integrity.operator_token_env).
When the variable is not set, the gate is open - this is a local demonstrator,
not an internet-facing service.
"""

from __future__ import annotations

import os

from fastapi import Header, HTTPException

from app.config.loader import get_config
from app.services.economics_service import EconomicsService
from app.services.optimization_service import OptimizationService
from app.services.run_service import RunService
from app.services.scenario_service import ScenarioService
from app.services.simulation_service import SimulationService
from app.services.state_store import StateStore, get_state_store
from app.services.vision_service import VisionService, get_vision_service as _get_vision_service


def get_scenario_service() -> ScenarioService:
    return ScenarioService()


def get_simulation_service() -> SimulationService:
    return SimulationService()


def get_optimization_service() -> OptimizationService:
    return OptimizationService()


def get_economics_service() -> EconomicsService:
    return EconomicsService(get_config(), get_state_store())


def get_run_service() -> RunService:
    return RunService(get_config(), get_state_store())


def get_vision_service() -> VisionService:
    return _get_vision_service(get_config(), get_state_store())


def current_store() -> StateStore:
    return get_state_store()


def require_operator(
    x_operator_token: str | None = Header(default=None, alias="X-Operator-Token"),
) -> None:
    """Permit state-changing requests locally or with the configured token."""
    env_name = str(get_config().op("state_integrity.operator_token_env"))
    expected = os.environ.get(env_name) if env_name else None
    if expected and x_operator_token != expected:
        raise HTTPException(
            status_code=403,
            detail=f"state-changing endpoint requires the operator token from {env_name}",
        )