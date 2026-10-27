"""FastAPI application: assembles routes, error mapping, and the live-state loop.

Start with: uvicorn app.main:app --reload
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from app import __version__
from app.api import routes_controls, routes_decisions, routes_scenarios, routes_vision, websocket
from app.config.loader import get_config
from app.services.scenario_service import ScenarioService
from app.services.state_store import get_state_store


async def _staleness_ticker() -> None:
    """Mark the current recommendation stale once it outlives the display policy."""
    cfg = get_config()
    max_age_s = float(cfg.op("state_integrity.stale_after_s", 120))
    store = get_state_store()
    while True:
        await asyncio.sleep(min(15.0, max(max_age_s / 2.0, 1.0)))
        decision_at = store.last_decision_wall_utc
        if decision_at is None or store.recommendation_stale:
            continue
        age_s = (datetime.now(timezone.utc) - decision_at).total_seconds()
        if age_s > max_age_s:
            store.set_stale(True, reason=f"recommendation older than {max_age_s:.0f}s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    store = get_state_store()
    if store.scenario is None:
        # Bootstrap the deterministic demo state so /api/state is usable on boot.
        ScenarioService(store=store, cfg=get_config()).reset_demo()
    ticker = asyncio.create_task(_staleness_ticker())
    try:
        yield
    finally:
        ticker.cancel()
        try:
            await ticker
        except asyncio.CancelledError:
            pass


def create_app() -> FastAPI:
    app = FastAPI(
        title="NEXUS-7 Mobility Demonstrator",
        version=__version__,
        description=(
            "Fleet simulation, dispatch optimization, and live digital twin. "
            "All outputs are simulation estimates; no vehicle hardware is commanded."
        ),
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:4173",
            "http://127.0.0.1:4173",
        ],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # -- error mapping -----------------------------------------------------
    @app.exception_handler(RuntimeError)
    async def _runtime_error(request: Request, exc: RuntimeError) -> JSONResponse:
        if "no scenario" in str(exc):
            return JSONResponse(status_code=409, content={"detail": str(exc)})
        return JSONResponse(status_code=500, content={"detail": str(exc)})

    @app.exception_handler(KeyError)
    async def _key_error(request: Request, exc: KeyError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": f"not found: {exc.args[0]}"})

    @app.exception_handler(FileNotFoundError)
    async def _file_error(request: Request, exc: FileNotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(ValueError)
    async def _value_error(request: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    # -- routes ------------------------------------------------------------
    app.include_router(routes_scenarios.router)
    app.include_router(routes_decisions.router)
    app.include_router(routes_controls.router)
    app.include_router(routes_vision.router)
    app.include_router(websocket.router)
    return app


app = create_app()