"""WebSocket live-state protocol: full snapshot on connect, then versioned deltas.

Behaviour follows state_integrity.ws_snapshot_on_connect (full_snapshot_then_deltas).
Clients must drop any delta whose state_version is older than the last applied
one; a {"type":"resync"} message re-sends the full snapshot.
"""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.schemas.events import EventEnvelope, EventType
from app.services.state_store import get_state_store
from app.services.websocket_manager import (
    encode_envelope,
    get_websocket_manager,
    make_event,
)

router = APIRouter()


def _snapshot_event() -> str:
    snapshot = get_state_store().snapshot()
    event = make_event(
        EventType.STATE_SNAPSHOT,
        snapshot.scenario_id,
        snapshot.state_version,
        snapshot.model_dump(mode="json"),
        snapshot.run_id,
    )
    return encode_envelope(event)


async def _pump(queue: asyncio.Queue[str], websocket: WebSocket) -> None:
    try:
        while True:
            payload = await queue.get()
            await websocket.send_text(payload)
    except asyncio.CancelledError:
        raise


@router.websocket("/api/events")
async def events(websocket: WebSocket) -> None:
    """Stream STATE_SNAPSHOT on connect followed by versioned STATE_DELTA events."""
    await websocket.accept()
    manager = get_websocket_manager()
    store = get_state_store()
    sub = manager.subscribe()
    loop = asyncio.get_running_loop()

    # Store listeners fire on worker threads; hop onto the event loop thread.
    def listener(envelope: EventEnvelope) -> None:
        loop.call_soon_threadsafe(_enqueue, envelope)

    def _enqueue(envelope: EventEnvelope) -> None:
        asyncio.ensure_future(manager.deliver_to(sub.connection_id, envelope))

    unsubscribe = store.subscribe(listener)
    sender = asyncio.create_task(_pump(sub.queue, websocket))
    try:
        # Full snapshot first, per the configured reconnect policy.
        await websocket.send_text(_snapshot_event())
        while True:
            raw = await websocket.receive_text()
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(message, dict) and message.get("type") == "resync":
                await websocket.send_text(_snapshot_event())
    except WebSocketDisconnect:
        pass
    finally:
        unsubscribe()
        manager.unsubscribe(sub.connection_id)
        sender.cancel()
        try:
            await sender
        except asyncio.CancelledError:
            pass