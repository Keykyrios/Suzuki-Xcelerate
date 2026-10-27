"""WebSocket protocol layer: envelope encoding and subscriber fan-out.

The manager keeps one broadcast queue per connection and is transport-agnostic;
FastAPI handlers in app.api only route websockets in and events out.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass, field

from app.schemas.events import EventEnvelope, EventType


def encode_envelope(event: EventEnvelope) -> str:
    return event.model_dump_json()


@dataclass(slots=True)
class Subscriber:
    queue: asyncio.Queue[str]
    connection_id: str
    scenario_filter: str | None = None


@dataclass
class WebSocketManager:
    """Tracks subscribers and dispatches event envelopes to matching queues."""

    _subscribers: dict[str, Subscriber] = field(default_factory=dict)
    _dropped: int = 0
    max_queue: int = 1000

    def subscribe(self, scenario_filter: str | None = None) -> Subscriber:
        sub = Subscriber(queue=asyncio.Queue(maxsize=self.max_queue), connection_id=uuid.uuid4().hex, scenario_filter=scenario_filter)
        self._subscribers[sub.connection_id] = sub
        return sub

    def unsubscribe(self, connection_id: str) -> None:
        self._subscribers.pop(connection_id, None)

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    @property
    def dropped_count(self) -> int:
        return self._dropped

    async def dispatch(self, event: EventEnvelope) -> int:
        payload = encode_envelope(event)
        delivered = 0
        for sub in list(self._subscribers.values()):
            if sub.scenario_filter and sub.scenario_filter != event.scenario_id:
                continue
            try:
                sub.queue.put_nowait(payload)
                delivered += 1
            except asyncio.QueueFull:
                self._dropped += 1
        return delivered

    async def deliver_to(self, connection_id: str, event: EventEnvelope) -> bool:
        sub = self._subscribers.get(connection_id)
        if sub is None:
            return False
        try:
            sub.queue.put_nowait(encode_envelope(event))
            return True
        except asyncio.QueueFull:
            self._dropped += 1
            return False


def make_event(
    event_type: EventType,
    scenario_id: str,
    state_version: int,
    payload: dict,
    run_id: str | None = None,
) -> EventEnvelope:
    from app.schemas.common import utcnow

    return EventEnvelope(
        event_id=uuid.uuid4().hex,
        event_type=event_type,
        state_version=state_version,
        timestamp_utc=utcnow(),
        scenario_id=scenario_id,
        run_id=run_id,
        payload=payload,
    )


_manager: WebSocketManager | None = None


def get_websocket_manager() -> WebSocketManager:
    global _manager
    if _manager is None:
        _manager = WebSocketManager()
    return _manager