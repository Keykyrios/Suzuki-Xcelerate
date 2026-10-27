"""Operator chat hub: a flat in-memory sequence for demo observations and notes.

Messages are operational annotations shared between connected operators. They
are capped in memory, ordered by an append-only sequence, and are not persisted
or used for free-text generation anywhere else in the system.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass

from app.config.loader import ConfigBundle, get_config


@dataclass(frozen=True, slots=True)
class ChatMessage:
    seq: int
    author: str
    text: str
    system: bool = False


class ChatHub:
    """Linear message log; all reads and writes are lock-guarded."""

    def __init__(self, max_messages: int = 500):
        self._lock = threading.Lock()
        self._messages: deque[ChatMessage] = deque(maxlen=max_messages)
        self._seq = 0

    @property
    def latest_seq(self) -> int:
        with self._lock:
            return self._seq

    def post(self, author: str, text: str, *, system: bool = False) -> ChatMessage:
        text = text.strip()
        if not text:
            raise ValueError("chat message text must be non-empty")
        with self._lock:
            self._seq += 1
            msg = ChatMessage(seq=self._seq, author=author, text=text, system=system)
            self._messages.append(msg)
            return msg

    def system_notice(self, text: str) -> ChatMessage:
        return self.post("system", text, system=True)

    def messages_since(self, after_seq: int = 0) -> list[ChatMessage]:
        with self._lock:
            return [m for m in self._messages if m.seq > after_seq]

    def recent(self, limit: int = 50) -> list[ChatMessage]:
        with self._lock:
            items = list(self._messages)
        return items[-limit:]

    def clear(self) -> None:
        with self._lock:
            self._messages.clear()


_hub: ChatHub | None = None


def get_chat_hub(cfg: ConfigBundle | None = None) -> ChatHub:
    global _hub
    if _hub is None:
        bundle = cfg or get_config()
        limit = int(bundle.registries["operating_policy"].value("chat.max_messages", 500))
        _hub = ChatHub(max_messages=limit)
    return _hub