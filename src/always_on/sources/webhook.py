"""HTTP webhook event source."""

from __future__ import annotations

import asyncio
from typing import Any, Literal

from always_on.models import IncomingEvent
from always_on.sources.base import EventSource


class WebhookSource(EventSource):
    """Queue-backed source populated by POST /api/events/webhook."""

    name = "webhook"

    def __init__(self) -> None:
        self._queue: asyncio.Queue[IncomingEvent] = asyncio.Queue()

    def push(
        self,
        *,
        content: str,
        actor_id: str = "webhook",
        payload: dict[str, Any] | None = None,
        priority: Literal["normal", "high"] = "normal",
        hard_override: bool = False,
    ) -> IncomingEvent:
        """Push an external webhook payload into the daemon event queue."""
        event = IncomingEvent(
            source="webhook",
            actor_id=actor_id,
            content=content,
            payload=payload or {},
            priority=priority,
            hard_override=hard_override,
        )
        self._queue.put_nowait(event)
        return event

    async def poll(self) -> list[IncomingEvent]:
        events: list[IncomingEvent] = []
        while not self._queue.empty():
            events.append(self._queue.get_nowait())
        return events
