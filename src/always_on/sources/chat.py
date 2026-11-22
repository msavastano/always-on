"""Direct user chat input source (CLI / local web dashboard)."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from always_on.models import ChatReply, IncomingEvent, new_id
from always_on.sources.base import EventSource


@dataclass
class PendingChatRequest:
    """In-flight chat event awaiting the daemon's response."""

    event: IncomingEvent
    future: asyncio.Future[ChatReply]


class ChatSource(EventSource):
    """Queue-backed source for direct interactive user chat messages."""

    name = "chat"

    def __init__(self) -> None:
        self._queue: asyncio.Queue[PendingChatRequest] = asyncio.Queue()
        self._pending_futures: dict[str, asyncio.Future[ChatReply]] = {}

    def submit(
        self,
        message: str,
        *,
        actor_id: str = "user",
        is_correction: bool = False,
    ) -> tuple[IncomingEvent, asyncio.Future[ChatReply]]:
        """Enqueue a user chat message and return a Future that resolves to ChatReply."""
        loop = asyncio.get_running_loop()
        turn_id = new_id("turn")
        event = IncomingEvent(
            source="chat",
            actor_id=actor_id,
            content=message,
            payload={"message": message, "turn_id": turn_id},
            priority="high",
            hard_override=True,
            is_correction=is_correction,
        )
        fut: asyncio.Future[ChatReply] = loop.create_future()
        self._pending_futures[event.id] = fut
        self._queue.put_nowait(PendingChatRequest(event=event, future=fut))
        return event, fut

    def resolve(self, event_id: str, reply: ChatReply) -> None:
        """Resolve the waiting caller's future with an agent-tagged ChatReply."""
        fut = self._pending_futures.pop(event_id, None)
        if fut is not None and not fut.done():
            fut.set_result(reply)

    def fail(self, event_id: str, exc: Exception) -> None:
        """Fail the waiting caller's future if an unrecoverable error occurs."""
        fut = self._pending_futures.pop(event_id, None)
        if fut is not None and not fut.done():
            fut.set_exception(exc)

    async def poll(self) -> list[IncomingEvent]:
        events: list[IncomingEvent] = []
        while not self._queue.empty():
            req = self._queue.get_nowait()
            events.append(req.event)
        return events
