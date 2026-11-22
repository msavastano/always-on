"""Internal clock / heartbeat source for smooth temporal-pressure check-ins."""

from __future__ import annotations

import time

from always_on.models import IncomingEvent
from always_on.sources.base import EventSource


class HeartbeatSource(EventSource):
    """Periodically emits zero-token clock ticks so temporal pressure can wake the agent."""

    name = "heartbeat"

    def __init__(self, emit_interval_seconds: int = 60) -> None:
        self.emit_interval_seconds = max(5, int(emit_interval_seconds))
        self._last_emitted_monotonic: float = time.monotonic()

    async def poll(self) -> list[IncomingEvent]:
        now_m = time.monotonic()
        if (now_m - self._last_emitted_monotonic) < self.emit_interval_seconds:
            return []
        self._last_emitted_monotonic = now_m
        return [
            IncomingEvent(
                source="heartbeat",
                actor_id="clock",
                content="Periodic internal heartbeat check-in.",
                payload={"kind": "heartbeat"},
                priority="normal",
                hard_override=False,
            )
        ]
