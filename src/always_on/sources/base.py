"""Base EventSource protocol and EventIngestor (self-trigger + deduplication filter)."""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from always_on.config import Settings
from always_on.db.store import AgentStore
from always_on.models import IncomingEvent


class EventSource(ABC):
    """Protocol for pluggable input sources feeding the outer event loop."""

    name: str = "source"

    async def start(self) -> None:
        """Optional lifecycle hook called when the daemon starts."""

    async def stop(self) -> None:
        """Optional lifecycle hook called when the daemon stops."""

    @abstractmethod
    async def poll(self) -> list[IncomingEvent]:
        """Non-blocking poll returning any new events since the previous tick."""


@dataclass
class IngestionDecision:
    """Result of pre-embedding ingestion filtering."""

    accepted: bool
    drop_reason: Literal["self_trigger", "duplicate_cooldown"] | None = None
    payload_hash: str | None = None


def compute_payload_hash(event: IncomingEvent) -> str:
    """Canonical SHA-256 of raw event payload (excluding volatile ID/timestamp)."""
    canonical = json.dumps(
        {
            "source": event.source,
            "actor_id": event.actor_id,
            "content": event.content,
            "payload": event.payload,
        },
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class EventIngestor:
    """Enforces self-trigger prevention and rolling deduplication BEFORE embeddings."""

    def __init__(self, store: AgentStore, settings: Settings) -> None:
        self.store = store
        self.settings = settings
        self.dropped_self_triggers: int = 0
        self.dropped_duplicates: int = 0

    async def evaluate(
        self, event: IncomingEvent, *, now: datetime | None = None
    ) -> IngestionDecision:
        """Filter out agent self-triggers and duplicate raw payloads within cooldown."""
        # 1. Self-Trigger Prevention: NEVER process outputs emitted by the agent itself
        if event.actor_id == "agent":
            self.dropped_self_triggers += 1
            return IngestionDecision(accepted=False, drop_reason="self_trigger")

        # 2. Heartbeat ticks evaluate temporal pressure every interval; skip payload dedup
        if event.source == "heartbeat":
            return IngestionDecision(accepted=True)

        # 3. Rolling Deduplication Cooldown BEFORE computing embeddings
        payload_hash = compute_payload_hash(event)
        is_dup = await self.store.is_duplicate_payload(
            payload_hash,
            cooldown_seconds=self.settings.dedup_cooldown_seconds,
            now=now,
        )
        if is_dup:
            self.dropped_duplicates += 1
            return IngestionDecision(
                accepted=False,
                drop_reason="duplicate_cooldown",
                payload_hash=payload_hash,
            )

        return IngestionDecision(accepted=True, payload_hash=payload_hash)
