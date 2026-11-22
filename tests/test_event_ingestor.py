"""Unit tests for EventIngestor: self-trigger prevention and rolling dedup cooldown."""

from __future__ import annotations

from datetime import timedelta

import pytest

from always_on.config import Settings
from always_on.db.store import AgentStore
from always_on.models import IncomingEvent, utc_now
from always_on.sources.base import EventIngestor


@pytest.mark.asyncio
async def test_self_trigger_prevention_drops_agent_events(
    store: AgentStore, test_settings: Settings
) -> None:
    ingestor = EventIngestor(store, test_settings)
    agent_evt = IncomingEvent(
        source="webhook",
        actor_id="agent",
        content="Output emitted by the agent itself",
    )
    decision = await ingestor.evaluate(agent_evt)
    assert decision.accepted is False
    assert decision.drop_reason == "self_trigger"
    assert ingestor.dropped_self_triggers == 1


@pytest.mark.asyncio
async def test_deduplication_cooldown_drops_identical_payloads_before_embeddings(
    store: AgentStore, test_settings: Settings
) -> None:
    test_settings.dedup_cooldown_seconds = 300
    ingestor = EventIngestor(store, test_settings)
    t0 = utc_now()

    evt1 = IncomingEvent(
        source="webhook",
        actor_id="github",
        content="Push to main branch",
        payload={"repo": "always-on", "sha": "abc123"},
    )
    evt2_dup = IncomingEvent(
        source="webhook",
        actor_id="github",
        content="Push to main branch",
        payload={"sha": "abc123", "repo": "always-on"},  # Same payload, different dict key order
    )

    d1 = await ingestor.evaluate(evt1, now=t0)
    assert d1.accepted is True

    # Arriving 2 minutes later (< 5m cooldown) -> dropped as duplicate!
    d2 = await ingestor.evaluate(evt2_dup, now=t0 + timedelta(seconds=120))
    assert d2.accepted is False
    assert d2.drop_reason == "duplicate_cooldown"
    assert ingestor.dropped_duplicates == 1

    # Arriving 6 minutes later (> 5m cooldown) -> accepted again!
    d3 = await ingestor.evaluate(evt2_dup, now=t0 + timedelta(seconds=360))
    assert d3.accepted is True
