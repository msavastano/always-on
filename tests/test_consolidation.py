"""Unit tests for transactional sleep_and_consolidate and deterministic guardrails."""

from __future__ import annotations

import json

import pytest

from always_on.cognitive.consolidation import sleep_and_consolidate
from always_on.config import Settings
from always_on.db.store import AgentStore
from always_on.embeddings.provider import HashEmbeddingProvider
from always_on.models import FastBufferEpisode, MemoryItem
from always_on.safety.circuit_breaker import AgentCircuitBreaker
from tests.conftest import FakeBlock, FakeMessage, make_mock_client


async def _seed_episode(
    store: AgentStore,
    embedder: HashEmbeddingProvider,
    ep_id: str,
    summary: str,
) -> FastBufferEpisode:
    ep = FastBufferEpisode(
        id=ep_id,
        source="chat",
        actor_id="user",
        salience_score=1.0,
        summary=summary,
        token_count=400,
    )
    await store.add_fast_buffer_episode(ep, embedder.embed(summary))
    return ep


@pytest.mark.asyncio
async def test_transactional_add_update_archive_and_soft_archiving(
    store: AgentStore,
    test_settings: Settings,
    embedder: HashEmbeddingProvider,
) -> None:
    # Seed 2 core memory items (1 to update, 1 to archive)
    m_update = await store.insert_core_memory_item(
        MemoryItem(
            id="mem_proj1",
            category="project",
            content="User is prototyping a CLI tool in Python.",
            pinned=False,
            confidence=0.8,
            source_episodes=["ep_old_1"],
        )
    )
    m_archive = await store.insert_core_memory_item(
        MemoryItem(
            id="mem_oldpref",
            category="preference",
            content="User prefers synchronous HTTP libraries.",
            pinned=False,
            confidence=0.7,
            source_episodes=["ep_old_2"],
        )
    )

    await _seed_episode(store, embedder, "ep_1", "User switched CLI project to FastAPI + asyncio.")
    await _seed_episode(store, embedder, "ep_2", "User now prefers httpx async client and SQLite WAL.")

    plan_json = {
        "operations": [
            {
                "op": "ADD",
                "target_id": None,
                "category": "preference",
                "content": "User prefers SQLite in WAL mode for local persistence.",
                "confidence": 0.9,
                "pinned": False,
                "source_episode_ids": ["ep_2"],
                "reason": "Explicitly stated preference.",
            },
            {
                "op": "UPDATE",
                "target_id": m_update.id,
                "category": "project",
                "content": "User is building an async FastAPI and CLI daemon in Python.",
                "confidence": 0.92,
                "pinned": False,
                "source_episode_ids": ["ep_1"],
                "reason": "Architecture updated.",
            },
            {
                "op": "ARCHIVE",
                "target_id": m_archive.id,
                "category": None,
                "content": None,
                "confidence": 0.9,
                "pinned": False,
                "source_episode_ids": ["ep_2"],
                "reason": "Superseded by async httpx preference.",
            },
        ],
        "active_priorities": [
            "Support the async Python FastAPI daemon implementation.",
            "Enforce SQLite WAL persistence and transactional safety.",
            "Watch inbox for architecture notes.",
        ],
        "reflection": "Consolidated architecture update and async preferences.",
    }

    llm_client, sdk = make_mock_client(
        test_settings,
        [
            FakeMessage(
                content=[
                    FakeBlock(type="thinking", thinking="Reviewing episodes for durable diffs..."),
                    FakeBlock(type="text", text=json.dumps(plan_json)),
                ]
            )
        ],
    )
    breaker = AgentCircuitBreaker(store, test_settings, embedder)

    report = await sleep_and_consolidate(
        store=store,
        llm_client=llm_client,
        embedder=embedder,
        circuit_breaker=breaker,
        settings=test_settings,
        trigger_reason="fatigue",
        fatigue_before=0.88,
    )

    assert report.status == "committed"
    assert len(report.added_ids) == 1
    assert report.updated_ids == ["mem_proj1"]
    assert report.archived_ids == ["mem_oldpref"]
    assert len(report.rejected_operations) == 0

    # Verify Haiku 5.5 API constraints for sleep call
    call_kwargs = sdk.messages.calls[0]
    assert call_kwargs["thinking"] == {"type": "adaptive"}
    assert call_kwargs["output_config"]["effort"] == "medium"
    assert call_kwargs["output_config"]["format"]["type"] == "json_schema"
    for forbidden in ("temperature", "top_p", "top_k"):
        assert forbidden not in call_kwargs

    # Verify fast_buffer flushed ONLY after successful commit
    assert await store.list_fast_buffer() == []

    # Verify active Core Memory state
    core_map = await store.get_core_memory_map()
    assert "mem_oldpref" not in core_map
    assert core_map["mem_proj1"].version == 2
    assert "FastAPI" in core_map["mem_proj1"].content
    assert "ep_1" in core_map["mem_proj1"].source_episodes
    assert len(core_map) == 2

    # Verify immutable soft-archive audit trail (both UPDATE_SUPERSEDED and ARCHIVE!)
    archive = await store.list_memory_archive()
    actions = {(a.memory_id, a.action, a.version) for a in archive}
    assert ("mem_proj1", "UPDATE_SUPERSEDED", 1) in actions
    assert ("mem_oldpref", "ARCHIVE", 1) in actions

    # Verify active priorities refreshed & embedded
    prio_vecs = await store.get_active_priority_vectors()
    assert len(prio_vecs) == 3


@pytest.mark.asyncio
async def test_strict_provenance_rejects_invalid_or_missing_episode_ids(
    store: AgentStore,
    test_settings: Settings,
    embedder: HashEmbeddingProvider,
) -> None:
    await _seed_episode(store, embedder, "ep_valid", "User likes dark roast coffee.")

    plan_json = {
        "operations": [
            {
                "op": "ADD",
                "target_id": None,
                "category": "fact",
                "content": "Fabricated fact citing nonexistent episode.",
                "confidence": 0.9,
                "pinned": False,
                "source_episode_ids": ["ep_hallucinated_999"],
                "reason": "Bad provenance",
            },
            {
                "op": "ADD",
                "target_id": None,
                "category": "fact",
                "content": "Fact with empty provenance list.",
                "confidence": 0.9,
                "pinned": False,
                "source_episode_ids": [],
                "reason": "Missing provenance",
            },
            {
                "op": "ADD",
                "target_id": None,
                "category": "preference",
                "content": "User enjoys dark roast coffee.",
                "confidence": 0.85,
                "pinned": False,
                "source_episode_ids": ["ep_valid"],
                "reason": "Grounded in ep_valid",
            },
        ],
        "active_priorities": [
            "Track user preferences accurately.",
            "Assist with daily tasks.",
            "Monitor inbox notifications.",
        ],
        "reflection": "Tested provenance.",
    }

    llm_client, _ = make_mock_client(
        test_settings,
        [FakeMessage(content=[FakeBlock(type="text", text=json.dumps(plan_json))])],
    )
    breaker = AgentCircuitBreaker(store, test_settings, embedder)

    report = await sleep_and_consolidate(
        store=store,
        llm_client=llm_client,
        embedder=embedder,
        circuit_breaker=breaker,
        settings=test_settings,
    )

    assert report.status == "committed"
    assert len(report.added_ids) == 1
    assert len(report.rejected_operations) == 2
    assert all(r.guardrail == "strict_provenance" for r in report.rejected_operations)

    core = await store.list_core_memory()
    assert len(core) == 1
    assert core[0].content == "User enjoys dark roast coffee."


@pytest.mark.asyncio
async def test_pinned_protection_blocks_update_and_archive(
    store: AgentStore,
    test_settings: Settings,
    embedder: HashEmbeddingProvider,
) -> None:
    await store.insert_core_memory_item(
        MemoryItem(
            id="mem_pin_1",
            category="preference",
            content="Never send external emails without explicit confirmation.",
            pinned=True,
            confidence=1.0,
            source_episodes=["ep_init"],
        )
    )
    await store.insert_core_memory_item(
        MemoryItem(
            id="mem_pin_2",
            category="fact",
            content="User timezone is America/New_York.",
            pinned=True,
            confidence=1.0,
            source_episodes=["ep_init"],
        )
    )
    await _seed_episode(store, embedder, "ep_try", "User briefly mentioned Paris travel.")

    plan_json = {
        "operations": [
            {
                "op": "UPDATE",
                "target_id": "mem_pin_2",
                "category": "fact",
                "content": "User timezone is Europe/Paris.",
                "confidence": 0.9,
                "pinned": False,
                "source_episode_ids": ["ep_try"],
                "reason": "Attempting to overwrite pinned item",
            },
            {
                "op": "ARCHIVE",
                "target_id": "mem_pin_1",
                "category": None,
                "content": None,
                "confidence": 0.9,
                "pinned": False,
                "source_episode_ids": ["ep_try"],
                "reason": "Attempting to archive pinned item",
            },
        ],
        "active_priorities": [
            "Respect pinned user constraints.",
            "Assist with upcoming travel planning.",
            "Keep inbox organized.",
        ],
        "reflection": "Attempted pinned mutations.",
    }

    llm_client, _ = make_mock_client(
        test_settings,
        [FakeMessage(content=[FakeBlock(type="text", text=json.dumps(plan_json))])],
    )
    breaker = AgentCircuitBreaker(store, test_settings, embedder)

    report = await sleep_and_consolidate(
        store=store,
        llm_client=llm_client,
        embedder=embedder,
        circuit_breaker=breaker,
        settings=test_settings,
    )

    assert report.status == "committed"
    assert report.updated_ids == []
    assert report.archived_ids == []
    assert len(report.rejected_operations) == 2
    assert all(r.guardrail == "pinned_protection" for r in report.rejected_operations)

    # Both pinned items remain completely unmodified and nothing was archived
    core_map = await store.get_core_memory_map()
    assert core_map["mem_pin_1"].content == "Never send external emails without explicit confirmation."
    assert core_map["mem_pin_2"].content == "User timezone is America/New_York."
    assert core_map["mem_pin_2"].version == 1
    assert await store.list_memory_archive() == []


@pytest.mark.asyncio
async def test_churn_circuit_breaker_aborts_mass_archive_and_preserves_buffer(
    store: AgentStore,
    test_settings: Settings,
    embedder: HashEmbeddingProvider,
) -> None:
    # Seed 12 Core Memory items (> 10 threshold). Archiving 4/12 = 33.3% > 25% -> MUST ABORT!
    for i in range(12):
        await store.insert_core_memory_item(
            MemoryItem(
                id=f"mem_{i:02d}",
                category="fact",
                content=f"Durable fact number {i}.",
                pinned=False,
                confidence=0.85,
                source_episodes=["ep_seed"],
            )
        )
    await _seed_episode(store, embedder, "ep_mass", "User cleaned up some old notes.")

    plan_json = {
        "operations": [
            {
                "op": "ARCHIVE",
                "target_id": f"mem_{i:02d}",
                "category": None,
                "content": None,
                "confidence": 0.8,
                "pinned": False,
                "source_episode_ids": ["ep_mass"],
                "reason": "Aggressive cleanup",
            }
            for i in range(4)  # 4/12 = 33.3% > 25%
        ]
        + [
            {
                "op": "ADD",
                "target_id": None,
                "category": "lesson",
                "content": "This ADD must also not commit when churn breaker trips.",
                "confidence": 0.9,
                "pinned": False,
                "source_episode_ids": ["ep_mass"],
                "reason": "Should be aborted with whole pass",
            }
        ],
        "active_priorities": ["Priority A", "Priority B", "Priority C"],
        "reflection": "Tried to archive 4 items.",
    }

    llm_client, _ = make_mock_client(
        test_settings,
        [FakeMessage(content=[FakeBlock(type="text", text=json.dumps(plan_json))])],
    )
    breaker = AgentCircuitBreaker(store, test_settings, embedder)

    report = await sleep_and_consolidate(
        store=store,
        llm_client=llm_client,
        embedder=embedder,
        circuit_breaker=breaker,
        settings=test_settings,
    )

    assert report.status == "aborted"
    assert report.abort_reason is not None
    assert "Churn circuit breaker tripped" in report.abort_reason

    # All 12 core memory items must remain untouched, NO ADD committed, fast_buffer NOT flushed!
    assert len(await store.list_core_memory()) == 12
    assert await store.list_memory_archive() == []
    assert await store.get_fast_buffer_ids() == {"ep_mass"}

    cycles = await store.list_sleep_cycles()
    assert cycles[0]["status"] == "aborted"
