"""Unit tests for persistent AgentCircuitBreaker and per-wake WakeGuard."""

from __future__ import annotations

from datetime import timedelta

import pytest

from always_on.config import Settings
from always_on.db.store import AgentStore
from always_on.embeddings.provider import HashEmbeddingProvider
from always_on.models import TokenUsage, utc_now
from always_on.safety.circuit_breaker import AgentCircuitBreaker, CircuitBreakerTripped


@pytest.mark.asyncio
async def test_daily_usd_cap_persists_across_restarts(
    test_settings: Settings,
    embedder: HashEmbeddingProvider,
) -> None:
    test_settings.max_daily_usd = 0.25

    # First process lifetime records spend from real token usage
    store1 = AgentStore(test_settings.db_path)
    await store1.connect()
    cb1 = AgentCircuitBreaker(store1, test_settings, embedder)

    # Standard <=100k prompt pricing:
    # 30k input ($0.003) + 30k cache_read ($0.0003) + 30k cache_write ($0.00375) + 40k output ($0.02) = $0.02705 per call
    usage = TokenUsage(
        input_tokens=30_000,
        cache_read_tokens=30_000,
        cache_creation_tokens=30_000,
        output_tokens=40_000,
    )
    single_cost = await cb1.record_usage(usage)
    assert single_cost == pytest.approx(0.02705, rel=1e-4)

    # Record 9 more calls -> total $0.2705 > $0.25 daily cap
    for _ in range(9):
        await cb1.record_usage(usage)
    await store1.close()

    # Verify >100k prompt tiered pricing multiplier (5x)
    long_cost = test_settings.calculate_cost_usd(
        input_tokens=100_000,
        cache_read_tokens=50_000,
        cache_creation_tokens=0,
        output_tokens=10_000,
    )
    assert long_cost == pytest.approx(5.0 * (0.01 + 0.0005 + 0.005), rel=1e-4)

    # Simulate crash-restart with a brand-new AgentStore + AgentCircuitBreaker on the same SQLite DB
    store2 = AgentStore(test_settings.db_path)
    await store2.connect()
    cb2 = AgentCircuitBreaker(store2, test_settings, embedder)
    try:
        with pytest.raises(CircuitBreakerTripped) as exc_info:
            await cb2.check_before_call(is_wake_start=True)
        assert exc_info.value.reason == "daily_usd_cap"
    finally:
        await store2.close()


@pytest.mark.asyncio
async def test_rolling_hourly_rate_limit(
    store: AgentStore,
    test_settings: Settings,
    embedder: HashEmbeddingProvider,
) -> None:
    test_settings.max_wakes_per_hour = 3
    cb = AgentCircuitBreaker(store, test_settings, embedder)
    now = utc_now()

    # Log 1 old wake (65 min ago -> outside rolling window) and 3 recent wakes
    await store.log_wake(
        source="chat", salience=1.0, effort="low", usd_cost=0.0, steps=0, status="ok",
        timestamp=now - timedelta(minutes=65),
    )
    for mins_ago in (40, 20, 5):
        await store.log_wake(
            source="chat", salience=1.0, effort="low", usd_cost=0.0, steps=0, status="ok",
            timestamp=now - timedelta(minutes=mins_ago),
        )

    # At `now`: 3 wakes in last 60m -> tripped
    with pytest.raises(CircuitBreakerTripped) as exc_info:
        await cb.check_before_call(is_wake_start=True, now=now)
    assert exc_info.value.reason == "hourly_rate_limit"

    # 25 minutes later: the 40-min-old wake fell outside the rolling hour -> allowed again!
    await cb.check_before_call(is_wake_start=True, now=now + timedelta(minutes=25))


def test_per_wake_step_limit_and_exact_duplicate_tool_blocking(
    test_settings: Settings,
    embedder: HashEmbeddingProvider,
) -> None:
    cb = AgentCircuitBreaker(AgentStore(":memory:"), test_settings, embedder)
    guard = cb.new_wake_guard()

    # Step cap = 6
    for i in range(1, 7):
        assert guard.check_and_increment_step() == i
    with pytest.raises(CircuitBreakerTripped) as exc_info:
        guard.check_and_increment_step()
    assert exc_info.value.reason == "step_limit"

    # Exact duplicate tool call detection (key-order invariant SHA-256)
    guard2 = cb.new_wake_guard()
    guard2.check_tool_call("read_note", {"filename": "todo.md", "mode": "r"})
    with pytest.raises(CircuitBreakerTripped) as dup_exc:
        guard2.check_tool_call("read_note", {"mode": "r", "filename": "todo.md"})
    assert dup_exc.value.reason == "duplicate_tool_call"


def test_semantic_spinning_detection(
    test_settings: Settings,
    embedder: HashEmbeddingProvider,
) -> None:
    cb = AgentCircuitBreaker(AgentStore(":memory:"), test_settings, embedder)

    # Distinct reasoning steps should NOT trip
    distinct_guard = cb.new_wake_guard()
    distinct_guard.record_thought_and_check_spin("First I will inspect the inbox directory files.")
    distinct_guard.record_thought_and_check_spin("Next I should verify the current UTC timestamp.")
    distinct_guard.record_thought_and_check_spin("Finally I will summarize findings into a note.")

    # Three near-identical spinning thoughts MUST trip > 0.92 cosine similarity
    spin_guard = cb.new_wake_guard()
    spin_guard.record_thought_and_check_spin(
        "I need to retry reading the missing configuration file again to see if it appears."
    )
    spin_guard.record_thought_and_check_spin(
        "I need to retry reading the missing configuration file again to see if it appears now."
    )
    with pytest.raises(CircuitBreakerTripped) as exc_info:
        spin_guard.record_thought_and_check_spin(
            "I need to retry reading the missing configuration file again to see if it appears."
        )
    assert exc_info.value.reason == "semantic_spinning"


@pytest.mark.asyncio
async def test_exponential_backoff_persists_and_caps_at_one_hour(
    test_settings: Settings,
    embedder: HashEmbeddingProvider,
) -> None:
    store = AgentStore(test_settings.db_path)
    await store.connect()
    cb = AgentCircuitBreaker(store, test_settings, embedder)
    t0 = utc_now()

    expected = [60, 120, 240, 480, 960, 1920, 3600, 3600]
    actual = [await cb.record_error(now=t0) for _ in expected]
    assert actual == expected

    # Cooldown must trip before t0 + 3600s, even after DB restart
    await store.close()
    store_restarted = AgentStore(test_settings.db_path)
    await store_restarted.connect()
    cb_restarted = AgentCircuitBreaker(store_restarted, test_settings, embedder)
    try:
        with pytest.raises(CircuitBreakerTripped) as exc_info:
            await cb_restarted.check_before_call(now=t0 + timedelta(minutes=30))
        assert exc_info.value.reason == "cooldown_backoff"

        # After cooldown elapses -> allowed, and record_success clears error streak
        await cb_restarted.check_before_call(now=t0 + timedelta(seconds=3601))
        await cb_restarted.record_success()
        snap = await cb_restarted.snapshot(now=t0 + timedelta(seconds=3602))
        assert snap.consecutive_errors == 0
        assert snap.in_cooldown is False
    finally:
        await store_restarted.close()
