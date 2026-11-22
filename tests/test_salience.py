"""Unit tests for zero-token local salience gating (compute_salience)."""

from __future__ import annotations

import numpy as np
import pytest

from always_on.cognitive.salience import compute_salience, compute_temporal_pressure
from always_on.config import Settings
from always_on.embeddings.provider import HashEmbeddingProvider
from always_on.models import IncomingEvent


def test_hard_override_direct_chat(test_settings: Settings) -> None:
    evt = IncomingEvent(source="chat", actor_id="user", content="Hey, check my schedule")
    res = compute_salience(
        evt,
        event_embedding=None,
        recent_buffer_embeddings=[],
        priority_embeddings=[],
        idle_seconds=0.0,
        settings=test_settings,
    )
    assert res.score == 1.0
    assert res.should_wake is True
    assert res.breakdown.hard_override is True


def test_hard_override_high_priority_alert(test_settings: Settings) -> None:
    evt = IncomingEvent(
        source="webhook",
        actor_id="pagerduty",
        content="Production database disk full",
        priority="high",
    )
    res = compute_salience(
        evt,
        event_embedding=None,
        recent_buffer_embeddings=[],
        priority_embeddings=[],
        idle_seconds=0.0,
        settings=test_settings,
    )
    assert res.score == 1.0
    assert res.should_wake is True
    assert res.breakdown.hard_override is True


def test_novelty_suppresses_repetitive_events(
    test_settings: Settings, embedder: HashEmbeddingProvider
) -> None:
    text = "Automated daily backup completed cleanly for analytics cluster"
    vec = embedder.embed(text)
    evt = IncomingEvent(source="webhook", actor_id="ci", content=text)

    # Identical content already in recent buffer -> novelty ~ 0.0, irrelevant priority -> gated out
    unrelated_prio = embedder.embed("Prepare quarterly tax filing documents for accountant")
    res = compute_salience(
        evt,
        event_embedding=vec,
        recent_buffer_embeddings=[vec],
        priority_embeddings=[unrelated_prio],
        idle_seconds=10.0,
        settings=test_settings,
    )
    assert res.breakdown.novelty < 0.05
    assert res.score <= test_settings.wake_threshold
    assert res.should_wake is False


def test_novel_and_goal_relevant_event_wakes(
    test_settings: Settings, embedder: HashEmbeddingProvider
) -> None:
    # Orthogonal vectors for crystal-clear deterministic geometry
    prio_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    buf_vec = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    evt_vec = np.array([0.95, 0.05, 0.30, 0.0], dtype=np.float32)
    evt_vec /= np.linalg.norm(evt_vec)

    evt = IncomingEvent(
        source="file_watcher",
        actor_id="filesystem",
        content="New design spec uploaded for priority project",
    )
    res = compute_salience(
        evt,
        event_embedding=evt_vec,
        recent_buffer_embeddings=[buf_vec],
        priority_embeddings=[prio_vec],
        idle_seconds=0.0,
        settings=test_settings,
    )
    assert res.breakdown.novelty > 0.9
    assert res.breakdown.goal_relevance > 0.9
    assert res.score > test_settings.wake_threshold
    assert res.should_wake is True


def test_temporal_pressure_curve_and_heartbeat_wakeup(test_settings: Settings) -> None:
    interval = test_settings.heartbeat_interval_seconds
    p0 = compute_temporal_pressure(0, heartbeat_interval_seconds=interval)
    p_half = compute_temporal_pressure(interval * 0.5, heartbeat_interval_seconds=interval)
    p_full = compute_temporal_pressure(interval * 1.0, heartbeat_interval_seconds=interval)
    p_long = compute_temporal_pressure(interval * 10.0, heartbeat_interval_seconds=interval)

    assert p0 == pytest.approx(0.0)
    assert p0 < p_half < p_full < p_long <= 0.65
    assert p_half < test_settings.wake_threshold
    assert p_full > test_settings.wake_threshold

    hb = IncomingEvent(source="heartbeat", actor_id="clock", content="tick")
    quiet_res = compute_salience(
        hb,
        event_embedding=None,
        recent_buffer_embeddings=[],
        priority_embeddings=[],
        idle_seconds=interval * 0.25,
        settings=test_settings,
    )
    assert quiet_res.should_wake is False

    overdue_res = compute_salience(
        hb,
        event_embedding=None,
        recent_buffer_embeddings=[],
        priority_embeddings=[],
        idle_seconds=interval * 1.1,
        settings=test_settings,
    )
    assert overdue_res.should_wake is True
    assert overdue_res.score <= 0.65
