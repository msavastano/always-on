"""Unit tests for working-buffer fatigue scoring and sleep trigger guards."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pytest

from always_on.cognitive.fatigue import compute_fatigue, compute_semantic_fragmentation
from always_on.config import Settings
from always_on.models import FastBufferEpisode, utc_now


def _make_episode(
    *,
    tokens: int = 500,
    had_error: bool = False,
    had_correction: bool = False,
    summary: str = "episode",
) -> FastBufferEpisode:
    return FastBufferEpisode(
        source="chat",
        actor_id="user",
        salience_score=1.0,
        summary=summary,
        token_count=tokens,
        had_error=had_error,
        had_correction=had_correction,
    )


def test_empty_buffer_never_triggers_sleep(test_settings: Settings) -> None:
    now = utc_now()
    res = compute_fatigue(
        [],
        [],
        last_sleep_at=now - timedelta(hours=12),
        last_activity_at=now - timedelta(hours=2),
        last_user_activity_at=None,
        settings=test_settings,
        now=now,
    )
    assert res.score == 0.0
    assert res.should_sleep is False
    assert res.trigger_reason is None


def test_semantic_fragmentation_measures_context_switching() -> None:
    identical = [
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
    ]
    orthogonal = [
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
    ]
    assert compute_semantic_fragmentation(identical) == pytest.approx(0.0, abs=1e-5)
    assert compute_semantic_fragmentation(orthogonal) == pytest.approx(1.0, abs=1e-5)


def test_blended_fatigue_triggers_sleep_when_idle_from_convo(test_settings: Settings) -> None:
    now = utc_now()
    # High token usage + high fragmentation + errors & corrections + time drift
    episodes = [
        _make_episode(tokens=4500, had_error=True),
        _make_episode(tokens=4500, had_correction=True),
        _make_episode(tokens=4500, had_correction=True),
    ]
    vecs = [
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
    ]
    res = compute_fatigue(
        episodes,
        vecs,
        last_sleep_at=now - timedelta(hours=18),
        last_activity_at=now - timedelta(minutes=10),
        last_user_activity_at=now - timedelta(minutes=10),  # > 5m grace -> not mid-convo
        settings=test_settings,
        now=now,
    )
    assert res.score >= test_settings.sleep_threshold
    assert res.should_sleep is True
    assert res.trigger_reason == "fatigue"
    assert res.breakdown.is_mid_conversation is False


def test_mid_conversation_guard_blocks_sleep_unless_critical_saturation(
    test_settings: Settings,
) -> None:
    now = utc_now()
    # High fatigue (~0.88) during an active conversation (user spoke 20s ago), saturation = 87.5% <= 95%
    episodes = [
        _make_episode(tokens=7000, had_error=True, had_correction=True),
        _make_episode(tokens=7000, had_error=True, had_correction=True),
    ]
    vecs = [
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0], dtype=np.float32),
    ]
    guarded = compute_fatigue(
        episodes,
        vecs,
        last_sleep_at=now - timedelta(hours=20),
        last_activity_at=now - timedelta(seconds=20),
        last_user_activity_at=now - timedelta(seconds=20),
        settings=test_settings,
        now=now,
    )
    assert guarded.score >= test_settings.sleep_threshold
    assert guarded.breakdown.is_mid_conversation is True
    assert guarded.should_sleep is False

    # Now push token saturation > 95% (> 15,200 of 16,000) -> MUST override mid-conversation guard
    critical_episodes = [
        _make_episode(tokens=8000),
        _make_episode(tokens=7600),
    ]
    critical = compute_fatigue(
        critical_episodes,
        vecs,
        last_sleep_at=now - timedelta(minutes=10),
        last_activity_at=now - timedelta(seconds=5),
        last_user_activity_at=now - timedelta(seconds=5),
        settings=test_settings,
        now=now,
    )
    assert critical.breakdown.token_saturation > 0.95
    assert critical.breakdown.is_mid_conversation is True
    assert critical.should_sleep is True
    assert critical.trigger_reason == "critical_saturation"


def test_idle_window_triggers_sleep_with_non_empty_buffer(test_settings: Settings) -> None:
    now = utc_now()
    episodes = [_make_episode(tokens=300)]
    vecs = [np.array([1.0, 0.0], dtype=np.float32)]

    res = compute_fatigue(
        episodes,
        vecs,
        last_sleep_at=now - timedelta(hours=1),
        last_activity_at=now - timedelta(minutes=35),
        last_user_activity_at=now - timedelta(minutes=35),
        settings=test_settings,
        now=now,
    )
    assert res.score < test_settings.sleep_threshold
    assert res.should_sleep is True
    assert res.trigger_reason == "idle"
