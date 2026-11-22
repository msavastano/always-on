"""Fast working buffer fatigue scoring and mid-conversation sleep trigger guard."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

import numpy as np

from always_on.config import Settings
from always_on.models import FastBufferEpisode, FatigueBreakdown, FatigueResult, utc_now


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def compute_semantic_fragmentation(embeddings: Sequence[np.ndarray]) -> float:
    """Average pairwise cosine distance across buffer embeddings in [0.0, 1.0].

    Measures topic/context-switching clutter in working memory. Returns 0.0 when
    fewer than 2 entries exist.
    """
    n = len(embeddings)
    if n < 2:
        return 0.0

    mat = np.vstack([np.asarray(v, dtype=np.float32) for v in embeddings])
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    safe_norms = np.where(norms > 0.0, norms, 1.0)
    normed = mat / safe_norms

    sim_matrix = np.clip(normed @ normed.T, 0.0, 1.0)
    iu = np.triu_indices(n, k=1)
    distances = 1.0 - sim_matrix[iu]
    return _clamp01(float(np.mean(distances)))


def compute_fatigue(
    episodes: Sequence[FastBufferEpisode],
    embeddings: Sequence[np.ndarray],
    *,
    last_sleep_at: datetime,
    last_activity_at: datetime | None,
    last_user_activity_at: datetime | None,
    settings: Settings,
    now: datetime | None = None,
) -> FatigueResult:
    """Compute working-memory fatigue in [0.0, 1.0] and determine whether to trigger sleep.

    Blends four signals:
      1. Token Saturation (buffer tokens vs. max working budget, e.g. 16k)
      2. Semantic Fragmentation (avg pairwise cosine distance across buffer)
      3. Lesson / Error Load (weighted tool errors and user corrections)
      4. Gentle Time Drift (hours since last sleep)

    Guards against sleeping mid-conversation unless token saturation is critical (> 0.95).
    """
    current = now or utc_now()
    hours_since_sleep = max(0.0, (current - last_sleep_at).total_seconds() / 3600.0)
    ref_activity = last_activity_at or last_sleep_at
    idle_seconds = max(0.0, (current - ref_activity).total_seconds())

    is_mid_conversation = (
        last_user_activity_at is not None
        and max(0.0, (current - last_user_activity_at).total_seconds())
        < settings.active_conversation_grace_seconds
    )

    if not episodes:
        return FatigueResult(
            score=0.0,
            should_sleep=False,
            trigger_reason=None,
            breakdown=FatigueBreakdown(
                token_saturation=0.0,
                semantic_fragmentation=0.0,
                lesson_error_load=0.0,
                time_drift=round(
                    _clamp01(hours_since_sleep / settings.fatigue_drift_horizon_hours), 4
                ),
                buffer_entries=0,
                buffer_tokens=0,
                error_count=0,
                correction_count=0,
                hours_since_sleep=round(hours_since_sleep, 3),
                idle_seconds=round(idle_seconds, 2),
                is_mid_conversation=is_mid_conversation,
            ),
        )

    # 1. Token Saturation
    total_tokens = sum(max(0, ep.token_count) for ep in episodes)
    token_saturation = _clamp01(total_tokens / float(settings.max_working_budget_tokens))

    # 2. Semantic Fragmentation
    semantic_fragmentation = compute_semantic_fragmentation(embeddings)

    # 3. Lesson / Error Load (weighted count of tool errors and user corrections)
    error_count = sum(
        1 for ep in episodes if ep.had_error or any(tc.is_error for tc in ep.tool_calls)
    )
    correction_count = sum(1 for ep in episodes if ep.had_correction)
    weighted_lessons = (1.0 * error_count) + (1.5 * correction_count)
    lesson_error_load = _clamp01(weighted_lessons / 5.0)

    # 4. Gentle Time Drift
    time_drift = _clamp01(hours_since_sleep / settings.fatigue_drift_horizon_hours)

    # Blend signals; allow extreme token saturation to surface directly in fatigue score
    blended = (
        0.40 * token_saturation
        + 0.20 * semantic_fragmentation
        + 0.25 * lesson_error_load
        + 0.15 * time_drift
    )
    if token_saturation >= settings.sleep_threshold:
        score = max(blended, token_saturation)
    else:
        score = blended
    score = round(_clamp01(score), 4)

    # Sleep Trigger Decision
    should_sleep = False
    trigger_reason = None

    if token_saturation > settings.critical_saturation_threshold:
        # Critical token saturation (> 0.95) overrides mid-conversation guard
        should_sleep = True
        trigger_reason = "critical_saturation"
    elif is_mid_conversation:
        # Never interrupt an active user conversation unless saturation is critical
        should_sleep = False
        trigger_reason = None
    elif score >= settings.sleep_threshold:
        should_sleep = True
        trigger_reason = "fatigue"
    elif idle_seconds >= settings.idle_sleep_seconds:
        should_sleep = True
        trigger_reason = "idle"

    return FatigueResult(
        score=score,
        should_sleep=should_sleep,
        trigger_reason=trigger_reason,
        breakdown=FatigueBreakdown(
            token_saturation=round(token_saturation, 4),
            semantic_fragmentation=round(semantic_fragmentation, 4),
            lesson_error_load=round(lesson_error_load, 4),
            time_drift=round(time_drift, 4),
            buffer_entries=len(episodes),
            buffer_tokens=total_tokens,
            error_count=error_count,
            correction_count=correction_count,
            hours_since_sleep=round(hours_since_sleep, 3),
            idle_seconds=round(idle_seconds, 2),
            is_mid_conversation=is_mid_conversation,
        ),
    )
