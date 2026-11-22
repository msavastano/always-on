"""Zero-token local salience gating using embeddings and vector math."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from always_on.config import Settings
from always_on.embeddings.provider import cosine_similarity
from always_on.models import IncomingEvent, SalienceBreakdown, SalienceResult


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def compute_temporal_pressure(
    idle_seconds: float,
    *,
    heartbeat_interval_seconds: float,
    asymptote: float = 0.65,
    steepness: float = 2.6,
) -> float:
    """Smooth exponential curve rising from 0.0 toward asymptote (e.g. 0.65) over idle time.

    With asymptote=0.65 and steepness=2.6, pressure reaches ~0.602 right at
    1.0 * heartbeat_interval_seconds, naturally crossing WAKE_THRESHOLD=0.60
    during quiet stretches while remaining bounded below 0.65.
    """
    elapsed = max(0.0, float(idle_seconds))
    interval = max(1.0, float(heartbeat_interval_seconds))
    pressure = asymptote * (1.0 - math.exp(-steepness * (elapsed / interval)))
    return _clamp01(pressure)


def compute_salience(
    event: IncomingEvent,
    *,
    event_embedding: np.ndarray | None,
    recent_buffer_embeddings: Sequence[np.ndarray],
    priority_embeddings: Sequence[np.ndarray],
    idle_seconds: float,
    settings: Settings,
) -> SalienceResult:
    """Score an incoming event from 0.0 to 1.0 using zero LLM tokens.

    Combines four signals:
      1. Hard Override: Direct user chat or high-priority alert -> 1.0 immediately.
      2. Novelty (Semantic Surprise): 1.0 - max cosine similarity vs. recent fast_buffer.
      3. Goal Relevance: max cosine similarity vs. active priority embeddings.
      4. Temporal Pressure: smooth exponential curve rising toward ~0.65 over idle time.
    """
    elapsed_idle = max(0.0, float(idle_seconds))
    temporal = compute_temporal_pressure(
        elapsed_idle,
        heartbeat_interval_seconds=settings.heartbeat_interval_seconds,
        asymptote=settings.temporal_pressure_asymptote,
        steepness=settings.temporal_pressure_steepness,
    )

    # 1. Hard Override: direct user message or explicit high-priority alert
    if event.hard_override or event.source == "chat" or event.priority == "high":
        return SalienceResult(
            score=1.0,
            should_wake=True,
            threshold=settings.wake_threshold,
            breakdown=SalienceBreakdown(
                hard_override=True,
                novelty=1.0,
                goal_relevance=1.0,
                temporal_pressure=round(temporal, 4),
                idle_seconds=round(elapsed_idle, 2),
            ),
        )

    # Internal heartbeat ticks carry no novel external content; rely strictly on temporal pressure
    if event.source == "heartbeat":
        score = round(temporal, 4)
        return SalienceResult(
            score=score,
            should_wake=score > settings.wake_threshold,
            threshold=settings.wake_threshold,
            breakdown=SalienceBreakdown(
                hard_override=False,
                novelty=0.0,
                goal_relevance=0.0,
                temporal_pressure=score,
                idle_seconds=round(elapsed_idle, 2),
            ),
        )

    if event_embedding is None:
        raise ValueError("event_embedding is required for non-override, non-heartbeat events")

    # 2. Novelty (Semantic Surprise): 1.0 - max_cosine_similarity against recent buffer
    if not recent_buffer_embeddings:
        novelty = 1.0
    else:
        max_buf_sim = max(
            max(0.0, cosine_similarity(event_embedding, buf_vec))
            for buf_vec in recent_buffer_embeddings
        )
        novelty = _clamp01(1.0 - max_buf_sim)

    # 3. Goal Relevance: max_cosine_similarity against active priority embeddings
    if not priority_embeddings:
        goal_relevance = 0.0
    else:
        goal_relevance = _clamp01(
            max(
                max(0.0, cosine_similarity(event_embedding, prio_vec))
                for prio_vec in priority_embeddings
            )
        )

    # 4. Blend signals cleanly in [0.0, 1.0]; temporal pressure provides a floor during long idles
    w_nov = settings.salience_weight_novelty
    w_rel = settings.salience_weight_relevance
    w_tmp = settings.salience_weight_temporal
    weight_sum = w_nov + w_rel + w_tmp
    if weight_sum <= 0.0:
        blended = temporal
    else:
        blended = (
            w_nov * novelty + w_rel * goal_relevance + w_tmp * temporal
        ) / weight_sum

    score = round(_clamp01(max(temporal, blended)), 4)

    return SalienceResult(
        score=score,
        should_wake=score > settings.wake_threshold,
        threshold=settings.wake_threshold,
        breakdown=SalienceBreakdown(
            hard_override=False,
            novelty=round(novelty, 4),
            goal_relevance=round(goal_relevance, 4),
            temporal_pressure=round(temporal, 4),
            idle_seconds=round(elapsed_idle, 2),
        ),
    )
