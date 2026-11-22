"""Deterministic, disk-persisted circuit breaker and per-wake loop guard."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from always_on.config import Settings
from always_on.db.store import AgentStore, parse_iso
from always_on.embeddings.provider import EmbeddingProvider, cosine_similarity
from always_on.models import CircuitBreakerSnapshot, TokenUsage, utc_now


class CircuitBreakerTripped(RuntimeError):
    """Raised when a deterministic spend, rate, step, loop, or backoff guardrail trips."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"[{reason}] {detail}")
        self.reason = reason
        self.detail = detail


def hash_tool_call(tool_name: str, args: dict[str, Any]) -> str:
    """Compute canonical SHA-256 fingerprint of a tool name + arguments."""
    canonical = json.dumps(
        {"tool": tool_name, "args": args},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class WakeGuard:
    """Per-wake deterministic guard enforcing step limits, duplicate tool blocks, and spin detection."""

    def __init__(
        self,
        *,
        max_steps: int,
        spin_threshold: float,
        embedder: EmbeddingProvider,
    ) -> None:
        self.max_steps = max_steps
        self.spin_threshold = spin_threshold
        self.embedder = embedder
        self.steps_taken: int = 0
        self._seen_tool_hashes: set[str] = set()
        self._thought_vectors: list[np.ndarray] = []

    def check_and_increment_step(self) -> int:
        """Enforce maximum tool-call iterations per wake-up."""
        if self.steps_taken >= self.max_steps:
            raise CircuitBreakerTripped(
                "step_limit",
                f"Reached per-wake step limit ({self.max_steps} iterations).",
            )
        self.steps_taken += 1
        return self.steps_taken

    def check_tool_call(self, tool_name: str, args: dict[str, Any]) -> str:
        """Block exact-duplicate tool calls (SHA-256 of tool + canonical args) within a wake-up."""
        digest = hash_tool_call(tool_name, args)
        if digest in self._seen_tool_hashes:
            raise CircuitBreakerTripped(
                "duplicate_tool_call",
                f"Blocked exact-duplicate tool call '{tool_name}' (sha256={digest[:12]}).",
            )
        self._seen_tool_hashes.add(digest)
        return digest

    def record_thought_and_check_spin(self, thought_text: str) -> None:
        """Abort if the agent's last 3 consecutive thoughts have > spin_threshold cosine similarity."""
        cleaned = thought_text.strip()
        if not cleaned:
            return
        vec = self.embedder.embed(cleaned)
        self._thought_vectors.append(vec)
        if len(self._thought_vectors) < 3:
            return

        v1, v2, v3 = self._thought_vectors[-3:]
        sim_12 = cosine_similarity(v1, v2)
        sim_23 = cosine_similarity(v2, v3)
        sim_13 = cosine_similarity(v1, v3)
        if (
            sim_12 > self.spin_threshold
            and sim_23 > self.spin_threshold
            and sim_13 > self.spin_threshold
        ):
            raise CircuitBreakerTripped(
                "semantic_spinning",
                (
                    f"Detected semantic spinning across last 3 consecutive thoughts "
                    f"(cos_sim={sim_12:.3f}, {sim_23:.3f}, {sim_13:.3f} > {self.spin_threshold})."
                ),
            )


class AgentCircuitBreaker:
    """Disk-backed circuit breaker wrapping every frontier LLM call.

    All counters (daily USD spend, rolling hourly wake timestamps, consecutive error
    counts, and exponential backoff cooldowns) are persisted in SQLite so crash-restarts
    cannot reset spend or safety limits.
    """

    def __init__(
        self,
        store: AgentStore,
        settings: Settings,
        embedder: EmbeddingProvider,
    ) -> None:
        self.store = store
        self.settings = settings
        self.embedder = embedder

    async def check_before_call(
        self,
        *,
        is_wake_start: bool = False,
        allow_during_rate_limit: bool = False,
        now: datetime | None = None,
    ) -> None:
        """Validate daily USD cap, backoff cooldown, and rolling hourly wake limit."""
        current = now or utc_now()
        day = current.strftime("%Y-%m-%d")

        # 1. Hard Daily USD Cap (persisted on disk)
        spend = await self.store.get_daily_spend(day)
        if spend.usd_spent >= self.settings.max_daily_usd:
            raise CircuitBreakerTripped(
                "daily_usd_cap",
                (
                    f"Daily USD spend cap reached: ${spend.usd_spent:.4f} >= "
                    f"${self.settings.max_daily_usd:.2f} for {day}."
                ),
            )

        # 2. Exponential Backoff Cooldown (persisted on disk)
        state = await self.store.get_breaker_state_row()
        cooldown_until = parse_iso(state.get("cooldown_until"))
        if cooldown_until is not None and current < cooldown_until:
            remaining = int((cooldown_until - current).total_seconds())
            raise CircuitBreakerTripped(
                "cooldown_backoff",
                (
                    f"Circuit breaker in exponential backoff cooldown for {remaining}s "
                    f"(consecutive_errors={state['consecutive_errors']})."
                ),
            )

        # 3. Rolling Hourly Rate Limit (persisted in wake_log)
        if is_wake_start and not allow_during_rate_limit:
            window_start = current - timedelta(hours=1)
            wakes_last_hour = await self.store.count_wakes_since(window_start)
            if wakes_last_hour >= self.settings.max_wakes_per_hour:
                raise CircuitBreakerTripped(
                    "hourly_rate_limit",
                    (
                        f"Rolling hourly wake limit reached: {wakes_last_hour} >= "
                        f"{self.settings.max_wakes_per_hour} wakes in last 60m."
                    ),
                )

    async def record_usage(
        self, usage: TokenUsage, *, now: datetime | None = None
    ) -> float:
        """Calculate exact USD cost from token usage + server searches and persist to SQLite."""
        current = now or utc_now()
        day = current.strftime("%Y-%m-%d")
        usd_cost = self.settings.calculate_cost_usd(
            input_tokens=usage.input_tokens,
            cache_read_tokens=usage.cache_read_tokens,
            cache_creation_tokens=usage.cache_creation_tokens,
            output_tokens=usage.output_tokens,
            web_search_requests=usage.web_search_requests,
        )
        await self.store.record_spend(usage, usd_cost, date_utc=day)
        return usd_cost

    async def check_web_search_quota(self, *, now: datetime | None = None) -> None:
        """Enforce persistent daily web search cap and daily USD spend cap."""
        current = now or utc_now()
        day = current.strftime("%Y-%m-%d")
        spend = await self.store.get_daily_spend(day)
        if spend.usd_spent >= self.settings.max_daily_usd:
            raise CircuitBreakerTripped(
                "daily_usd_cap",
                f"Daily USD spend cap reached (${spend.usd_spent:.4f} >= ${self.settings.max_daily_usd:.2f}).",
            )
        if spend.web_searches >= self.settings.max_web_searches_per_day:
            raise CircuitBreakerTripped(
                "daily_search_cap",
                (
                    f"Daily web search limit reached: {spend.web_searches} >= "
                    f"{self.settings.max_web_searches_per_day} searches for {day}."
                ),
            )

    async def record_client_web_search(
        self, count: int = 1, *, now: datetime | None = None
    ) -> None:
        """Persist a client-side web search execution to daily_spend."""
        day = (now or utc_now()).strftime("%Y-%m-%d")
        await self.store.record_web_search(count=count, usd_cost=0.0, date_utc=day)

    async def record_success(self) -> None:
        """Reset consecutive error streak and clear backoff cooldown."""
        await self.store.reset_breaker_errors()

    async def record_error(self, *, now: datetime | None = None) -> int:
        """Increment consecutive error streak and persist exponential backoff cooldown.

        Progression (with base=60s, max=3600s): 60s -> 120s -> 240s -> 480s -> ... -> 3600s.
        """
        current = now or utc_now()
        state = await self.store.get_breaker_state_row()
        consecutive = int(state["consecutive_errors"]) + 1
        delay_seconds = min(
            self.settings.backoff_max_seconds,
            int(self.settings.backoff_base_seconds * (2 ** (consecutive - 1))),
        )
        cooldown_until = (current + timedelta(seconds=delay_seconds)).isoformat()
        await self.store.update_breaker_error(consecutive, cooldown_until)
        return delay_seconds

    def new_wake_guard(self) -> WakeGuard:
        """Create a fresh per-wake guard for step cap, duplicate tool, and spin detection."""
        return WakeGuard(
            max_steps=self.settings.max_steps_per_wake,
            spin_threshold=self.settings.semantic_spin_threshold,
            embedder=self.embedder,
        )

    async def snapshot(self, *, now: datetime | None = None) -> CircuitBreakerSnapshot:
        """Return full persisted circuit breaker state for telemetry/UI."""
        current = now or utc_now()
        day = current.strftime("%Y-%m-%d")
        spend = await self.store.get_daily_spend(day)
        state = await self.store.get_breaker_state_row()
        wakes_last_hour = await self.store.count_wakes_since(current - timedelta(hours=1))
        cooldown_until_dt = parse_iso(state.get("cooldown_until"))
        in_cooldown = cooldown_until_dt is not None and current < cooldown_until_dt

        return CircuitBreakerSnapshot(
            daily_spend=spend,
            max_daily_usd=self.settings.max_daily_usd,
            max_web_searches_per_day=self.settings.max_web_searches_per_day,
            wakes_last_hour=wakes_last_hour,
            max_wakes_per_hour=self.settings.max_wakes_per_hour,
            consecutive_errors=int(state["consecutive_errors"]),
            cooldown_until=state.get("cooldown_until"),
            in_cooldown=in_cooldown,
            daily_cap_exceeded=spend.usd_spent >= self.settings.max_daily_usd,
            daily_search_cap_exceeded=spend.web_searches >= self.settings.max_web_searches_per_day,
            hourly_rate_exceeded=wakes_last_hour >= self.settings.max_wakes_per_hour,
            last_sleep_at=str(state["last_sleep_at"]),
            last_wake_at=state.get("last_wake_at"),
            last_user_activity_at=state.get("last_user_activity_at"),
        )
