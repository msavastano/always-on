"""Continuous always-on outer event loop daemon orchestrating CLS memory & safety."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import re
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from always_on.cognitive.consolidation import sleep_and_consolidate
from always_on.cognitive.fatigue import compute_fatigue
from always_on.cognitive.salience import compute_salience
from always_on.config import Settings
from always_on.db.store import DEFAULT_PRIORITIES, AgentStore, parse_iso
from always_on.embeddings.provider import EmbeddingProvider, create_embedding_provider
from always_on.llm.client import HaikuClient, MissingAPIKeyError
from always_on.llm.prompts import (
    build_cached_system_blocks,
    build_wake_user_message,
    estimate_token_count,
)
from always_on.models import (
    ChatReply,
    FastBufferEpisode,
    FatigueResult,
    IncomingEvent,
    SalienceResult,
    SleepCycleReport,
    TokenUsage,
    ToolCallRecord,
    utc_now,
    utc_now_iso,
)
from always_on.safety.circuit_breaker import AgentCircuitBreaker, CircuitBreakerTripped
from always_on.sources.base import EventIngestor, EventSource
from always_on.sources.chat import ChatSource
from always_on.sources.file_watcher import FileWatcherSource
from always_on.sources.heartbeat import HeartbeatSource
from always_on.sources.webhook import WebhookSource
from always_on.tools.registry import ToolRegistry
from always_on.tools.web_search import WebSearchProvider
from always_on.watchers.claude_code import check_and_notify_once
from always_on.watchers.vercel import poll_vercel_and_notify_once

logger = logging.getLogger("always_on.daemon")

_CORRECTION_CUE_RE = re.compile(
    r"\b("
    r"actually,\s|correction:|that(?:'s| is)\s+(?:wrong|incorrect)|"
    r"you(?:'re| are)\s+wrong|no,\s+i\s+(?:meant|prefer|want)|"
    r"stop\s+doing\s+that|don(?:'t| not)\s+do\s+that"
    r")\b",
    re.IGNORECASE,
)


def detect_user_correction(event: IncomingEvent) -> bool:
    """Detect whether an incoming user message flags a correction or preference fix."""
    if event.is_correction:
        return True
    if event.source != "chat":
        return False
    return bool(_CORRECTION_CUE_RE.search(event.content))


@dataclass
class ActivityRecord:
    """Lightweight in-memory feed item shown in the live dashboard."""

    timestamp: str
    kind: str
    source: str
    summary: str
    salience: float | None = None
    status: str = "ok"

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "kind": self.kind,
            "source": self.source,
            "summary": self.summary,
            "salience": self.salience,
            "status": self.status,
        }


class AlwaysOnDaemon:
    """Continuous CLS AI Agent daemon with zero-token salience gating and sleep consolidation."""

    def __init__(
        self,
        settings: Settings,
        *,
        store: AgentStore | None = None,
        embedder: EmbeddingProvider | None = None,
        llm_client: HaikuClient | None = None,
        web_search_provider: WebSearchProvider | None = None,
        extra_sources: list[EventSource] | None = None,
    ) -> None:
        self.settings = settings
        self.store = store or AgentStore(settings.db_path)
        self.embedder = embedder or create_embedding_provider(
            settings.embedding_provider, settings.embedding_model
        )
        self.llm_client = llm_client or HaikuClient(settings)
        self.circuit_breaker = AgentCircuitBreaker(self.store, settings, self.embedder)
        self.ingestor = EventIngestor(self.store, settings)
        self.tools = ToolRegistry(
            store=self.store,
            inbox_dir=settings.inbox_dir,
            notes_dir=settings.notes_dir,
            settings=settings,
            circuit_breaker=self.circuit_breaker,
            web_search_provider=web_search_provider,
        )

        # Built-in pluggable event sources
        self.chat_source = ChatSource()
        self.file_watcher_source = FileWatcherSource(settings.inbox_dir)
        self.webhook_source = WebhookSource()
        self.heartbeat_source = HeartbeatSource(settings.heartbeat_emit_seconds)
        self.sources: list[EventSource] = [
            self.chat_source,
            self.file_watcher_source,
            self.webhook_source,
            self.heartbeat_source,
            *(extra_sources or []),
        ]

        # Runtime state & live telemetry
        self._agent_lock = asyncio.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_task: asyncio.Task[None] | None = None
        self._watcher_tasks: list[asyncio.Task[None]] = []
        self._watcher_status: dict[str, dict[str, Any]] = {}
        self._stop_event = asyncio.Event()
        self._shutting_down = False
        self._started = False
        self.last_salience: SalienceResult | None = None
        self.last_fatigue: FatigueResult | None = None
        self.last_sleep_report: SleepCycleReport | None = None
        self.activity_feed: deque[ActivityRecord] = deque(maxlen=60)
        self.chat_history: deque[dict[str, Any]] = deque(maxlen=100)
        self._sse_subscribers: set[asyncio.Queue[dict[str, Any]]] = set()

    @property
    def is_stopping(self) -> bool:
        return self._shutting_down or self._stop_event.is_set()

    def request_shutdown(self) -> None:
        """Signal-safe hook to immediately wake and terminate loops & SSE streams."""
        self._shutting_down = True
        loop = self._loop
        if loop is not None and not loop.is_closed():
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(self._stop_event.set)
        else:
            self._stop_event.set()

    async def start(self, *, run_background_loop: bool = True) -> None:
        """Connect SQLite, seed default priorities, start sources, and spawn outer event loop."""
        if self._started:
            return
        self._loop = asyncio.get_running_loop()
        self._shutting_down = False
        await self.store.connect()

        # Seed default priorities on first boot so Goal Relevance works immediately
        existing_priorities = await self.store.list_active_priorities()
        if not existing_priorities:
            vecs = self.embedder.embed_batch(DEFAULT_PRIORITIES)
            await self.store.replace_active_priorities(list(zip(DEFAULT_PRIORITIES, vecs)))

        for src in self.sources:
            await src.start()

        self.last_fatigue = await self.compute_current_fatigue()
        self._stop_event.clear()
        self._started = True
        self._record_activity(
            kind="system",
            source="daemon",
            summary=f"{self.settings.agent_name} daemon started (model={self.settings.anthropic_model}).",
        )

        if run_background_loop:
            self._loop_task = asyncio.create_task(self._outer_loop(), name="always-on-loop")
            if self.settings.enable_claude_code_watcher:
                self._watcher_tasks.append(
                    asyncio.create_task(
                        self._run_claude_code_watcher_loop(),
                        name="watcher-claude-code",
                    )
                )
            if self.settings.enable_vercel_watcher:
                self._watcher_tasks.append(
                    asyncio.create_task(
                        self._run_vercel_watcher_loop(),
                        name="watcher-vercel",
                    )
                )

    async def stop(self) -> None:
        """Gracefully stop the outer event loop, watchers, sources, LLM client, and SQLite store."""
        if not self._started:
            return
        self._shutting_down = True
        self._stop_event.set()
        for task in self._watcher_tasks:
            task.cancel()
        if self._watcher_tasks:
            with contextlib.suppress(asyncio.CancelledError, asyncio.TimeoutError):
                await asyncio.wait_for(
                    asyncio.gather(*self._watcher_tasks, return_exceptions=True),
                    timeout=2.0,
                )
            self._watcher_tasks.clear()
        if self._loop_task is not None:
            self._loop_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, asyncio.TimeoutError):
                await asyncio.wait_for(self._loop_task, timeout=2.0)
            self._loop_task = None

        for src in self.sources:
            with contextlib.suppress(Exception):
                await src.stop()

        with contextlib.suppress(Exception):
            await asyncio.wait_for(self.llm_client.close(), timeout=2.0)
        with contextlib.suppress(Exception):
            await asyncio.wait_for(self.store.close(), timeout=2.0)
        self._started = False

    async def _deliver_watcher_webhook(self, body: dict[str, Any]) -> dict[str, Any]:
        """Deliver a managed watcher payload straight into WebhookSource (zero HTTP loopback)."""
        event = await self.push_webhook(
            content=str(body.get("content", "")),
            actor_id=str(body.get("actor_id", "webhook")),
            payload=dict(body.get("payload") or {}),
            priority=str(body.get("priority", "normal")),
            hard_override=bool(body.get("hard_override", False)),
        )
        return {"accepted": True, "event_id": event.id}

    async def _run_claude_code_watcher_loop(self) -> None:
        """Background watcher polling anthropics/claude-code releases."""
        interval = self.settings.claude_code_watcher_interval_seconds
        token = self.settings.github_token or os.environ.get("GITHUB_TOKEN") or None
        while not self._stop_event.is_set():
            try:
                res = await check_and_notify_once(
                    state_path=self.settings.claude_code_watcher_state_path,
                    priority=self.settings.claude_code_watcher_priority,
                    github_token=token,
                    deliver_fn=self._deliver_watcher_webhook,
                )
                self._watcher_status["claude_code"] = {
                    "enabled": True,
                    "interval_seconds": interval,
                    "last_status": res.status,
                    "last_tag": res.tag,
                    "last_message": res.message,
                    "last_checked_at": utc_now_iso(),
                }
                if res.status in ("notified", "error"):
                    self._record_activity(
                        kind="watcher",
                        source="claude_code",
                        summary=res.message,
                        status=res.status,
                    )
                    await self._broadcast_telemetry()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("Managed Claude Code watcher failed: %s", exc)
                self._watcher_status["claude_code"] = {
                    "enabled": True,
                    "interval_seconds": interval,
                    "last_status": "error",
                    "last_message": str(exc),
                    "last_checked_at": utc_now_iso(),
                }
                self._record_activity(
                    kind="watcher",
                    source="claude_code",
                    summary=f"Watcher error: {exc}",
                    status="error",
                )
                await self._broadcast_telemetry()

            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=interval)
            except asyncio.TimeoutError:
                continue

    async def _run_vercel_watcher_loop(self) -> None:
        """Background zero-tunnel watcher polling Vercel deployments."""
        interval = self.settings.vercel_watcher_interval_seconds
        while not self._stop_event.is_set():
            try:
                res = await poll_vercel_and_notify_once(
                    api_token=self.settings.vercel_api_token,
                    state_path=self.settings.vercel_watcher_state_path,
                    team_id=self.settings.vercel_team_id,
                    project_filter=self.settings.vercel_project_filter,
                    target_filter=self.settings.vercel_watcher_target,
                    policy=self.settings.vercel_priority_policy,
                    deliver_fn=self._deliver_watcher_webhook,
                )
                self._watcher_status["vercel"] = {
                    "enabled": True,
                    "interval_seconds": interval,
                    "policy": self.settings.vercel_priority_policy,
                    "last_status": res.status,
                    "notified_count": res.notified_count,
                    "last_message": res.message,
                    "last_checked_at": utc_now_iso(),
                }
                if res.status in ("notified", "seeded", "error"):
                    self._record_activity(
                        kind="watcher",
                        source="vercel",
                        summary=res.message,
                        status=res.status,
                    )
                    await self._broadcast_telemetry()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("Managed Vercel watcher failed: %s", exc)
                self._watcher_status["vercel"] = {
                    "enabled": True,
                    "interval_seconds": interval,
                    "policy": self.settings.vercel_priority_policy,
                    "last_status": "error",
                    "last_message": str(exc),
                    "last_checked_at": utc_now_iso(),
                }
                self._record_activity(
                    kind="watcher",
                    source="vercel",
                    summary=f"Watcher error: {exc}",
                    status="error",
                )
                await self._broadcast_telemetry()

            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=interval)
            except asyncio.TimeoutError:
                continue

    async def _outer_loop(self) -> None:
        """Lightweight, non-blocking zero-token outer event loop."""
        while not self._stop_event.is_set():
            try:
                await self.tick_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("Unhandled error in outer event loop tick: %s", exc)
                self._record_activity(
                    kind="error",
                    source="daemon",
                    summary=f"Loop tick error: {exc}",
                    status="error",
                )

            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self.settings.loop_tick_seconds,
                )
            except asyncio.TimeoutError:
                continue

    async def tick_once(self) -> None:
        """Execute a single zero-token poll, salience evaluation, and fatigue check."""
        # 1. Poll all pluggable input sources
        polled_events: list[IncomingEvent] = []
        for src in self.sources:
            try:
                batch = await src.poll()
                if batch:
                    polled_events.extend(batch)
            except Exception as exc:
                logger.warning("Source %s poll failed: %s", src.name, exc)

        # 2. Ingest, filter, score salience, and conditionally wake
        for event in polled_events:
            await self.process_event(event)

        # 3. Recompute fatigue and trigger sleep consolidation if warranted
        fatigue = await self.compute_current_fatigue()
        self.last_fatigue = fatigue
        if fatigue.should_sleep and fatigue.trigger_reason is not None:
            await self.trigger_sleep(reason=fatigue.trigger_reason)

    async def process_event(self, event: IncomingEvent) -> ChatReply | None:
        """Evaluate one incoming event through EventIngestor -> compute_salience -> optional wake."""
        decision = await self.ingestor.evaluate(event)
        if not decision.accepted:
            self._record_activity(
                kind="dropped",
                source=event.source,
                summary=f"Dropped event ({decision.drop_reason}): {event.content[:80]}",
                status=decision.drop_reason or "dropped",
            )
            if event.source == "chat":
                self.chat_source.resolve(
                    event.id,
                    ChatReply(
                        reply=f"Ignored event ({decision.drop_reason}).",
                        salience=0.0,
                        status=decision.drop_reason or "dropped",
                    ),
                )
            await self._broadcast_telemetry()
            return None

        # Compute local embedding only when needed (never on hard override or heartbeat)
        event_vec: np.ndarray | None = None
        if not (event.hard_override or event.source == "chat" or event.priority == "high"):
            if event.source != "heartbeat":
                event_vec = self.embedder.embed(event.content)

        # Gather recent buffer vectors, priority vectors, and elapsed idle time
        recent_pairs = await self.store.list_fast_buffer_with_vectors(limit_recent=20)
        recent_vecs = [vec for _, vec in recent_pairs]
        prio_pairs = await self.store.get_active_priority_vectors()
        prio_vecs = [vec for _, vec in prio_pairs]

        breaker_row = await self.store.get_breaker_state_row()
        last_wake = parse_iso(breaker_row.get("last_wake_at"))
        last_sleep = parse_iso(breaker_row.get("last_sleep_at")) or utc_now()
        ref_time = last_wake or last_sleep
        idle_seconds = max(0.0, (utc_now() - ref_time).total_seconds())

        salience = compute_salience(
            event,
            event_embedding=event_vec,
            recent_buffer_embeddings=recent_vecs,
            priority_embeddings=prio_vecs,
            idle_seconds=idle_seconds,
            settings=self.settings,
        )
        self.last_salience = salience

        if not salience.should_wake:
            if event.source != "heartbeat":
                self._record_activity(
                    kind="gated",
                    source=event.source,
                    summary=f"Below threshold ({salience.score:.2f} <= {salience.threshold:.2f}): {event.content[:80]}",
                    salience=salience.score,
                    status="ignored",
                )
            await self._broadcast_telemetry()
            return None

        # Salience exceeded wake threshold -> invoke bounded Claude Haiku 5.5 wake-up
        reply = await self._handle_wake_up(event, salience, event_vec)
        if event.source == "chat":
            self.chat_source.resolve(event.id, reply)
        await self._broadcast_telemetry()
        return reply

    async def send_chat(self, message: str, *, is_correction: bool = False) -> ChatReply:
        """Submit a direct user chat message through ChatSource and await the agent's reply."""
        self.chat_history.append(
            {
                "role": "user",
                "actor_id": "user",
                "content": message,
                "is_correction": is_correction,
                "timestamp": utc_now_iso(),
            }
        )
        _, fut = self.chat_source.submit(message, actor_id="user", is_correction=is_correction)
        if self._loop_task is None:
            # If running synchronously without background loop (e.g. unit tests / one-shot CLI)
            await self.tick_once()
        reply = await fut
        return reply

    async def push_webhook(
        self,
        *,
        content: str,
        actor_id: str = "webhook",
        payload: dict[str, Any] | None = None,
        priority: str = "normal",
        hard_override: bool = False,
    ) -> IncomingEvent:
        """Enqueue an incoming HTTP webhook event into the outer loop."""
        prio_val = "high" if priority == "high" else "normal"
        event = self.webhook_source.push(
            content=content,
            actor_id=actor_id,
            payload=payload,
            priority=prio_val,
            hard_override=hard_override,
        )
        return event

    async def compute_current_fatigue(self) -> FatigueResult:
        """Compute live working-memory fatigue from SQLite fast_buffer and clock state."""
        pairs = await self.store.list_fast_buffer_with_vectors()
        episodes = [ep for ep, _ in pairs]
        embeddings = [vec for _, vec in pairs]

        state = await self.store.get_breaker_state_row()
        last_sleep = parse_iso(state.get("last_sleep_at")) or utc_now()
        last_wake = parse_iso(state.get("last_wake_at"))
        last_user = parse_iso(state.get("last_user_activity_at"))

        last_ep_dt = parse_iso(episodes[-1].created_at) if episodes else None
        candidates = [dt for dt in (last_wake, last_user, last_ep_dt) if dt is not None]
        last_activity = max(candidates) if candidates else last_sleep

        return compute_fatigue(
            episodes,
            embeddings,
            last_sleep_at=last_sleep,
            last_activity_at=last_activity,
            last_user_activity_at=last_user,
            settings=self.settings,
        )

    async def trigger_sleep(self, *, reason: str = "manual") -> SleepCycleReport:
        """Run an atomic offline sleep_and_consolidate cycle."""
        async with self._agent_lock:
            fatigue_before = (await self.compute_current_fatigue()).score
            report = await sleep_and_consolidate(
                store=self.store,
                llm_client=self.llm_client,
                embedder=self.embedder,
                circuit_breaker=self.circuit_breaker,
                settings=self.settings,
                trigger_reason=reason,
                fatigue_before=fatigue_before,
            )
            self.last_sleep_report = report
            self.last_fatigue = await self.compute_current_fatigue()
            self._record_activity(
                kind="sleep",
                source="consolidation",
                summary=(
                    f"Sleep ({reason}) -> {report.status}: "
                    f"+{len(report.added_ids)} / ~{len(report.updated_ids)} / "
                    f"-{len(report.archived_ids)} (rejected={len(report.rejected_operations)})"
                    + (f" [{report.abort_reason}]" if report.abort_reason else "")
                ),
                status=report.status,
            )
        await self._broadcast_telemetry()
        return report

    async def _handle_wake_up(
        self,
        event: IncomingEvent,
        salience: SalienceResult,
        precomputed_vec: np.ndarray | None,
    ) -> ChatReply:
        """Execute a bounded, circuit-breaker-protected Claude Haiku 5.5 wake-up."""
        async with self._agent_lock:
            now = utc_now()
            if event.source == "chat":
                await self.store.touch_last_user_activity(now)

            # Check circuit breaker before invoking frontier LLM
            try:
                await self.circuit_breaker.check_before_call(is_wake_start=True, now=now)
            except CircuitBreakerTripped as exc:
                await self.store.log_wake(
                    source=event.source,
                    salience=salience.score,
                    effort=self.settings.wake_effort,
                    usd_cost=0.0,
                    steps=0,
                    status="circuit_tripped",
                )
                self._record_activity(
                    kind="breaker",
                    source=event.source,
                    summary=str(exc),
                    salience=salience.score,
                    status="circuit_tripped",
                )
                reply = ChatReply(
                    reply=f"Circuit breaker active: {exc.detail}",
                    salience=salience.score,
                    status="circuit_tripped",
                )
                if event.source == "chat":
                    self._append_agent_chat(reply)
                return reply

            core_memory = await self.store.list_core_memory()
            active_priorities = await self.store.list_active_priorities()
            recent_episodes = await self.store.list_fast_buffer()

            system_blocks = build_cached_system_blocks(
                agent_name=self.settings.agent_name,
                core_memory=core_memory,
                active_priorities=active_priorities,
            )
            user_prompt = build_wake_user_message(
                event_source=event.source,
                actor_id=event.actor_id,
                content=event.content,
                salience_score=salience.score,
                recent_episodes=recent_episodes,
            )
            messages: list[dict[str, Any]] = [{"role": "user", "content": user_prompt}]
            daily_spend = await self.store.get_daily_spend(now.strftime("%Y-%m-%d"))
            search_quota_remaining = max(
                0, self.settings.max_web_searches_per_day - daily_spend.web_searches
            )
            tool_defs = self.tools.tool_definitions(
                event_source=event.source,
                search_quota_remaining=search_quota_remaining,
            )

            wake_guard = self.circuit_breaker.new_wake_guard()
            all_thoughts: list[str] = []
            tool_records: list[ToolCallRecord] = []
            total_usage = TokenUsage()
            total_usd = 0.0
            final_text = ""
            status = "ok"
            had_error = False
            retried_compact = False
            active_container_id: str | None = None

            while True:
                try:
                    # Enforce hard daily USD cap before every iteration
                    await self.circuit_breaker.check_before_call(is_wake_start=False)
                    response = await self.llm_client.create_message(
                        system=system_blocks,
                        messages=messages,
                        effort=self.settings.wake_effort,
                        max_tokens=self.settings.max_output_tokens_wake,
                        tools=tool_defs,
                        container=active_container_id,
                    )
                    if response.container_id:
                        active_container_id = response.container_id
                    call_usd = await self.circuit_breaker.record_usage(response.usage)
                    total_usd += call_usd
                    total_usage.input_tokens += response.usage.input_tokens
                    total_usage.cache_read_tokens += response.usage.cache_read_tokens
                    total_usage.cache_creation_tokens += response.usage.cache_creation_tokens
                    total_usage.output_tokens += response.usage.output_tokens
                    total_usage.web_search_requests += response.usage.web_search_requests
                    tool_records.extend(response.server_tool_calls)
                except MissingAPIKeyError as exc:
                    status = "error"
                    had_error = True
                    final_text = str(exc)
                    break
                except CircuitBreakerTripped as exc:
                    status = "circuit_tripped"
                    had_error = True
                    final_text = f"Halted wake-up: {exc.detail}"
                    break
                except Exception as exc:
                    delay = await self.circuit_breaker.record_error()
                    status = "error"
                    had_error = True
                    final_text = (
                        f"LLM call failed ({exc}). Entering exponential backoff for {delay}s."
                    )
                    break

                # Inspect thinking blocks & guard against semantic spinning
                try:
                    for thought in response.thinking:
                        all_thoughts.append(thought)
                        wake_guard.record_thought_and_check_spin(thought)
                except CircuitBreakerTripped as exc:
                    status = "circuit_tripped"
                    had_error = True
                    final_text = response.text or f"Wake-up halted: {exc.detail}"
                    break

                # Gracefully handle Claude Haiku 5.5 stop reasons
                if response.is_refusal:
                    status = "refusal"
                    final_text = (
                        response.text
                        or "I cannot fulfill this request due to safety policies."
                    )
                    await self.circuit_breaker.record_success()
                    break

                if response.is_context_exceeded:
                    if not retried_compact:
                        retried_compact = True
                        compact_prompt = build_wake_user_message(
                            event_source=event.source,
                            actor_id=event.actor_id,
                            content=event.content,
                            salience_score=salience.score,
                            recent_episodes=[],
                        )
                        messages = [{"role": "user", "content": compact_prompt}]
                        continue
                    status = "context_exceeded"
                    had_error = True
                    final_text = (
                        "Working context exceeded model window. "
                        "Triggering consolidation on the next tick to free working memory."
                    )
                    break

                # Server-side tool loop paused (e.g., long code_execution): continue cleanly with user turn
                if response.is_pause_turn:
                    try:
                        wake_guard.check_and_increment_step()
                    except CircuitBreakerTripped as exc:
                        status = "circuit_tripped"
                        had_error = True
                        final_text = response.text or f"Stopped paused server loop: {exc.detail}"
                        break
                    messages.append({"role": "assistant", "content": response.raw_content})
                    messages.append(
                        {"role": "user", "content": "Continue and complete your response."}
                    )
                    continue

                # Tool execution loop
                if response.stop_reason == "tool_use" and response.tool_uses:
                    try:
                        wake_guard.check_and_increment_step()
                    except CircuitBreakerTripped as exc:
                        status = "circuit_tripped"
                        had_error = True
                        final_text = response.text or f"Stopped tool loop: {exc.detail}"
                        break

                    # Append assistant turn preserving thinking & tool_use blocks
                    messages.append({"role": "assistant", "content": response.raw_content})

                    tool_result_blocks: list[dict[str, Any]] = []
                    loop_aborted = False
                    for tc in response.tool_uses:
                        try:
                            wake_guard.check_tool_call(tc.name, tc.input)
                        except CircuitBreakerTripped as exc:
                            tool_records.append(
                                ToolCallRecord(
                                    name=tc.name,
                                    args=tc.input,
                                    result=exc.detail,
                                    is_error=True,
                                    blocked_reason=exc.reason,
                                )
                            )
                            status = "circuit_tripped"
                            had_error = True
                            final_text = f"Stopped duplicate tool loop: {exc.detail}"
                            loop_aborted = True
                            break

                        result_text, is_err = await self.tools.execute(tc.name, tc.input)
                        if is_err:
                            had_error = True
                        tool_records.append(
                            ToolCallRecord(
                                name=tc.name,
                                args=tc.input,
                                result=result_text,
                                is_error=is_err,
                            )
                        )
                        tool_result_blocks.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": tc.id,
                                "content": result_text,
                                "is_error": is_err,
                            }
                        )

                    if loop_aborted:
                        break

                    # Always end messages array with a `user` turn (never assistant prefill)
                    messages.append({"role": "user", "content": tool_result_blocks})
                    continue

                # Standard completion (`end_turn`, `max_tokens`, etc.)
                final_text = response.text
                await self.circuit_breaker.record_success()
                break

            # Persist wake log & update wake clock
            await self.store.touch_last_wake()
            await self.store.log_wake(
                source=event.source,
                salience=salience.score,
                effort=self.settings.wake_effort,
                usd_cost=total_usd,
                steps=wake_guard.steps_taken,
                status=status,
            )

            # Persist episode into fast_buffer with local embedding
            had_correction = detect_user_correction(event)
            episode_summary = f"[{event.source}] {event.content[:400]}"
            episode_vec = (
                precomputed_vec
                if precomputed_vec is not None
                else self.embedder.embed(f"{episode_summary}\n{final_text[:300]}")
            )
            approx_episode_tokens = (
                total_usage.output_tokens
                + estimate_token_count(event.content)
                + estimate_token_count(final_text)
                + sum(estimate_token_count(t.result) for t in tool_records)
            )
            episode = FastBufferEpisode(
                source=event.source,
                actor_id=event.actor_id,
                event_payload=event.payload,
                salience_score=salience.score,
                summary=episode_summary,
                thoughts=all_thoughts,
                tool_calls=tool_records,
                agent_response=final_text,
                token_count=max(1, approx_episode_tokens),
                had_error=had_error,
                had_correction=had_correction,
            )
            await self.store.add_fast_buffer_episode(episode, episode_vec)
            self.last_fatigue = await self.compute_current_fatigue()

            self._record_activity(
                kind="wake",
                source=event.source,
                summary=f"Woke ({status}, ${total_usd:.5f}): {event.content[:75]}",
                salience=salience.score,
                status=status,
            )

            reply = ChatReply(
                actor_id="agent",
                episode_id=episode.id,
                reply=final_text,
                thoughts=all_thoughts,
                tool_calls=tool_records,
                salience=salience.score,
                status=status,
                usd_cost=total_usd,
                usage=total_usage,
            )
            if event.source == "chat":
                self._append_agent_chat(reply, source="chat", trigger_actor=event.actor_id)
            elif event.source in ("webhook", "file_watcher") and final_text:
                self._append_agent_chat(
                    reply,
                    source=event.source,
                    trigger_actor=event.actor_id,
                    trigger_preview=event.content.splitlines()[0][:140],
                )
            return reply

    def _append_agent_chat(
        self,
        reply: ChatReply,
        *,
        source: str = "chat",
        trigger_actor: str = "user",
        trigger_preview: str | None = None,
    ) -> None:
        self.chat_history.append(
            {
                "role": "assistant",
                "actor_id": reply.actor_id,
                "source": source,
                "trigger_actor": trigger_actor,
                "trigger_preview": trigger_preview,
                "episode_id": reply.episode_id,
                "content": reply.reply,
                "thoughts": reply.thoughts,
                "tool_calls": [tc.model_dump() for tc in reply.tool_calls],
                "salience": reply.salience,
                "status": reply.status,
                "usd_cost": reply.usd_cost,
                "timestamp": reply.timestamp,
            }
        )

    def _record_activity(
        self,
        *,
        kind: str,
        source: str,
        summary: str,
        salience: float | None = None,
        status: str = "ok",
    ) -> None:
        self.activity_feed.appendleft(
            ActivityRecord(
                timestamp=utc_now_iso(),
                kind=kind,
                source=source,
                summary=summary,
                salience=salience,
                status=status,
            )
        )

    # -------------------------------------------------------------------------
    # Telemetry & SSE subscription for local Web Dashboard / CLI
    # -------------------------------------------------------------------------

    def subscribe_sse(self) -> asyncio.Queue[dict[str, Any]]:
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=16)
        self._sse_subscribers.add(q)
        return q

    def unsubscribe_sse(self, q: asyncio.Queue[dict[str, Any]]) -> None:
        self._sse_subscribers.discard(q)

    async def get_telemetry_snapshot(self) -> dict[str, Any]:
        """Build complete live telemetry payload for dashboard and CLI status."""
        fatigue = await self.compute_current_fatigue()
        self.last_fatigue = fatigue
        breaker = await self.circuit_breaker.snapshot()
        core_memory = await self.store.list_core_memory()
        archive = await self.store.list_memory_archive(limit=50)
        priorities = await self.store.list_active_priorities()
        buffer_episodes = await self.store.list_fast_buffer()
        sleep_cycles = await self.store.list_sleep_cycles(limit=15)
        recent_wakes = await self.store.list_recent_wakes(limit=20)

        return {
            "agent_name": self.settings.agent_name,
            "model": self.settings.anthropic_model,
            "web_search_provider": self.settings.web_search_provider,
            "wake_threshold": self.settings.wake_threshold,
            "sleep_threshold": self.settings.sleep_threshold,
            "timestamp": utc_now_iso(),
            "salience": self.last_salience.model_dump() if self.last_salience else None,
            "fatigue": fatigue.model_dump(),
            "circuit_breaker": breaker.model_dump(),
            "ingestor": {
                "dropped_self_triggers": self.ingestor.dropped_self_triggers,
                "dropped_duplicates": self.ingestor.dropped_duplicates,
            },
            "core_memory": [m.model_dump() for m in core_memory],
            "memory_archive": [a.model_dump() for a in archive],
            "active_priorities": [p.model_dump() for p in priorities],
            "fast_buffer": [ep.model_dump() for ep in buffer_episodes],
            "sleep_cycles": sleep_cycles,
            "recent_wakes": [w.model_dump() for w in recent_wakes],
            "activity_feed": [a.to_dict() for a in self.activity_feed],
            "chat_history": list(self.chat_history),
            "watchers": dict(self._watcher_status),
        }

    async def _broadcast_telemetry(self) -> None:
        if not self._sse_subscribers:
            return
        payload = await self.get_telemetry_snapshot()
        dead: list[asyncio.Queue[dict[str, Any]]] = []
        for q in self._sse_subscribers:
            try:
                if q.full():
                    q.get_nowait()
                q.put_nowait(payload)
            except Exception:
                dead.append(q)
        for q in dead:
            self._sse_subscribers.discard(q)
