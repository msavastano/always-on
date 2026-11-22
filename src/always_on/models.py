"""Domain models for events, memory, cognitive scoring, consolidation, and safety."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

MemoryCategory = Literal["fact", "preference", "project", "lesson"]
EventSourceType = Literal["chat", "file_watcher", "webhook", "heartbeat"]
ConsolidationOpType = Literal["ADD", "UPDATE", "ARCHIVE"]


def utc_now() -> datetime:
    """Return timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    """Return ISO-8601 UTC timestamp string."""
    return utc_now().isoformat()


def new_id(prefix: str) -> str:
    """Generate a compact readable unique ID."""
    return f"{prefix}_{uuid4().hex[:10]}"


class IncomingEvent(BaseModel):
    """Normalized event entering the outer event loop."""

    id: str = Field(default_factory=lambda: new_id("evt"))
    source: EventSourceType
    actor_id: str = Field(default="external")
    content: str
    payload: dict[str, Any] = Field(default_factory=dict)
    priority: Literal["normal", "high"] = "normal"
    hard_override: bool = False
    is_correction: bool = False
    created_at: datetime = Field(default_factory=utc_now)


class SalienceBreakdown(BaseModel):
    """Per-signal breakdown of local zero-token salience scoring."""

    hard_override: bool = False
    novelty: float = Field(default=0.0, ge=0.0, le=1.0)
    goal_relevance: float = Field(default=0.0, ge=0.0, le=1.0)
    temporal_pressure: float = Field(default=0.0, ge=0.0, le=1.0)
    idle_seconds: float = Field(default=0.0, ge=0.0)


class SalienceResult(BaseModel):
    """Output of compute_salience()."""

    score: float = Field(ge=0.0, le=1.0)
    should_wake: bool
    threshold: float
    breakdown: SalienceBreakdown


class FatigueBreakdown(BaseModel):
    """Per-signal breakdown of working-buffer fatigue scoring."""

    token_saturation: float = Field(default=0.0, ge=0.0, le=1.0)
    semantic_fragmentation: float = Field(default=0.0, ge=0.0, le=1.0)
    lesson_error_load: float = Field(default=0.0, ge=0.0, le=1.0)
    time_drift: float = Field(default=0.0, ge=0.0, le=1.0)
    buffer_entries: int = 0
    buffer_tokens: int = 0
    error_count: int = 0
    correction_count: int = 0
    hours_since_sleep: float = 0.0
    idle_seconds: float = 0.0
    is_mid_conversation: bool = False


class FatigueResult(BaseModel):
    """Output of compute_fatigue()."""

    score: float = Field(ge=0.0, le=1.0)
    should_sleep: bool
    trigger_reason: Literal["fatigue", "idle", "critical_saturation"] | None = None
    breakdown: FatigueBreakdown


class MemoryItem(BaseModel):
    """Active long-term Core Memory item."""

    id: str = Field(default_factory=lambda: new_id("mem"))
    category: MemoryCategory
    content: str
    pinned: bool = False
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    source_episodes: list[str] = Field(default_factory=list)
    version: int = 1
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)


class ArchivedMemoryItem(BaseModel):
    """Immutable historical snapshot of an overwritten or archived memory item."""

    archive_id: int
    memory_id: str
    version: int
    category: MemoryCategory
    content: str
    pinned: bool
    confidence: float
    source_episodes: list[str]
    action: Literal["UPDATE_SUPERSEDED", "ARCHIVE"]
    reason: str
    sleep_cycle_id: str
    archived_at: str


class ActivePriority(BaseModel):
    """An active goal/priority statement used for salience goal-relevance scoring."""

    id: int | None = None
    rank: int
    description: str
    updated_at: str = Field(default_factory=utc_now_iso)


class ToolCallRecord(BaseModel):
    """Record of one tool invocation during a wake-up."""

    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    result: str = ""
    is_error: bool = False
    blocked_reason: str | None = None


class FastBufferEpisode(BaseModel):
    """One wake-up episode accumulated in the fast working buffer."""

    id: str = Field(default_factory=lambda: new_id("ep"))
    created_at: str = Field(default_factory=utc_now_iso)
    source: EventSourceType
    actor_id: str
    event_payload: dict[str, Any] = Field(default_factory=dict)
    salience_score: float = Field(ge=0.0, le=1.0)
    summary: str
    thoughts: list[str] = Field(default_factory=list)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    agent_response: str = ""
    token_count: int = Field(default=0, ge=0)
    had_error: bool = False
    had_correction: bool = False


class ConsolidationOperation(BaseModel):
    """Single atomic diff operation proposed by the offline sleep pass."""

    op: ConsolidationOpType
    target_id: str | None = None
    category: MemoryCategory | None = None
    content: str | None = None
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    pinned: bool = False
    source_episode_ids: list[str] = Field(default_factory=list)
    reason: str = ""

    @field_validator("content")
    @classmethod
    def _strip_content(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return " ".join(value.strip().split())


class ConsolidationPlan(BaseModel):
    """Structured JSON output emitted by sleep_and_consolidate()."""

    operations: list[ConsolidationOperation] = Field(default_factory=list)
    active_priorities: list[str] = Field(default_factory=list)
    reflection: str = ""


class RejectedOperation(BaseModel):
    """Audit record of an operation rejected by deterministic guardrails."""

    op: ConsolidationOperation
    guardrail: str
    reason: str


class SleepCycleReport(BaseModel):
    """Result of executing sleep_and_consolidate()."""

    sleep_cycle_id: str
    status: Literal["committed", "aborted", "skipped", "failed"]
    trigger_reason: str
    fatigue_before: float
    episodes_processed: int
    added_ids: list[str] = Field(default_factory=list)
    updated_ids: list[str] = Field(default_factory=list)
    archived_ids: list[str] = Field(default_factory=list)
    rejected_operations: list[RejectedOperation] = Field(default_factory=list)
    active_priorities: list[str] = Field(default_factory=list)
    reflection: str = ""
    abort_reason: str | None = None
    usd_cost: float = 0.0


class TokenUsage(BaseModel):
    """Actual token usage (and optional Anthropic server-side searches) from API response."""

    input_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    output_tokens: int = 0
    web_search_requests: int = 0

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.cache_read_tokens
            + self.cache_creation_tokens
            + self.output_tokens
        )


class DailySpend(BaseModel):
    """Persistent daily token, web search, & USD accounting."""

    date_utc: str
    input_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    output_tokens: int = 0
    web_searches: int = 0
    usd_spent: float = 0.0
    updated_at: str = Field(default_factory=utc_now_iso)


class CircuitBreakerSnapshot(BaseModel):
    """Live persisted circuit breaker state."""

    daily_spend: DailySpend
    max_daily_usd: float
    max_web_searches_per_day: int = 50
    wakes_last_hour: int
    max_wakes_per_hour: int
    consecutive_errors: int
    cooldown_until: str | None = None
    in_cooldown: bool = False
    daily_cap_exceeded: bool = False
    daily_search_cap_exceeded: bool = False
    hourly_rate_exceeded: bool = False
    last_sleep_at: str
    last_wake_at: str | None = None
    last_user_activity_at: str | None = None


class WakeLogEntry(BaseModel):
    """One row from wake_log."""

    id: int
    timestamp: str
    source: str
    salience: float
    effort: str
    usd_cost: float
    steps: int
    status: str


class ChatReply(BaseModel):
    """Structured response sent back to a chat user (tagged actor_id='agent')."""

    actor_id: Literal["agent"] = "agent"
    episode_id: str | None = None
    reply: str
    thoughts: list[str] = Field(default_factory=list)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    salience: float = 1.0
    status: str = "ok"
    usd_cost: float = 0.0
    usage: TokenUsage = Field(default_factory=TokenUsage)
    timestamp: str = Field(default_factory=utc_now_iso)


def consolidation_json_schema() -> dict[str, Any]:
    """Strict JSON Schema for Anthropic output_config.format constrained decoding.

    Uses `anyOf` for nullable fields (including nullable enums) to match
    Anthropic's strict JSON Schema validator and `anthropic.transform_schema()`.
    """
    return {
        "type": "object",
        "properties": {
            "operations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "op": {
                            "type": "string",
                            "enum": ["ADD", "UPDATE", "ARCHIVE"],
                        },
                        "target_id": {
                            "anyOf": [
                                {"type": "string"},
                                {"type": "null"},
                            ],
                        },
                        "category": {
                            "anyOf": [
                                {
                                    "type": "string",
                                    "enum": ["fact", "preference", "project", "lesson"],
                                },
                                {"type": "null"},
                            ],
                        },
                        "content": {
                            "anyOf": [
                                {"type": "string"},
                                {"type": "null"},
                            ],
                        },
                        "confidence": {
                            "type": "number",
                        },
                        "pinned": {
                            "type": "boolean",
                        },
                        "source_episode_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "reason": {
                            "type": "string",
                        },
                    },
                    "required": [
                        "op",
                        "target_id",
                        "category",
                        "content",
                        "confidence",
                        "pinned",
                        "source_episode_ids",
                        "reason",
                    ],
                    "additionalProperties": False,
                },
            },
            "active_priorities": {
                "type": "array",
                "items": {"type": "string"},
            },
            "reflection": {
                "type": "string",
            },
        },
        "required": ["operations", "active_priorities", "reflection"],
        "additionalProperties": False,
    }
