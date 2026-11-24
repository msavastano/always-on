"""Typed runtime configuration loaded from environment / .env."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

EffortLevel = Literal["low", "medium", "high", "xhigh", "max"]
WebSearchProviderType = Literal["ddg", "brave", "tavily", "searxng", "anthropic", "disabled"]
CodeExecutionToolVersion = Literal[
    "code_execution_20250825",
    "code_execution_20260120",
    "code_execution_20260521",
]
VercelPriorityPolicy = Literal["smart", "failures_only", "all_high"]


class Settings(BaseSettings):
    """Application settings for the Always-On CLS AI Agent."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Anthropic / Claude Haiku 5.5 ---
    anthropic_api_key: str = Field(default="", description="Anthropic API key")
    anthropic_model: str = Field(
        default="claude-haiku-5-5",
        description="Target Claude model ID (defaults to claude-haiku-5-5)",
    )
    wake_effort: EffortLevel = Field(
        default="low",
        description="Adaptive thinking effort for routine wake-ups",
    )
    sleep_effort: EffortLevel = Field(
        default="medium",
        description="Adaptive thinking effort for offline sleep consolidation",
    )
    max_output_tokens_wake: int = Field(default=2048, ge=256, le=16384)
    max_output_tokens_sleep: int = Field(default=4096, ge=512, le=16384)

    # --- Token Pricing (USD per 1M tokens) ---
    price_input_per_m: float = Field(default=0.10, ge=0.0)
    price_cache_read_per_m: float = Field(default=0.01, ge=0.0)
    price_cache_write_per_m: float = Field(default=0.125, ge=0.0)
    price_output_per_m: float = Field(default=0.50, ge=0.0)
    price_anthropic_web_search_per_1k: float = Field(default=10.0, ge=0.0)
    long_context_threshold_tokens: int = Field(default=100_000, ge=1)
    long_context_multiplier: float = Field(default=5.0, ge=1.0)

    # --- Web Search & Webpage Fetch Tools ---
    web_search_provider: WebSearchProviderType = Field(default="ddg")
    brave_search_api_key: str = Field(default="")
    tavily_api_key: str = Field(default="")
    searxng_url: str = Field(default="")
    web_search_max_results: int = Field(default=5, ge=1, le=10)
    web_search_max_snippet_chars: int = Field(default=300, ge=80, le=1000)
    web_search_cache_ttl_seconds: int = Field(default=1800, ge=0)
    web_search_timeout_seconds: float = Field(default=10.0, gt=0.5, le=60.0)
    max_web_searches_per_day: int = Field(default=50, ge=1)
    allow_search_on_heartbeat: bool = Field(default=False)
    enable_fetch_webpage: bool = Field(default=True)
    fetch_webpage_max_chars: int = Field(default=4000, ge=500, le=12000)
    anthropic_web_search_max_uses: int = Field(default=3, ge=1, le=10)

    # --- Anthropic Native Server-Side Code Execution ---
    enable_code_execution: bool = Field(
        default=True,
        description="Enable Anthropic server-side sandboxed code_execution tool",
    )
    code_execution_tool_version: CodeExecutionToolVersion = Field(
        default="code_execution_20260120",
        description="Anthropic server-side code execution tool type version",
    )
    allow_code_execution_on_background: bool = Field(
        default=False,
        description="If false, restrict server-side code_execution strictly to direct user chat wake-ups",
    )

    # --- Storage & Paths ---
    db_path: Path = Field(default=Path("data/agent.db"))
    inbox_dir: Path = Field(default=Path("data/inbox"))
    notes_dir: Path = Field(default=Path("data/notes"))

    # --- Local Embeddings ---
    embedding_provider: Literal["fastembed", "mock"] = Field(default="fastembed")
    embedding_model: str = Field(default="BAAI/bge-small-en-v1.5")

    # --- Outer Event Loop & Sources ---
    loop_tick_seconds: float = Field(default=1.0, gt=0.05)
    dedup_cooldown_seconds: int = Field(default=300, ge=1)
    heartbeat_emit_seconds: int = Field(default=60, ge=5)
    heartbeat_interval_seconds: int = Field(default=3600, ge=30)

    # --- Local Salience Gating ---
    wake_threshold: float = Field(default=0.60, ge=0.0, le=1.0)
    salience_weight_novelty: float = Field(default=0.45, ge=0.0, le=1.0)
    salience_weight_relevance: float = Field(default=0.45, ge=0.0, le=1.0)
    salience_weight_temporal: float = Field(default=0.10, ge=0.0, le=1.0)
    temporal_pressure_asymptote: float = Field(default=0.65, gt=0.0, le=1.0)
    temporal_pressure_steepness: float = Field(default=2.6, gt=0.1)

    # --- Fast Working Buffer & Fatigue ---
    max_working_budget_tokens: int = Field(default=16000, ge=512)
    sleep_threshold: float = Field(default=0.85, ge=0.1, le=1.0)
    critical_saturation_threshold: float = Field(default=0.95, ge=0.5, le=1.0)
    idle_sleep_seconds: int = Field(default=1800, ge=30)
    active_conversation_grace_seconds: int = Field(default=300, ge=0)
    fatigue_drift_horizon_hours: float = Field(default=24.0, gt=0.1)

    # --- Consolidation Guardrails ---
    churn_max_archive_ratio: float = Field(default=0.25, gt=0.0, le=1.0)
    churn_min_core_items: int = Field(default=10, ge=1)

    # --- Circuit Breakers & Spend Guardrails ---
    max_daily_usd: float = Field(default=1.00, gt=0.0)
    max_wakes_per_hour: int = Field(default=30, ge=1)
    max_steps_per_wake: int = Field(default=6, ge=1)
    semantic_spin_threshold: float = Field(default=0.92, gt=0.5, le=1.0)
    backoff_base_seconds: int = Field(default=60, ge=1)
    backoff_max_seconds: int = Field(default=3600, ge=10)

    # --- Web Dashboard & Identity ---
    host: str = Field(default="127.0.0.1")
    port: int = Field(default=8420, ge=1, le=65535)
    agent_name: str = Field(default="Aion")
    webhook_secret: str = Field(
        default="",
        description="Optional shared secret required on POST /api/events/webhook if non-empty",
    )

    # --- Vercel Deployment Webhook & Poller ---
    vercel_webhook_secret: str = Field(
        default="",
        description="Vercel Webhook Secret for HMAC-SHA1 x-vercel-signature verification",
    )
    vercel_api_token: str = Field(
        default="",
        description="Vercel Personal Access Token for watch-vercel polling & error log enrichment",
    )
    vercel_team_id: str = Field(default="")
    vercel_project_filter: str = Field(
        default="",
        description="Optional comma-separated Vercel project names/IDs (empty = all projects)",
    )
    vercel_priority_policy: VercelPriorityPolicy = Field(default="smart")

    # --- Managed Background Watchers (run inside `always-on run`) ---
    enable_claude_code_watcher: bool = Field(
        default=False,
        description="Run the Claude Code release watcher automatically inside `always-on run`",
    )
    claude_code_watcher_interval_seconds: int = Field(default=900, ge=30)
    claude_code_watcher_priority: Literal["normal", "high"] = Field(default="high")
    claude_code_watcher_state_path: Path = Field(
        default=Path("data/claude_code_watcher_state.json")
    )
    github_token: str = Field(
        default="",
        description="Optional GitHub token for 5,000 req/hr API quota",
    )

    enable_vercel_watcher: bool = Field(
        default=False,
        description="Run the zero-tunnel Vercel deployment poller automatically inside `always-on run`",
    )
    vercel_watcher_interval_seconds: int = Field(default=60, ge=15)
    vercel_watcher_target: str = Field(
        default="",
        description="Optional environment filter: 'production' or 'preview' (empty = both)",
    )
    vercel_watcher_state_path: Path = Field(
        default=Path("data/vercel_watcher_state.json")
    )

    # --- 2-Way Discord Bot Bridge ---
    enable_discord_bot: bool = Field(
        default=False,
        description="Run the 2-way Discord bot gateway inside `always-on run`",
    )
    discord_bot_token: str = Field(default="")
    discord_allowed_user_ids: str = Field(
        default="",
        description="Comma-separated numeric Discord user IDs allowed to talk to the bot (fail-closed if empty)",
    )
    discord_notify_channel_id: str = Field(
        default="",
        description="Numeric Discord channel ID for proactive alerts and direct chat without @mention",
    )
    discord_passive_channel_ids: str = Field(
        default="",
        description="Optional comma-separated channel IDs ingested at priority=normal via local salience gating",
    )
    discord_quiet_hours: str = Field(
        default="",
        description="Optional local HH:MM-HH:MM window suppressing non-critical proactive Discord alerts (e.g. 22:00-08:00)",
    )
    discord_guild_id: str = Field(
        default="",
        description="Optional numeric server/guild ID for instant slash command sync",
    )

    # --- MCP Client Support (External Tool Servers) ---
    enable_mcp: bool = Field(
        default=True,
        description="Enable MCP client connections from mcp_config_path or mcp_servers_json",
    )
    mcp_config_path: Path = Field(
        default=Path("mcp_servers.json"),
        description="Path to Claude-Desktop-style mcpServers JSON config",
    )
    mcp_servers_json: str = Field(
        default="",
        description="Optional inline JSON string override for mcpServers",
    )
    mcp_startup_timeout_seconds: float = Field(default=45.0, gt=1.0, le=120.0)
    mcp_tool_timeout_seconds: float = Field(default=30.0, gt=1.0, le=300.0)
    mcp_max_result_chars: int = Field(default=8000, ge=500, le=32000)
    allow_mcp_on_background: bool = Field(
        default=False,
        description="If false, restrict MCP tools to direct user chat wake-ups unless a server sets allow_on_background=true",
    )

    def calculate_cost_usd(
        self,
        *,
        input_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_creation_tokens: int = 0,
        output_tokens: int = 0,
        web_search_requests: int = 0,
    ) -> float:
        """Compute USD cost from actual API usage tokens + Anthropic server-side searches."""
        total_prompt = input_tokens + cache_read_tokens + cache_creation_tokens
        mult = (
            self.long_context_multiplier
            if total_prompt > self.long_context_threshold_tokens
            else 1.0
        )
        usd = (
            (input_tokens / 1_000_000.0) * self.price_input_per_m * mult
            + (cache_read_tokens / 1_000_000.0) * self.price_cache_read_per_m * mult
            + (cache_creation_tokens / 1_000_000.0) * self.price_cache_write_per_m * mult
            + (output_tokens / 1_000_000.0) * self.price_output_per_m * mult
            + (web_search_requests / 1_000.0) * self.price_anthropic_web_search_per_1k
        )
        return round(usd, 8)
