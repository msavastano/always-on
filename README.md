# Always-On CLS AI Agent

A production-ready, local-first, always-on personal AI agent daemon inspired by **Complementary Learning Systems (CLS)** theory and powered by **Claude Haiku 5.5** (`claude-haiku-5-5`).

Instead of waking up stateless on every prompt, the agent runs a continuous zero-token outer event loop, gates LLM calls with local ONNX embeddings (`FastEmbed`), accumulates episodic working memory in a fast SQLite buffer, and runs transactional offline **"sleep and consolidate"** cycles to safely evolve long-term Core Memory.

---

## Architecture Highlights

### 1. Strict Claude Haiku 5.5 Client (`src/always_on/llm/client.py`)
- Default model `claude-haiku-5-5` (configurable via `ANTHROPIC_MODEL` in `.env`).
- Never sends forbidden sampling parameters (`temperature`, `top_p`, `top_k`).
- Always uses adaptive thinking (`thinking={"type": "adaptive"}`) with cost/depth controlled via `output_config={"effort": ...}`:
  - `low` for routine wake-ups and tool actions.
  - `medium` (configurable to `high`) + `output_config.format` (`json_schema`) for offline sleep consolidation.
- **Prompt Caching:** System instructions + active Core Memory form a deterministic ≥ 512-token prefix with `cache_control={"type": "ephemeral"}`.
- Parses content blocks strictly by `block.type` (`thinking`, `text`, `tool_use`), gracefully handles `refusal` and `model_context_window_exceeded`, and enforces no assistant prefill (every call ends on a `user` turn).

### 2. Zero-Token Outer Event Loop & Pluggable Sources (`src/always_on/daemon.py`, `src/always_on/sources/`)
- Continuous non-blocking `asyncio` loop (1-second tick).
- Built-in pluggable sources:
  - **Direct Chat** (Web Dashboard & CLI)
  - **Local Inbox Watcher** (`data/inbox/` for new/modified files)
  - **HTTP Webhooks** (`POST /api/events/webhook`)
  - **Internal Heartbeat** (clock ticks driving temporal pressure)
- **Self-Trigger Prevention:** Agent outputs are tagged `actor_id="agent"` and dropped by `EventIngestor` before dedup or embedding.
- **Rolling Deduplication Cooldown:** Identical raw event payloads arriving within 5 minutes (configurable) are dropped before computing embeddings.

### 3. Local Salience Gating & Fatigue Scoring (`src/always_on/cognitive/`)
- **`compute_salience()`** (zero LLM tokens): combines **Hard Override** (`1.0` on direct chat / high-priority alerts), **Novelty** (`1 - max_cos_sim` vs. recent `fast_buffer`), **Goal Relevance** (`max_cos_sim` vs. embedded active priorities), and **Temporal Pressure** (smooth exponential curve approaching `0.65` over `HEARTBEAT_INTERVAL_SECONDS`). Wakes only when `salience > WAKE_THRESHOLD` (`0.60`).
- **`compute_fatigue()`**: blends **Token Saturation** (vs. 16k budget), **Semantic Fragmentation** (mean pairwise cosine distance across buffer), **Lesson / Error Load** (weighted tool errors & user corrections), and **Gentle Time Drift** (hours since last sleep). Triggers sleep at `fatigue >= 0.85` or after 30 minutes idle with a non-empty buffer, while guarding active conversations unless token saturation is critical (`> 0.95`).

### 4. Transactional Sleep Consolidation (`src/always_on/cognitive/consolidation.py`)
- Emits atomic diff operations (`ADD`, `UPDATE`, `ARCHIVE`) + 3–5 updated active priorities via constrained JSON Schema output.
- **Deterministic Guardrails (enforced in code before SQLite `BEGIN IMMEDIATE ... COMMIT`):**
  - **Strict Provenance:** Rejects any operation whose `source_episode_ids` are missing or not a subset of current `fast_buffer` IDs.
  - **Pinned Protection:** Rejects any `UPDATE` or `ARCHIVE` targeting a `pinned: true` memory item.
  - **Churn Circuit Breaker:** Aborts the entire commit (preserving `fast_buffer`) if a single pass attempts to `ARCHIVE > 25%` of active Core Memory (when `> 10` items exist).
  - **Soft-Archiving:** Never hard-deletes; every superseded or archived version is written to `memory_archive` with version, timestamp, sleep cycle ID, and reason.

### 5. Disk-Persisted Circuit Breakers (`src/always_on/safety/circuit_breaker.py`)
All spend, rate-limit, search quota, and backoff state lives in SQLite (`data/agent.db`, WAL mode) so crash-restarts never reset limits:
1. **Hard Daily USD Cap** (default `$1.00/day`, computed from actual input, cache read, cache write, output tokens, and any Anthropic server-side web search fees at `$10 / 1,000`).
2. **Daily Web Search Cap** (default `50` searches/day, backed by a 30-minute SQLite query cache so repeat lookups cost `$0` and don't consume quota).
3. **Rolling Hourly Rate Limit** (default `30` wake-ups/hour).
4. **Per-Wake Step Limit** (default `6` tool iterations/wake-up).
5. **Loop Detection:** SHA-256 exact-duplicate tool call blocking + semantic spinning abort (> `0.92` cosine similarity across 3 consecutive thoughts).
6. **Exponential Backoff:** `60s -> 120s -> 240s -> ...` capped at `3600s` (1 hour) on consecutive errors.

### 6. Pluggable Web Search & Webpage Fetch (`src/always_on/tools/web_search.py`)
- Controlled via `WEB_SEARCH_PROVIDER` in `.env`:
  - `ddg` *(default, zero API key via `ddgs`)*
  - `brave` (requires `BRAVE_SEARCH_API_KEY`)
  - `tavily` (requires `TAVILY_API_KEY`)
  - `searxng` (self-hosted, requires `SEARXNG_URL`)
  - `anthropic` (native server-side `web_search_20260318` billed at `$0.01/search` and tracked inside `daily_spend`)
  - `disabled`
- Includes companion `fetch_webpage` tool with SSRF protection (blocks localhost/private/link-local IPs across redirects), HTML script/chrome stripping, and strict character caps to protect the 16k working buffer.
- All external web text is wrapped in `<untrusted_web_content>` tags with prompt-injection and memory-poisoning rules in both wake-up and consolidation prompts. Search tools are omitted on idle `heartbeat` ticks by default (`ALLOW_SEARCH_ON_HEARTBEAT=false`).

### 7. Anthropic Native Server-Side Code Execution (`ENABLE_CODE_EXECUTION=true`)
- Exposes Anthropic's isolated cloud `code_execution` sandbox (`code_execution_20260120` by default) on direct `chat` wake-ups (disabled on background webhooks/heartbeats by default via `ALLOW_CODE_EXECUTION_ON_BACKGROUND=false`).
- Supports multi-turn container reuse and `pause_turn` continuation (always ending on a `user` turn), and surfaces executed commands + `stdout`/`stderr` inside the chat window's collapsible **Tool Calls** panel.

---

## Quick Start

### 1. Prerequisites
- Python 3.11+ (tested on Python 3.13)
- [`uv`](https://docs.astral.sh/uv/) package manager
- An Anthropic API key (`ANTHROPIC_API_KEY`)

### 2. Install & Configure
```bash
# Install dependencies (including FastEmbed ONNX runtime and dev test tools)
uv sync --extra dev

# Create your local .env from the template
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY=sk-ant-...
```

> **Tip (Offline / Zero-Download Dry Run):** Set `EMBEDDING_PROVIDER=mock` in `.env` if you want to test locally without downloading the ~65MB `BAAI/bge-small-en-v1.5` ONNX model on first boot.

### 3. Run the Daemon + Local Web Dashboard
```bash
uv run always-on run
```
Open **http://127.0.0.1:8420** in your browser to:
- Chat directly with the agent (with optional "Flag as user correction" toggle)
- Watch live SSE telemetry (`salience`, `fatigue`, `daily_spend`, circuit breaker state)
- Inspect Active Priorities, Core Memory (with one-click **Pin / Unpin**), immutable **Archive History**, **Fast Buffer**, and **Sleep & Activity** logs
- Simulate incoming webhooks (including testing `actor_id="agent"` self-trigger rejection)
- Manually trigger **Sleep & Consolidate**

### 4. Interactive CLI Commands
In a second terminal while the daemon is running:
```bash
# Interactive terminal chat (/status, /memory, /archive, /sleep, /correct <msg>, /webhook <msg>)
uv run always-on chat

# One-shot telemetry snapshot
uv run always-on status

# Manually trigger a sleep consolidation cycle
uv run always-on sleep
```

### 5. Trigger Events via Inbox Files, Webhooks, or Managed Watchers
- **Single-Command Managed Watchers (Recommended):**
  Enable either or both watchers in `.env` so `uv run always-on run` runs them automatically in the background and shuts everything down cleanly together on a single **Ctrl+C**:
  ```env
  ENABLE_CLAUDE_CODE_WATCHER=true
  ENABLE_VERCEL_WATCHER=true
  VERCEL_API_TOKEN=your_token_here
  ```
  Or override per launch without editing `.env`:
  ```powershell
  uv run always-on run --watch-all
  # Or selectively:
  uv run always-on run --watch-claude-code --watch-vercel
  ```
- **2-Way Discord Bot (Mobile/Desktop Chat + Proactive Alerts, Zero Tunnels):**
  1. In the [Discord Developer Portal](https://discord.com/developers/applications), create an application/bot, enable **Message Content Intent** under *Bot*, and invite it to your private server (scopes: `bot`, `applications.commands`).
  2. Add these to `.env`:
     ```env
     ENABLE_DISCORD_BOT=true
     DISCORD_BOT_TOKEN=your_bot_token
     DISCORD_ALLOWED_USER_IDS=your_numeric_discord_user_id
     DISCORD_NOTIFY_CHANNEL_ID=your_numeric_channel_id
     # Optional:
     # DISCORD_GUILD_ID=your_server_id          # instant slash command sync (/status, /sleep, /memory, /correct)
     # DISCORD_PASSIVE_CHANNEL_IDS=1234567890   # salience-gated channel (priority=normal)
     # DISCORD_QUIET_HOURS=22:00-08:00          # suppresses non-critical proactive alerts overnight
     ```
  3. Run `uv run always-on run` (or pass `--discord`). Chat messages persist in SQLite (`chat_messages`), so Discord and the web dashboard share the same conversation history across restarts.
- **File Watcher:** Drop or edit any text/markdown file inside `data/inbox/` — the daemon picks it up on the next 1s tick and scores it via zero-token local salience.
- **Standalone Claude Code Release Watcher (Optional Separate Process):**
  Polls `anthropics/claude-code` GitHub Releases (with HTTP `ETag` `304` zero-quota caching + automatic `CHANGELOG.md` fallback) and POSTs new versions to `POST /api/events/webhook`:
  ```powershell
  # Test right now by sending the current latest Claude Code release to Aion
  uv run always-on watch-claude-code --once --force --process-now

  # Keep watching continuously in a second terminal (checks every 15 min)
  uv run always-on watch-claude-code

  # Use salience-gated mode instead of hard-override priority=high
  uv run always-on watch-claude-code --priority normal
  ```
- **Vercel Deployment Watcher & Native Push Webhook:**
  Supports **smart priority** by default (`ERROR` failures & `production` successes wake immediately; `preview` deploys use zero-token local salience gating) and pulls failed build log tails automatically when `VERCEL_API_TOKEN` is set.
  - **Option A — Zero-Tunnel API Poller (Free/Hobby + Pro):**
    Set `VERCEL_API_TOKEN` in `.env` (from https://vercel.com/account/tokens), then:
    ```powershell
    # Test right now by sending your single latest deployment
    uv run always-on watch-vercel --once --force --process-now

    # Keep watching every 60 seconds (Ctrl+C to stop)
    uv run always-on watch-vercel
    ```
  - **Option B — Native Vercel Push Webhook (Pro/Enterprise + Tunnel):**
    Set `VERCEL_WEBHOOK_SECRET` in `.env`, expose local port `8420` (e.g. via `cloudflared tunnel --url http://127.0.0.1:8420` or Tailscale Funnel), and point Vercel Webhooks (`deployment.created`, `deployment.succeeded`, `deployment.error`) to:
    `https://<your-tunnel>/api/events/webhook/vercel` (verified via HMAC-SHA1 `x-vercel-signature`).
- **Manual Webhook Endpoint:**
  ```bash
  curl -X POST http://127.0.0.1:8420/api/events/webhook \
    -H "Content-Type: application/json" \
    -d '{"content": "CI build failed on main branch", "actor_id": "github", "priority": "normal"}'
  ```
  *(Set `WEBHOOK_SECRET` in `.env` to require `X-Webhook-Secret` or `Authorization: Bearer` on webhook calls.)*
- **External MCP Servers (GitHub, Calendar, Gmail, etc.):**
  1. Copy `mcp_servers.example.json` to `mcp_servers.json` (git-ignored) and set `"enabled": true` on the servers you want.
  2. Use `${VAR}` or `${VAR:-default}` placeholders in `args`, `env`, `url`, or `headers` to pull secrets cleanly from `.env` (e.g. `${GITHUB_TOKEN}`).
  3. By default (`ALLOW_MCP_ON_BACKGROUND=false`), external MCP tools (`mcp__<server>__<tool>`) are exposed only on direct `chat` wake-ups unless a server explicitly sets `"allow_on_background": true`, and all remote outputs are wrapped in `<untrusted_mcp_content>` tags.
  4. Verify your configuration before starting the daemon:
     ```powershell
     uv run always-on mcp
     ```

---

## Running Tests

All unit and integration tests run completely offline using mocked Claude Haiku 5.5 responses and deterministic local embeddings:

```bash
uv run pytest -v
```

### Test Coverage
- `tests/test_salience.py` — Hard overrides, novelty suppression, goal relevance, exponential temporal pressure curve, heartbeat wake gating.
- `tests/test_fatigue.py` — Empty buffer guard, token saturation, semantic fragmentation, error/correction lesson load, idle sleep trigger, mid-conversation guard + `> 0.95` critical saturation override.
- `tests/test_consolidation.py` — Transactional `ADD`/`UPDATE`/`ARCHIVE`, soft-archiving (`UPDATE_SUPERSEDED` & `ARCHIVE`), strict provenance rejection, pinned protection, 25% churn circuit breaker abort + buffer preservation.
- `tests/test_circuit_breaker.py` — Crash-persistent daily USD cap, rolling hourly rate limit, per-wake step cap, SHA-256 duplicate tool blocking, semantic spinning abort, exponential backoff progression & persistence.
- `tests/test_llm_constraints.py` — Zero forbidden sampling params, adaptive thinking + effort, ≥ 512-token cached prefix, block-type parsing, assistant prefill rejection, `refusal` & `model_context_window_exceeded` handling.
- `tests/test_event_ingestor.py` — `actor_id="agent"` self-trigger prevention and 5-minute deduplication cooldown.
- `tests/test_web_and_sources.py` — `FileWatcherSource` + FastAPI chat, webhook, sleep, telemetry, and pin endpoints.
