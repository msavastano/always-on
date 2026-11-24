# Always-On CLS Agent (Aion) — Feature Roadmap

Tracks implemented capabilities and the upcoming feature roadmap discussed for Aion.

## Completed (Shipped)

- [x] **Core CLS Daemon** — `AlwaysOnDaemon`, local ONNX FastEmbed salience gating, 4-signal working fatigue with mid-conversation guard, transactional `sleep_and_consolidate` with provenance & churn circuit breaker, disk-backed `AgentCircuitBreaker`, SQLite WAL persistence, FastAPI dashboard & Typer CLI
- [x] **Web Search & Webpage Fetch** — `ddg`, `brave`, `tavily`, `searxng`, and `anthropic` native web search providers plus SSRF-safe `fetch_webpage` with SQLite TTL cache and `<untrusted_web_content>` framing
- [x] **Server-Side Code Execution & Markdown UI** — Anthropic native server-side `code_execution_20260120` sandbox tool paired with safe pure-DOM Markdown rendering in the dashboard chat window
- [x] **Managed Background Watchers** — Claude Code GitHub releases watcher (ETag polling) and Vercel Deployment watcher (API poller & native HMAC-SHA1 webhook with `smart` priority policy) managed automatically inside `uv run always-on run`
- [x] **Persistent Chat History Across Restarts** — SQLite `chat_messages` table shared across the web dashboard and Discord bridge
- [x] **2-Way Discord Bot Bridge** — `discord.py` outbound WebSocket gateway, user allowlist, DM / notify-channel / `@mention` replies, passive salience-gated channels, `/status`, `/memory`, `/sleep`, `/correct` slash commands, proactive alerts, and `DISCORD_QUIET_HOURS` with critical override

## 1. High-Impact Quick Wins

- [ ] **Additional Outbound Notifiers (Windows Toast / `ntfy.sh` / Telegram)** — Pluggable desktop and mobile push channels alongside Discord, honoring Quiet Hours
- [ ] **Dashboard Quality-of-Life Controls**
  - [ ] Active Priority Editor & Core Memory CRUD directly in the web UI (add/edit/pin/remove items to steer salience immediately without waiting for sleep)
  - [ ] Live Watchers Card + "Check Now" Buttons (display managed Claude Code & Vercel watcher status, last checked time, last tag/deploy, and one-click manual refresh)
  - [ ] Prompt Cache Hit-Rate Metric on the Spend card (cache read vs. fresh input %)
- [ ] **Discord Bridge Polish**
  - [ ] Automatic DM fallback in `send_proactive_alert` when `DISCORD_NOTIFY_CHANNEL_ID` is unset
  - [ ] Optional Discord forwarding for automatic sleep consolidation summaries and/or heartbeat wake-ups
  - [ ] Color-coded Discord embeds (red for build `ERROR`, green for prod `READY`, purple for Claude Code releases) leveraging colors already returned by `classify_proactive_alert`

## 2. Deeper CLS Memory & Autonomy

- [ ] **Long-Term Episodic Archive + Semantic Recall Tool (`search_past_episodes`)** — Archive flushed `fast_buffer` episodes (with existing local embeddings) in a read-only `episode_archive` table for zero-token vector recall without bloating the cached system prefix
- [ ] **Persistent Reminders & Self-Scheduled Follow-Ups (`schedule_reminder` tool)** — SQLite `scheduled_events` table polled by the 1s outer loop for zero idle tokens, supporting user reminders and self-scheduled agent follow-ups
- [ ] **Morning Briefing / Daily Digest** — Configurable daily scheduled wake-up (e.g., 8:30 AM) compiling overnight Vercel deploys, Claude Code releases, inbox files, and Active Priorities into one concise Markdown brief (with optional Discord push)

## 3. Ecosystem & Safe Real-World Actions

- [x] **MCP Client Support** — Add an MCP client to `src/always_on/tools/registry.py` to connect to external MCP servers (Calendar, Gmail, GitHub, Linear, Obsidian, etc.) via config (`mcp_servers.json`)
- [ ] **More Zero-Token Semantic Radar Watchers**
  - [ ] GitHub Repo Watcher (PR reviews, failed Actions runs, new issues on personal repos using the zero-quota ETag pattern)
  - [ ] RSS / Hacker News / arXiv Feed Watcher (`priority="normal"` so `compute_salience` silently drops ~95% of items at `$0` and wakes only on Active Priority matches)
- [ ] **Human-in-the-Loop Approval Queue for Write Actions** — Approve / Deny UI card in the dashboard (and optionally Discord) before mutating external systems (Vercel redeploy/rollback, GitHub issues, sending emails)
