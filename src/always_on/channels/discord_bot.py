"""2-way Discord bot bridge running inside AlwaysOnDaemon over outbound WebSocket."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
from datetime import datetime, time as dt_time
from typing import TYPE_CHECKING, Any, Literal

from always_on.config import Settings
from always_on.models import ChatReply, IncomingEvent, utc_now_iso

if TYPE_CHECKING:
    from always_on.daemon import AlwaysOnDaemon

logger = logging.getLogger("always_on.discord")

_QUIET_HOURS_RE = re.compile(r"^\s*(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})\s*$")
_FENCE_OPEN_RE = re.compile(r"^```([A-Za-z0-9_+.-]*)\s*$")


def parse_id_set(raw_csv: str) -> set[int]:
    """Parse comma-separated positive Discord snowflake IDs safely."""
    ids: set[int] = set()
    for part in (raw_csv or "").split(","):
        cleaned = part.strip()
        if cleaned.isdigit():
            val = int(cleaned)
            if val > 0:
                ids.add(val)
    return ids


def parse_optional_int(raw: str) -> int | None:
    """Parse an optional single numeric Discord snowflake ID."""
    cleaned = (raw or "").strip()
    if cleaned.isdigit():
        val = int(cleaned)
        return val if val > 0 else None
    return None


def is_in_quiet_hours(spec: str, *, now_local: datetime | None = None) -> bool:
    """Return True if local wall-clock time falls inside `HH:MM-HH:MM` (24-hour)."""
    match = _QUIET_HOURS_RE.match(spec or "")
    if not match:
        return False
    sh, sm, eh, em = (int(g) for g in match.groups())
    if not (0 <= sh <= 23 and 0 <= sm <= 59 and 0 <= eh <= 23 and 0 <= em <= 59):
        return False

    start = dt_time(sh, sm)
    end = dt_time(eh, em)
    if start == end:
        return False

    current = (now_local or datetime.now()).time()
    if start < end:
        return start <= current < end
    # Overnight window (e.g. 22:00-08:00)
    return current >= start or current < end


def strip_bot_mention(
    content: str,
    bot_user_id: int | None,
    bot_role_ids: set[int] | None = None,
) -> str:
    """Remove `<@bot_id>`, `<@!bot_id>`, and managed bot role `<@&role_id>` tokens cleanly."""
    text = (content or "").strip()
    if bot_user_id is not None:
        text = re.sub(rf"<@!?{bot_user_id}>", "", text)
    for role_id in bot_role_ids or ():
        text = re.sub(rf"<@&{role_id}>", "", text)
    return text.strip()


def detect_bot_mention(message: Any, bot_user_id: int | None) -> tuple[bool, set[int]]:
    """Return `(mentioned, matched_bot_role_ids)` handling user, raw, and managed bot role pings."""
    if bot_user_id is None:
        return False, set()

    user_mentioned = (
        any(
            getattr(u, "id", None) == bot_user_id
            for u in (getattr(message, "mentions", None) or ())
        )
        or bot_user_id in (getattr(message, "raw_mentions", None) or ())
        or bool(re.search(rf"<@!?{bot_user_id}>", str(getattr(message, "content", "") or "")))
    )

    bot_role_ids: set[int] = set()
    for role in getattr(message, "role_mentions", None) or ():
        tags = getattr(role, "tags", None)
        if getattr(tags, "bot_id", None) == bot_user_id:
            role_id = getattr(role, "id", None)
            if role_id is not None:
                bot_role_ids.add(role_id)

    return bool(user_mentioned or bot_role_ids), bot_role_ids


def should_accept_discord_message(
    *,
    author_id: int,
    author_is_bot: bool,
    bot_user_id: int | None,
    is_dm: bool,
    channel_id: int,
    allowed_user_ids: set[int],
    passive_channel_ids: set[int],
    mentioned_bot: bool,
    notify_channel_id: int | None,
) -> Literal["direct", "passive", "ignore"]:
    """Gate incoming Discord messages with fail-closed allowlist and self-trigger protection."""
    if author_is_bot:
        return "ignore"
    if bot_user_id is not None and author_id == bot_user_id:
        return "ignore"
    if not allowed_user_ids or author_id not in allowed_user_ids:
        return "ignore"

    if channel_id in passive_channel_ids and not mentioned_bot:
        return "passive"

    if is_dm or mentioned_bot or (notify_channel_id is not None and channel_id == notify_channel_id):
        return "direct"

    return "ignore"


def chunk_discord_message(text: str, *, limit: int = 1950) -> list[str]:
    """Split Markdown text into chunks <= `limit` chars without breaking fenced code blocks."""
    cleaned = (text or "").strip()
    if not cleaned:
        return ["(empty response)"]
    if len(cleaned) <= limit:
        return [cleaned]

    lines = cleaned.splitlines(keepends=True)
    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0
    open_fence_lang: str | None = None

    def _flush() -> None:
        nonlocal buf, buf_len
        if not buf:
            return
        chunk_text = "".join(buf).rstrip()
        if open_fence_lang is not None:
            chunk_text = chunk_text + "\n```"
        if chunk_text:
            chunks.append(chunk_text)
        buf = []
        buf_len = 0
        if open_fence_lang is not None:
            reopen = f"```{open_fence_lang}\n"
            buf.append(reopen)
            buf_len = len(reopen)

    for line in lines:
        stripped = line.strip()
        fence_reserve = 5 if open_fence_lang is not None else 0

        # If a single line exceeds chunk capacity, flush current buffer then hard-slice line safely
        max_line_cap = max(64, limit - fence_reserve - (len(f"```{open_fence_lang or ''}\n") if open_fence_lang is not None else 0))
        if len(line) > max_line_cap:
            if buf and "".join(buf).strip() != f"```{open_fence_lang or ''}":
                _flush()
            pos = 0
            while pos < len(line):
                reopen_len = len(buf[0]) if buf else 0
                avail = max(32, limit - fence_reserve - reopen_len)
                part = line[pos : pos + avail]
                buf.append(part)
                buf_len += len(part)
                pos += len(part)
                if pos < len(line):
                    _flush()
        else:
            if buf and (buf_len + len(line) + fence_reserve) > limit:
                _flush()
            buf.append(line)
            buf_len += len(line)

        # Update fenced code state AFTER handling the line
        if open_fence_lang is None:
            m = _FENCE_OPEN_RE.match(stripped)
            if m:
                open_fence_lang = m.group(1)
        elif stripped == "```":
            open_fence_lang = None

    if buf:
        final_text = "".join(buf).rstrip()
        # Avoid trailing empty fence if last chunk ended right after reopening
        if open_fence_lang is not None and final_text == f"```{open_fence_lang}":
            final_text = ""
        if final_text:
            chunks.append(final_text)

    return chunks or ["(empty response)"]


def classify_proactive_alert(
    event: IncomingEvent, reply: ChatReply
) -> tuple[str, int, bool]:
    """Return (title, hex_color_int, is_critical) for a proactive Discord alert."""
    payload = event.payload or {}
    if payload.get("provider") == "vercel":
        state = str(payload.get("state") or "").upper()
        project = str(payload.get("project_name") or "Vercel")
        target = str(payload.get("target") or "")
        tag_str = f"{project} [{target}]" if target else project
        if state == "ERROR":
            return f"Vercel Build Failed: {tag_str}", 0xE53935, True
        if state == "READY":
            return f"Vercel Deployment Ready: {tag_str}", 0x2E7D32, False
        return f"Vercel Deployment {state}: {tag_str}".strip(), 0x1F5FD6, False

    if event.actor_id == "claude_code_watcher":
        tag = str(payload.get("tag") or "").strip()
        title = f"Claude Code Release {tag}".strip()
        return title, 0x7C4DFF, False

    is_critical = (
        event.priority == "high" and reply.status in ("error", "circuit_tripped")
    ) or (event.priority == "high" and "error" in event.content.lower())
    return f"Proactive Alert ({event.source})", 0x1F5FD6, is_critical


class DiscordBridge:
    """Manages an outbound discord.py Gateway client wired into AlwaysOnDaemon."""

    def __init__(self, daemon: "AlwaysOnDaemon", settings: Settings) -> None:
        self.daemon = daemon
        self.settings = settings
        self.allowed_user_ids = parse_id_set(settings.discord_allowed_user_ids)
        self.notify_channel_id = parse_optional_int(settings.discord_notify_channel_id)
        self.passive_channel_ids = parse_id_set(settings.discord_passive_channel_ids)
        self.guild_id = parse_optional_int(settings.discord_guild_id)

        self._client: Any | None = None
        self._task: asyncio.Task[None] | None = None
        self.status: dict[str, Any] = {
            "enabled": True,
            "connected": False,
            "bot_user": None,
            "last_error": None,
            "notify_channel_id": self.notify_channel_id,
            "passive_channel_ids": sorted(self.passive_channel_ids),
            "allowed_user_count": len(self.allowed_user_ids),
            "quiet_hours": settings.discord_quiet_hours or None,
        }

    async def start(self) -> None:
        """Validate config, register handlers & slash commands, and launch Gateway task."""
        token = (self.settings.discord_bot_token or "").strip()
        if not token:
            msg = "DISCORD_BOT_TOKEN is empty; Discord bridge disabled."
            logger.warning(msg)
            self.status["last_error"] = msg
            self.daemon._record_activity(
                kind="discord",
                source="discord",
                summary=msg,
                status="error",
            )
            return

        if not self.allowed_user_ids:
            logger.warning(
                "DISCORD_ALLOWED_USER_IDS is empty; inbound Discord messages will be ignored (fail-closed)."
            )

        import discord
        from discord import app_commands

        intents = discord.Intents.default()
        intents.message_content = True
        intents.dm_messages = True
        intents.guild_messages = True

        client = discord.Client(intents=intents)
        tree = app_commands.CommandTree(client)
        self._client = client

        async def _require_allowed(interaction: Any) -> bool:
            if not self.allowed_user_ids or interaction.user.id not in self.allowed_user_ids:
                await interaction.response.send_message("Unauthorized.", ephemeral=True)
                return False
            return True

        @tree.command(name="status", description="Show Aion salience, fatigue, spend, and watcher status")
        async def cmd_status(interaction: Any) -> None:
            if not await _require_allowed(interaction):
                return
            await interaction.response.defer(ephemeral=True)
            data = await self.daemon.get_telemetry_snapshot()
            fat = data.get("fatigue", {})
            fb = fat.get("breakdown", {})
            sal = data.get("salience") or {}
            cb = data.get("circuit_breaker", {})
            spend = cb.get("daily_spend", {})
            watchers = data.get("watchers") or {}
            watcher_txt = (
                ", ".join(f"{k}={v.get('last_status', '?')}" for k, v in sorted(watchers.items()))
                or "none"
            )
            mcp_info = data.get("mcp") or {}
            mcp_line = ""
            if mcp_info.get("server_count", 0) > 0:
                mcp_line = (
                    f"\n- **MCP Servers:** `{mcp_info.get('connected_count', 0)}/"
                    f"{mcp_info.get('server_count', 0)}` connected · `{mcp_info.get('tool_count', 0)}` tool(s)"
                )
            msg = (
                f"**{data.get('agent_name', 'Agent')} Telemetry** (`{data.get('model')}`)\n"
                f"- **Last Salience:** `{float(sal.get('score', 0.0)):.3f}` (wake > `{data.get('wake_threshold')}`)\n"
                f"- **Working Fatigue:** `{float(fat.get('score', 0.0)):.3f}` / `{data.get('sleep_threshold')}` "
                f"({fb.get('buffer_entries', 0)} eps, {fb.get('buffer_tokens', 0)} tok)\n"
                f"- **Daily Spend (UTC):** `${float(spend.get('usd_spent', 0.0)):.5f}` / `${float(cb.get('max_daily_usd', 1.0)):.2f}`\n"
                f"- **Web Searches:** `{spend.get('web_searches', 0)}` / `{cb.get('max_web_searches_per_day', 50)}` ({data.get('web_search_provider')})\n"
                f"- **Hourly Wakes:** `{cb.get('wakes_last_hour', 0)}` / `{cb.get('max_wakes_per_hour', 30)}`\n"
                f"- **Memory:** `{len(data.get('core_memory', []))}` core · `{len(data.get('memory_archive', []))}` archive · `{len(data.get('fast_buffer', []))}` buffer\n"
                f"- **Watchers:** {watcher_txt}"
                f"{mcp_line}"
            )
            await interaction.followup.send(msg, ephemeral=True)

        @tree.command(name="sleep", description="Trigger CLS sleep consolidation now")
        async def cmd_sleep(interaction: Any) -> None:
            if not await _require_allowed(interaction):
                return
            await interaction.response.defer()
            report = await self.daemon.trigger_sleep(reason="manual")
            summary = (
                f"**Sleep Consolidation ({report.status})**\n"
                f"- Episodes processed: `{report.episodes_processed}`\n"
                f"- Diffs: `+{len(report.added_ids)}` added · `~{len(report.updated_ids)}` updated · "
                f"`-{len(report.archived_ids)}` archived · `{len(report.rejected_operations)}` rejected\n"
                f"- Cost: `${report.usd_cost:.5f}`\n"
                f"- Note: {report.reflection or report.abort_reason or 'Done.'}"
            )
            chunks = chunk_discord_message(summary)
            await interaction.followup.send(chunks[0])
            for extra in chunks[1:]:
                await interaction.followup.send(extra)

        @tree.command(name="memory", description="Inspect Active Priorities and Core Memory")
        async def cmd_memory(interaction: Any) -> None:
            if not await _require_allowed(interaction):
                return
            await interaction.response.defer(ephemeral=True)
            prios = await self.daemon.store.list_active_priorities()
            core = await self.daemon.store.list_core_memory()
            lines = ["**Active Priorities**"]
            if prios:
                lines.extend(f"{p.rank}. {p.description}" for p in prios)
            else:
                lines.append("_(none)_")
            lines.append("")
            lines.append(f"**Core Memory ({len(core)})**")
            if core:
                for m in core:
                    pin = " [PIN]" if m.pinned else ""
                    lines.append(f"- `{m.id}` ({m.category}{pin}): {m.content}")
            else:
                lines.append("_(empty)_")
            chunks = chunk_discord_message("\n".join(lines))
            await interaction.followup.send(chunks[0], ephemeral=True)
            for extra in chunks[1:]:
                await interaction.followup.send(extra, ephemeral=True)

        @tree.command(name="correct", description="Send a message flagged as a user correction")
        @app_commands.describe(message="Your correction or preference update")
        async def cmd_correct(interaction: Any, message: str) -> None:
            if not await _require_allowed(interaction):
                return
            await interaction.response.defer()
            reply = await self.daemon.send_chat(
                message,
                is_correction=True,
                actor_id=f"discord:{interaction.user.name}",
                source="discord",
            )
            chunks = chunk_discord_message(reply.reply)
            await interaction.followup.send(chunks[0])
            for extra in chunks[1:]:
                await interaction.followup.send(extra)

        @client.event
        async def on_ready() -> None:
            self.status["connected"] = True
            self.status["bot_user"] = str(client.user)
            self.status["last_error"] = None
            try:
                if self.guild_id is not None:
                    guild_obj = discord.Object(id=self.guild_id)
                    tree.copy_global_to(guild=guild_obj)
                    await tree.sync(guild=guild_obj)
                else:
                    await tree.sync()
            except Exception as exc:
                logger.warning("Failed syncing Discord slash commands: %s", exc)

            self.daemon._record_activity(
                kind="discord",
                source="discord",
                summary=f"Discord bot connected as {client.user}.",
            )
            await self.daemon._broadcast_telemetry()

        async def _record_reply_error(channel_id: int, exc: Exception) -> None:
            logger.warning("Failed handling Discord message in channel %s: %s", channel_id, exc)
            self.daemon._record_activity(
                kind="discord",
                source="discord",
                summary=f"Discord reply error in channel {channel_id}: {exc}",
                status="error",
            )
            with contextlib.suppress(Exception):
                await self.daemon._broadcast_telemetry()

        @client.event
        async def on_message(message: Any) -> None:
            bot_uid = client.user.id if client.user is not None else None
            is_dm = isinstance(message.channel, discord.DMChannel)
            mentioned, bot_role_ids = detect_bot_mention(message, bot_uid)
            mode = should_accept_discord_message(
                author_id=message.author.id,
                author_is_bot=bool(message.author.bot),
                bot_user_id=bot_uid,
                is_dm=is_dm,
                channel_id=message.channel.id,
                allowed_user_ids=self.allowed_user_ids,
                passive_channel_ids=self.passive_channel_ids,
                mentioned_bot=mentioned,
                notify_channel_id=self.notify_channel_id,
            )
            if mode == "ignore":
                return

            cleaned = strip_bot_mention(message.content, bot_uid, bot_role_ids)
            if not cleaned:
                if mode == "direct" and mentioned:
                    try:
                        await message.reply(
                            "I saw your mention! Add a message after mentioning me (e.g., `@always-on check this`) and I'll reply right here.",
                            mention_author=False,
                        )
                    except Exception as exc:
                        await _record_reply_error(message.channel.id, exc)
                return

            actor_tag = f"discord:{message.author.name}"
            if mode == "passive":
                await self.daemon.push_webhook(
                    content=cleaned,
                    actor_id=actor_tag,
                    payload={
                        "provider": "discord",
                        "channel_id": message.channel.id,
                        "message_id": message.id,
                        "discord_passive": True,
                    },
                    priority="normal",
                    hard_override=False,
                )
                return

            try:
                async with message.channel.typing():
                    reply = await self.daemon.send_chat(
                        cleaned,
                        actor_id=actor_tag,
                        source="discord",
                    )
                chunks = chunk_discord_message(reply.reply)
                await message.reply(chunks[0], mention_author=False)
                for extra in chunks[1:]:
                    await message.channel.send(extra)
            except Exception as exc:
                await _record_reply_error(message.channel.id, exc)
                with contextlib.suppress(Exception):
                    await message.channel.send(f"Error processing request: {exc}")

        self.daemon.register_proactive_listener(self.send_proactive_alert)
        self._task = asyncio.create_task(
            self._run_gateway(token),
            name="discord-bot-gateway",
        )

    async def _run_gateway(self, token: str) -> None:
        assert self._client is not None
        try:
            await self._client.start(token)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Discord gateway stopped with error: %s", exc)
            self.status["connected"] = False
            self.status["last_error"] = str(exc)
            self.daemon._record_activity(
                kind="discord",
                source="discord",
                summary=f"Discord gateway error: {exc}",
                status="error",
            )
            with contextlib.suppress(Exception):
                await self.daemon._broadcast_telemetry()

    async def send_proactive_alert(
        self, event: IncomingEvent, reply: ChatReply
    ) -> None:
        """Push proactive background wake-up summaries (Vercel/Claude Code/Inbox) to Discord."""
        if self._client is None or not self._client.is_ready():
            return
        if self.notify_channel_id is None:
            return

        title, _color, is_critical = classify_proactive_alert(event, reply)
        if is_in_quiet_hours(self.settings.discord_quiet_hours) and not is_critical:
            logger.info("Suppressed non-critical Discord alert during quiet hours: %s", title)
            return

        channel = self._client.get_channel(self.notify_channel_id)
        if channel is None:
            channel = await self._client.fetch_channel(self.notify_channel_id)

        header = (
            f"**{title}** _(source: `{event.source}` · `{event.actor_id}` · "
            f"`${reply.usd_cost:.5f}`)_"
        )
        full_text = f"{header}\n\n{reply.reply}".strip()
        for chunk in chunk_discord_message(full_text):
            await channel.send(chunk)

    async def stop(self) -> None:
        """Gracefully close Discord WebSocket gateway and background task."""
        self.status["connected"] = False
        if self._client is not None:
            with contextlib.suppress(Exception, asyncio.TimeoutError):
                await asyncio.wait_for(self._client.close(), timeout=2.5)
            self._client = None
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError, asyncio.TimeoutError):
                await asyncio.wait_for(self._task, timeout=2.0)
            self._task = None
        self.status["bot_user"] = None
