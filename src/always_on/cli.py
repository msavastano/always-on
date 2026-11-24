"""Interactive CLI for running, chatting with, and inspecting the Always-On CLS Agent."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
from pathlib import Path
from types import FrameType
from typing import Any, Literal

import httpx
import typer
import uvicorn
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from always_on.config import Settings, VercelPriorityPolicy
from always_on.daemon import AlwaysOnDaemon
from always_on.tools.mcp_client import MCPClientManager
from always_on.watchers.claude_code import DEFAULT_STATE_PATH, check_and_notify_once
from always_on.watchers.vercel import DEFAULT_VERCEL_STATE_PATH, poll_vercel_and_notify_once
from always_on.web.app import create_app


class _GracefulUvicornServer(uvicorn.Server):
    """Uvicorn server that immediately wakes daemon & open SSE streams on Ctrl+C / SIGINT."""

    def __init__(self, config: uvicorn.Config, daemon: AlwaysOnDaemon) -> None:
        super().__init__(config)
        self._daemon = daemon

    def handle_exit(self, sig: int, frame: FrameType | None) -> None:
        self._daemon.request_shutdown()
        super().handle_exit(sig, frame)

app = typer.Typer(
    name="always-on",
    help="Always-On Continuous AI Agent Daemon (CLS Memory & Sleep Consolidation).",
    no_args_is_help=True,
)
console = Console()


@app.command()
def run(
    host: str | None = typer.Option(None, help="Bind host for web dashboard/API"),
    port: int | None = typer.Option(None, help="Bind port for web dashboard/API"),
    no_web: bool = typer.Option(False, help="Run headless daemon loop without web server"),
    watch_claude_code: bool | None = typer.Option(
        None,
        "--watch-claude-code/--no-watch-claude-code",
        help="Enable/disable embedded Claude Code release watcher (defaults to ENABLE_CLAUDE_CODE_WATCHER in .env)",
    ),
    watch_vercel: bool | None = typer.Option(
        None,
        "--watch-vercel/--no-watch-vercel",
        help="Enable/disable embedded Vercel deployment watcher (defaults to ENABLE_VERCEL_WATCHER in .env)",
    ),
    watch_all: bool = typer.Option(
        False,
        "--watch-all",
        help="Enable both Claude Code and Vercel background watchers",
    ),
    discord: bool | None = typer.Option(
        None,
        "--discord/--no-discord",
        help="Enable/disable 2-way Discord bot (defaults to ENABLE_DISCORD_BOT in .env)",
    ),
) -> None:
    """Start the continuous Always-On agent daemon (with local web dashboard by default)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    settings = Settings()
    if watch_all:
        settings.enable_claude_code_watcher = True
        settings.enable_vercel_watcher = True
    if watch_claude_code is not None:
        settings.enable_claude_code_watcher = watch_claude_code
    if watch_vercel is not None:
        settings.enable_vercel_watcher = watch_vercel
    if discord is not None:
        settings.enable_discord_bot = discord

    active_watchers: list[str] = []
    if settings.enable_claude_code_watcher:
        active_watchers.append(
            f"claude-code ({settings.claude_code_watcher_interval_seconds}s)"
        )
    if settings.enable_vercel_watcher:
        active_watchers.append(f"vercel ({settings.vercel_watcher_interval_seconds}s)")
    watchers_label = ", ".join(active_watchers) if active_watchers else "none"
    discord_label = "enabled" if settings.enable_discord_bot else "disabled"

    bind_host = host or settings.host
    bind_port = port or settings.port

    if no_web:
        console.print(
            Panel.fit(
                f"[bold green]{settings.agent_name}[/] daemon starting in headless mode\n"
                f"Model: [cyan]{settings.anthropic_model}[/] | DB: [dim]{settings.db_path}[/]\n"
                f"Watchers: [cyan]{watchers_label}[/] | Discord: [cyan]{discord_label}[/]",
                title="Always-On CLS Daemon",
            )
        )

        async def _run_headless() -> None:
            daemon = AlwaysOnDaemon(settings)
            await daemon.start(run_background_loop=True)

            def _on_signal(_sig: int, _frame: FrameType | None) -> None:
                daemon.request_shutdown()

            prev_sigint = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, _on_signal)
            if hasattr(signal, "SIGBREAK"):
                signal.signal(signal.SIGBREAK, _on_signal)
            try:
                await daemon._stop_event.wait()
            except (KeyboardInterrupt, asyncio.CancelledError):
                daemon.request_shutdown()
            finally:
                signal.signal(signal.SIGINT, prev_sigint)
                await daemon.stop()

        asyncio.run(_run_headless())
        return

    console.print(
        Panel.fit(
            f"[bold green]{settings.agent_name}[/] daemon starting with local dashboard\n"
            f"Dashboard: [bold cyan]http://{bind_host}:{bind_port}[/]\n"
            f"Webhook:   [dim]POST http://{bind_host}:{bind_port}/api/events/webhook[/]\n"
            f"Inbox dir: [dim]{settings.inbox_dir}[/] | Model: [cyan]{settings.anthropic_model}[/] | "
            f"Search: [cyan]{settings.web_search_provider}[/]\n"
            f"Watchers:  [cyan]{watchers_label}[/] | Discord: [cyan]{discord_label}[/]",
            title="Always-On CLS Agent",
        )
    )
    daemon = AlwaysOnDaemon(settings)
    web_app = create_app(settings=settings, daemon=daemon)
    config = uvicorn.Config(
        web_app,
        host=bind_host,
        port=bind_port,
        log_level="info",
        timeout_graceful_shutdown=2,
    )
    server = _GracefulUvicornServer(config, daemon=daemon)
    server.run()


@app.command()
def status(
    url: str = typer.Option("http://127.0.0.1:8420", help="Running daemon base URL"),
) -> None:
    """Inspect live salience, fatigue, daily spend, circuit breaker, and memory state."""
    data = _fetch_json(f"{url.rstrip('/')}/api/telemetry")
    _render_status(data)


@app.command()
def sleep(
    url: str = typer.Option("http://127.0.0.1:8420", help="Running daemon base URL"),
) -> None:
    """Manually trigger an offline sleep_and_consolidate pass on the running daemon."""
    with console.status("[bold magenta]Running transactional sleep consolidation..."):
        report = _post_json(f"{url.rstrip('/')}/api/sleep", {})
    _render_sleep_report(report)


@app.command("mcp")
def mcp_check() -> None:
    """Connect to configured external MCP servers, list discovered tools, and print diagnostics."""
    settings = Settings()

    async def _probe() -> dict[str, Any]:
        mgr = MCPClientManager(settings)
        try:
            return await mgr.start()
        finally:
            await mgr.stop()

    with console.status("[bold cyan]Probing configured MCP servers..."):
        snap = asyncio.run(_probe())

    if not snap.get("enabled"):
        console.print("[yellow]MCP client support is disabled (`ENABLE_MCP=false`).[/]")
        return

    if snap.get("config_error"):
        console.print(f"[bold red]MCP config error:[/] {snap['config_error']}")
        raise typer.Exit(code=1)

    servers = snap.get("servers") or {}
    if not servers:
        console.print(
            Panel.fit(
                f"No external MCP servers configured yet.\n"
                f"Expected file: [cyan]{snap.get('config_path')}[/]\n"
                "Copy [bold]mcp_servers.example.json[/] -> [bold]mcp_servers.json[/] "
                "to connect GitHub, Calendar, Gmail, etc.",
                title="MCP Servers (0 configured)",
            )
        )
        return

    table = Table(
        title=(
            f"MCP Servers ({snap.get('connected_count', 0)}/"
            f"{snap.get('server_count', 0)} connected · {snap.get('tool_count', 0)} tools)"
        )
    )
    table.add_column("Server", style="bold cyan")
    table.add_column("Transport")
    table.add_column("Status")
    table.add_column("Background")
    table.add_column("Tools / Error")

    for name, info in sorted(servers.items()):
        st = str(info.get("status", "unknown"))
        style = "green" if st == "connected" else ("dim" if st == "disabled" else "bold red")
        tools_list = info.get("tools") or []
        detail = ", ".join(tools_list) if tools_list else "(none)"
        if info.get("last_error"):
            detail = f"[red]{info['last_error']}[/]"
        table.add_row(
            name,
            str(info.get("transport", "stdio")),
            f"[{style}]{st}[/]",
            "allowed" if info.get("allow_on_background") else "chat-only",
            detail,
        )
    console.print(table)


@app.command()
def chat(
    url: str = typer.Option("http://127.0.0.1:8420", help="Running daemon base URL"),
) -> None:
    """Interactive terminal chat client connected to the running daemon."""
    base = url.rstrip("/")
    console.print(
        Panel.fit(
            f"Connected to [cyan]{base}[/]\n"
            "Commands: [bold]/status[/], [bold]/memory[/], [bold]/archive[/], "
            "[bold]/sleep[/], [bold]/correct <msg>[/], [bold]/webhook <msg>[/], [bold]/quit[/]",
            title="Always-On Interactive CLI",
        )
    )
    while True:
        try:
            raw = console.input("\n[bold cyan]you>[/] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\nBye.")
            break
        if not raw:
            continue
        if raw in ("/quit", "/exit"):
            break
        if raw == "/status":
            _render_status(_fetch_json(f"{base}/api/telemetry"))
            continue
        if raw == "/memory":
            mem = _fetch_json(f"{base}/api/memory")
            _render_core_memory(mem.get("core_memory", []))
            continue
        if raw == "/archive":
            mem = _fetch_json(f"{base}/api/memory")
            _render_archive(mem.get("memory_archive", []))
            continue
        if raw == "/sleep":
            with console.status("[bold magenta]Consolidating working memory..."):
                report = _post_json(f"{base}/api/sleep", {})
            _render_sleep_report(report)
            continue
        if raw.startswith("/webhook "):
            payload_text = raw[len("/webhook ") :].strip()
            res = _post_json(
                f"{base}/api/events/webhook?process_now=true",
                {"content": payload_text, "actor_id": "cli_webhook"},
            )
            console.print(f"[dim]Webhook enqueued: {res}[/]")
            continue

        is_correction = False
        message = raw
        if raw.startswith("/correct "):
            is_correction = True
            message = raw[len("/correct ") :].strip()

        with console.status("[bold green]Thinking..."):
            reply = _post_json(
                f"{base}/api/chat",
                {"message": message, "is_correction": is_correction},
            )
        status_tag = reply.get("status", "ok")
        cost = float(reply.get("usd_cost", 0.0))
        console.print(
            Panel(
                reply.get("reply", ""),
                title=f"[bold green]agent[/] [dim](status={status_tag}, ${cost:.5f})[/]",
            )
        )


@app.command("watch-claude-code")
def watch_claude_code(
    url: str = typer.Option("http://127.0.0.1:8420", help="Running daemon base URL"),
    once: bool = typer.Option(
        False,
        help="Check once and exit (ideal for Windows Task Scheduler or cron)",
    ),
    interval: int = typer.Option(
        900,
        min=30,
        help="Polling interval in seconds when running continuously (default: 900s / 15m)",
    ),
    force: bool = typer.Option(
        False,
        help="Force-send the latest release immediately even if already seen (for testing)",
    ),
    priority: Literal["normal", "high"] = typer.Option(
        "high",
        help="Webhook priority: 'high' wakes immediately (1.0), 'normal' uses local salience gating",
    ),
    process_now: bool = typer.Option(
        False,
        help="Wait for the daemon to process the webhook immediately instead of queuing for next tick",
    ),
    state_file: Path = typer.Option(
        DEFAULT_STATE_PATH,
        help="Path to persist last seen tag & GitHub ETag",
    ),
    webhook_secret: str | None = typer.Option(
        None,
        help="Optional webhook secret (defaults to WEBHOOK_SECRET from .env)",
    ),
    github_token: str | None = typer.Option(
        None,
        help="Optional GitHub token (defaults to GITHUB_TOKEN env var for 5,000 req/hr)",
    ),
) -> None:
    """Watch anthropics/claude-code releases and push new versions to the agent webhook."""
    settings = Settings()
    resolved_secret = webhook_secret if webhook_secret is not None else settings.webhook_secret
    resolved_token = github_token if github_token is not None else os.environ.get("GITHUB_TOKEN")

    def _print_result(res: Any) -> None:
        style = {
            "notified": "bold green",
            "up_to_date": "dim",
            "not_modified": "dim",
            "error": "bold red",
        }.get(res.status, "")
        event_part = f" [dim](event_id={res.event_id})[/]" if res.event_id else ""
        console.print(f"[{style}]\\[{res.status}][/] {res.message}{event_part}")

    if once:
        res = asyncio.run(
            check_and_notify_once(
                daemon_url=url,
                state_path=state_file,
                priority=priority,
                force=force,
                process_now=process_now,
                webhook_secret=resolved_secret,
                github_token=resolved_token,
            )
        )
        _print_result(res)
        if res.status == "error":
            raise typer.Exit(code=1)
        return

    console.print(
        Panel.fit(
            f"Watching [bold cyan]anthropics/claude-code[/] releases every [bold]{interval}s[/]\n"
            f"Target webhook: [cyan]{url.rstrip('/')}/api/events/webhook[/] (priority={priority})\n"
            f"State file: [dim]{state_file}[/] | Press [bold]Ctrl+C[/] to stop.",
            title="Claude Code Release Watcher",
        )
    )

    async def _run_loop() -> None:
        stop_event = asyncio.Event()
        loop = asyncio.get_running_loop()

        def _on_signal(_sig: int, _frame: FrameType | None) -> None:
            loop.call_soon_threadsafe(stop_event.set)

        prev_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, _on_signal)
        if hasattr(signal, "SIGBREAK"):
            signal.signal(signal.SIGBREAK, _on_signal)

        first_iteration = True
        try:
            while not stop_event.is_set():
                res = await check_and_notify_once(
                    daemon_url=url,
                    state_path=state_file,
                    priority=priority,
                    force=force and first_iteration,
                    process_now=process_now,
                    webhook_secret=resolved_secret,
                    github_token=resolved_token,
                )
                first_iteration = False
                _print_result(res)
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=interval)
                except asyncio.TimeoutError:
                    continue
        finally:
            signal.signal(signal.SIGINT, prev_sigint)

    try:
        asyncio.run(_run_loop())
    except KeyboardInterrupt:
        pass
    console.print("[dim]Claude Code release watcher stopped.[/]")


@app.command("watch-vercel")
def watch_vercel(
    url: str = typer.Option("http://127.0.0.1:8420", help="Running daemon base URL"),
    once: bool = typer.Option(
        False,
        help="Poll Vercel once and exit (ideal for Windows Task Scheduler or testing)",
    ),
    interval: int = typer.Option(
        60,
        min=15,
        help="Polling interval in seconds when running continuously (default: 60s)",
    ),
    force: bool = typer.Option(
        False,
        help="Force-deliver the latest matching deployment right now (for testing)",
    ),
    project: str | None = typer.Option(
        None,
        help="Optional comma-separated Vercel project names/IDs (defaults to VERCEL_PROJECT_FILTER)",
    ),
    team_id: str | None = typer.Option(
        None,
        help="Optional Vercel Team ID (defaults to VERCEL_TEAM_ID)",
    ),
    target: str = typer.Option(
        "",
        help="Optional environment filter: 'production' or 'preview' (empty = both)",
    ),
    policy: VercelPriorityPolicy | None = typer.Option(
        None,
        help="Salience policy: smart (default), failures_only, or all_high",
    ),
    process_now: bool = typer.Option(
        False,
        help="Wait for the daemon to process the webhook immediately",
    ),
    state_file: Path = typer.Option(
        DEFAULT_VERCEL_STATE_PATH,
        help="Path to persist seen Vercel deployment IDs & states",
    ),
    token: str | None = typer.Option(
        None,
        help="Vercel Personal Access Token (defaults to VERCEL_API_TOKEN from .env)",
    ),
    webhook_secret: str | None = typer.Option(
        None,
        help="Optional webhook secret (defaults to WEBHOOK_SECRET from .env)",
    ),
) -> None:
    """Zero-tunnel Vercel deployment poller (works on Free/Hobby + Pro) pushing to the agent webhook."""
    settings = Settings()
    resolved_token = token if token is not None else settings.vercel_api_token
    resolved_team = team_id if team_id is not None else settings.vercel_team_id
    resolved_project = project if project is not None else settings.vercel_project_filter
    resolved_policy = policy if policy is not None else settings.vercel_priority_policy
    resolved_secret = webhook_secret if webhook_secret is not None else settings.webhook_secret

    def _print_result(res: Any) -> None:
        style = {
            "notified": "bold green",
            "seeded": "cyan",
            "up_to_date": "dim",
            "error": "bold red",
        }.get(res.status, "")
        console.print(f"[{style}]\\[{res.status}][/] {res.message}")

    if once:
        res = asyncio.run(
            poll_vercel_and_notify_once(
                api_token=resolved_token,
                daemon_url=url,
                state_path=state_file,
                team_id=resolved_team,
                project_filter=resolved_project,
                target_filter=target,
                policy=resolved_policy,
                force=force,
                process_now=process_now,
                webhook_secret=resolved_secret,
            )
        )
        _print_result(res)
        if res.status == "error":
            raise typer.Exit(code=1)
        return

    console.print(
        Panel.fit(
            f"Watching [bold cyan]Vercel deployments[/] every [bold]{interval}s[/] "
            f"(policy=[cyan]{resolved_policy}[/])\n"
            f"Projects: [dim]{resolved_project or 'all'}[/] | "
            f"Target: [dim]{target or 'production + preview'}[/]\n"
            f"Webhook: [cyan]{url.rstrip('/')}/api/events/webhook[/] | Press [bold]Ctrl+C[/] to stop.",
            title="Vercel Deployment Watcher",
        )
    )

    async def _run_loop() -> None:
        stop_event = asyncio.Event()
        loop = asyncio.get_running_loop()

        def _on_signal(_sig: int, _frame: FrameType | None) -> None:
            loop.call_soon_threadsafe(stop_event.set)

        prev_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, _on_signal)
        if hasattr(signal, "SIGBREAK"):
            signal.signal(signal.SIGBREAK, _on_signal)

        first_iteration = True
        try:
            while not stop_event.is_set():
                res = await poll_vercel_and_notify_once(
                    api_token=resolved_token,
                    daemon_url=url,
                    state_path=state_file,
                    team_id=resolved_team,
                    project_filter=resolved_project,
                    target_filter=target,
                    policy=resolved_policy,
                    force=force and first_iteration,
                    process_now=process_now,
                    webhook_secret=resolved_secret,
                )
                first_iteration = False
                _print_result(res)
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=interval)
                except asyncio.TimeoutError:
                    continue
        finally:
            signal.signal(signal.SIGINT, prev_sigint)

    try:
        asyncio.run(_run_loop())
    except KeyboardInterrupt:
        pass
    console.print("[dim]Vercel deployment watcher stopped.[/]")


def _fetch_json(endpoint: str) -> dict[str, Any]:
    try:
        resp = httpx.get(endpoint, timeout=30.0)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        console.print(f"[bold red]Request failed:[/] {exc}")
        raise typer.Exit(code=1) from exc


def _post_json(endpoint: str, body: dict[str, Any]) -> dict[str, Any]:
    try:
        resp = httpx.post(endpoint, json=body, timeout=120.0)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        console.print(f"[bold red]Request failed:[/] {exc}")
        raise typer.Exit(code=1) from exc


def _render_status(data: dict[str, Any]) -> None:
    fatigue = data.get("fatigue", {})
    fb = fatigue.get("breakdown", {})
    salience = data.get("salience") or {}
    cb = data.get("circuit_breaker", {})
    spend = cb.get("daily_spend", {})

    table = Table(title=f"{data.get('agent_name', 'Agent')} Live Telemetry ({data.get('model')})")
    table.add_column("Metric", style="bold cyan")
    table.add_column("Value")
    table.add_row("Last Salience", f"{salience.get('score', 0.0):.3f} (threshold > {data.get('wake_threshold')})")
    table.add_row(
        "Fatigue",
        f"{fatigue.get('score', 0.0):.3f} / {data.get('sleep_threshold')} "
        f"(tokens={fb.get('buffer_tokens', 0)}, frag={fb.get('semantic_fragmentation', 0):.2f}, "
        f"errs={fb.get('error_count', 0)}, corr={fb.get('correction_count', 0)})",
    )
    table.add_row(
        "Daily Spend (UTC)",
        f"${spend.get('usd_spent', 0.0):.5f} / ${cb.get('max_daily_usd', 1.0):.2f} "
        f"(cache_read={spend.get('cache_read_tokens', 0)} tok)",
    )
    table.add_row(
        "Web Searches (UTC)",
        f"{spend.get('web_searches', 0)} / {cb.get('max_web_searches_per_day', 50)} "
        f"(provider={data.get('web_search_provider', 'ddg')})",
    )
    table.add_row(
        "Hourly Wakes",
        f"{cb.get('wakes_last_hour', 0)} / {cb.get('max_wakes_per_hour', 30)}",
    )
    table.add_row(
        "Backoff / Errors",
        f"errors={cb.get('consecutive_errors', 0)}, in_cooldown={cb.get('in_cooldown', False)}",
    )
    table.add_row(
        "Memory Counts",
        f"core={len(data.get('core_memory', []))}, "
        f"archive={len(data.get('memory_archive', []))}, "
        f"fast_buffer={len(data.get('fast_buffer', []))}",
    )
    watchers = data.get("watchers") or {}
    if watchers:
        parts = [
            f"{name}={info.get('last_status', 'unknown')} ({info.get('interval_seconds', '?')}s)"
            for name, info in sorted(watchers.items())
        ]
        table.add_row("Watchers", ", ".join(parts))
    discord_info = data.get("discord") or {}
    if discord_info.get("enabled"):
        state_lbl = (
            f"connected ({discord_info.get('bot_user')})"
            if discord_info.get("connected")
            else f"disconnected ({discord_info.get('last_error') or 'starting'})"
        )
        table.add_row("Discord Bot", state_lbl)
    mcp_info = data.get("mcp") or {}
    if mcp_info.get("server_count", 0) > 0:
        table.add_row(
            "MCP Servers",
            f"{mcp_info.get('connected_count', 0)}/{mcp_info.get('server_count', 0)} connected "
            f"({mcp_info.get('tool_count', 0)} tools)",
        )
    console.print(table)


def _render_core_memory(items: list[dict[str, Any]]) -> None:
    table = Table(title=f"Active Core Memory ({len(items)})")
    table.add_column("ID", style="dim")
    table.add_column("Cat", style="cyan")
    table.add_column("Pin")
    table.add_column("Conf")
    table.add_column("Content")
    for m in items:
        table.add_row(
            m["id"],
            m["category"],
            "PIN" if m["pinned"] else "",
            f"{m['confidence']:.2f}",
            m["content"],
        )
    console.print(table)


def _render_archive(items: list[dict[str, Any]]) -> None:
    table = Table(title=f"Archived Memory History ({len(items)})")
    table.add_column("Memory ID", style="dim")
    table.add_column("v")
    table.add_column("Action", style="yellow")
    table.add_column("Reason")
    table.add_column("Content")
    for a in items[:25]:
        table.add_row(
            a["memory_id"],
            str(a["version"]),
            a["action"],
            a["reason"],
            a["content"],
        )
    console.print(table)


def _render_sleep_report(report: dict[str, Any]) -> None:
    console.print(
        Panel(
            f"Cycle: [cyan]{report.get('sleep_cycle_id')}[/]\n"
            f"Status: [bold]{report.get('status')}[/] (trigger={report.get('trigger_reason')})\n"
            f"Episodes processed: {report.get('episodes_processed')}\n"
            f"Added: {len(report.get('added_ids', []))} | "
            f"Updated: {len(report.get('updated_ids', []))} | "
            f"Archived: {len(report.get('archived_ids', []))} | "
            f"Rejected: {len(report.get('rejected_operations', []))}\n"
            f"Cost: ${float(report.get('usd_cost', 0.0)):.5f}\n"
            f"Reflection: {report.get('reflection') or report.get('abort_reason') or '—'}",
            title="Sleep Consolidation Report",
        )
    )


if __name__ == "__main__":
    app()
