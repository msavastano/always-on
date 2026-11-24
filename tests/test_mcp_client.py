"""Unit and real-subprocess integration tests for external MCP client support."""

from __future__ import annotations

import contextlib
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from always_on.config import Settings
from always_on.daemon import AlwaysOnDaemon
from always_on.embeddings.provider import HashEmbeddingProvider
from always_on.tools.mcp_client import (
    MCPClientManager,
    MCPServerConfig,
    build_anthropic_tool_name,
    format_mcp_tool_result,
    load_dotenv_lookup,
    normalize_mcp_input_schema,
    parse_mcp_servers_config,
)
from tests.conftest import FakeBlock, FakeMessage, make_mock_client

_ANTHROPIC_TOOL_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


def test_parse_mcp_servers_config_env_expansion_and_missing_file_noop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing = tmp_path / "does_not_exist.json"
    assert parse_mcp_servers_config(config_path=missing) == []

    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        'DOTENV_TOKEN="from_dotenv"\nOVERRIDE_VAR=from_dotenv\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("OVERRIDE_VAR", "from_os_env")
    lookup = load_dotenv_lookup(dotenv_path)
    assert lookup["DOTENV_TOKEN"] == "from_dotenv"
    assert lookup["OVERRIDE_VAR"] == "from_os_env"

    cfg_path = tmp_path / "mcp_servers.json"
    cfg_path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "github": {
                        "command": "npx",
                        "args": ["-y", "${GITHUB_PKG:-@modelcontextprotocol/server-github}"],
                        "env": {
                            "TOKEN": "${DOTENV_TOKEN}",
                            "OVERRIDE": "${OVERRIDE_VAR}",
                        },
                    },
                    "remote-cal": {
                        "url": "https://mcp.example.com/mcp?v=${OVERRIDE_VAR}",
                        "headers": {"Authorization": "Bearer ${DOTENV_TOKEN}"},
                    },
                    "legacy-sse": {
                        "enabled": False,
                        "transport": "sse",
                        "url": "https://sse.example.com/sse",
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    servers = parse_mcp_servers_config(config_path=cfg_path, env_lookup=lookup)
    by_name = {s.name: s for s in servers}

    gh = by_name["github"]
    assert gh.enabled is True
    assert gh.transport == "stdio"
    assert gh.args == ["-y", "@modelcontextprotocol/server-github"]
    assert gh.env == {"TOKEN": "from_dotenv", "OVERRIDE": "from_os_env"}

    cal = by_name["remote-cal"]
    assert cal.transport == "streamable_http"
    assert cal.url == "https://mcp.example.com/mcp?v=from_os_env"
    assert cal.headers == {"Authorization": "Bearer from_dotenv"}

    sse = by_name["legacy-sse"]
    assert sse.enabled is False
    assert sse.transport == "sse"


def test_tool_name_sanitization_and_schema_normalization() -> None:
    short_name = build_anthropic_tool_name("google-calendar.v3", "events/list")
    assert short_name == "mcp__google-calendar_v3__events_list"
    assert _ANTHROPIC_TOOL_NAME_RE.match(short_name)

    long_a = build_anthropic_tool_name(
        "very_long_external_enterprise_server_name",
        "create_and_dispatch_multi_recipient_calendar_invitation_alpha",
    )
    long_b = build_anthropic_tool_name(
        "very_long_external_enterprise_server_name",
        "create_and_dispatch_multi_recipient_calendar_invitation_beta",
    )
    assert len(long_a) == 64
    assert len(long_b) == 64
    assert long_a != long_b
    assert _ANTHROPIC_TOOL_NAME_RE.match(long_a)
    assert _ANTHROPIC_TOOL_NAME_RE.match(long_b)

    norm = normalize_mcp_input_schema(
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "properties": {"q": {"type": "string"}},
        }
    )
    assert "$schema" not in norm
    assert norm["type"] == "object"
    assert norm["properties"] == {"q": {"type": "string"}}

    empty_norm = normalize_mcp_input_schema(None)
    assert empty_norm == {"type": "object", "properties": {}}

    long_text = "x" * 200
    framed, is_err = format_mcp_tool_result(
        SimpleNamespace(
            content=[SimpleNamespace(type="text", text=long_text)],
            is_error=False,
        ),
        server_name="github",
        tool_name="search_issues",
        max_chars=80,
    )
    assert is_err is False
    assert framed.startswith('<untrusted_mcp_content server="github" tool="search_issues">')
    assert "...[truncated]" in framed
    assert framed.endswith("</untrusted_mcp_content>")

    structured_framed, is_err_2 = format_mcp_tool_result(
        SimpleNamespace(content=[], structured_content={"ok": 1}, is_error=True),
        server_name="cal",
        tool_name="patch",
    )
    assert is_err_2 is True
    assert '"ok": 1' in structured_framed


@pytest.mark.asyncio
async def test_background_gating_allowlist_blocklist_and_fault_isolation(
    test_settings: Settings,
) -> None:
    test_settings.mcp_config_path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "broken": {"command": "does-not-matter"},
                    "calendar": {
                        "command": "does-not-matter",
                        "allowed_tools": ["list_events", "delete_event"],
                        "blocked_tools": ["delete_event"],
                    },
                    "bg_ok": {
                        "command": "does-not-matter",
                        "allow_on_background": True,
                    },
                    "off": {"enabled": False, "command": "nope"},
                }
            }
        ),
        encoding="utf-8",
    )

    class FakeMCPClient:
        def __init__(self, server_name: str) -> None:
            self.server_name = server_name

        async def list_tools(self) -> Any:
            if self.server_name == "calendar":
                return SimpleNamespace(
                    tools=[
                        SimpleNamespace(
                            name="list_events",
                            description="List events",
                            input_schema={"type": "object", "properties": {}},
                        ),
                        SimpleNamespace(
                            name="delete_event",
                            description="Delete event",
                            input_schema={"type": "object", "properties": {}},
                        ),
                        SimpleNamespace(
                            name="hidden_tool",
                            description="Hidden",
                            input_schema={"type": "object", "properties": {}},
                        ),
                    ],
                    next_cursor=None,
                )
            return SimpleNamespace(
                tools=[
                    SimpleNamespace(
                        name="ping",
                        description="Health ping",
                        input_schema={"type": "object", "properties": {}},
                    )
                ],
                next_cursor=None,
            )

        async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text=f"{self.server_name}:{name}:{arguments}")],
                is_error=False,
            )

    @contextlib.asynccontextmanager
    async def fake_session(cfg: MCPServerConfig) -> Any:
        if cfg.name == "broken":
            raise RuntimeError("Simulated handshake failure")
        yield FakeMCPClient(cfg.name)

    mgr = MCPClientManager(test_settings, session_factory=fake_session)
    snap = await mgr.start()
    try:
        assert snap["server_count"] == 4
        assert snap["connected_count"] == 2
        assert snap["tool_count"] == 2
        assert snap["servers"]["broken"]["status"] == "error"
        assert "Simulated handshake failure" in (snap["servers"]["broken"]["last_error"] or "")
        assert snap["servers"]["off"]["status"] == "disabled"
        assert snap["servers"]["calendar"]["tools"] == ["mcp__calendar__list_events"]
        assert snap["servers"]["bg_ok"]["tools"] == ["mcp__bg_ok__ping"]

        chat_names = {t["name"] for t in mgr.tool_definitions(event_source="chat")}
        assert chat_names == {"mcp__calendar__list_events", "mcp__bg_ok__ping"}

        webhook_names = {
            t["name"]
            for t in mgr.tool_definitions(
                event_source="webhook", default_allow_on_background=False
            )
        }
        assert webhook_names == {"mcp__bg_ok__ping"}

        heartbeat_all_allowed = {
            t["name"]
            for t in mgr.tool_definitions(
                event_source="heartbeat", default_allow_on_background=True
            )
        }
        assert heartbeat_all_allowed == {"mcp__calendar__list_events", "mcp__bg_ok__ping"}

        out, is_err = await mgr.call_tool("mcp__calendar__list_events", {"day": "mon"})
        assert is_err is False
        assert '<untrusted_mcp_content server="calendar" tool="list_events">' in out
        assert "calendar:list_events" in out
    finally:
        await mgr.stop()


@pytest.mark.asyncio
async def test_real_stdio_mcp_server_end_to_end_with_daemon_wake_loop(
    tmp_path: Path,
    test_settings: Settings,
    embedder: HashEmbeddingProvider,
) -> None:
    server_script = tmp_path / "demo_calendar_mcp.py"
    server_script.write_text(
        "\n".join(
            [
                "from mcp.server.mcpserver import MCPServer",
                "",
                'mcp = MCPServer("demo_calendar")',
                "",
                '@mcp.tool(description="List scheduled events for a specific date.")',
                "def list_events(date: str) -> str:",
                '    return f"Events on {date}: Nina volleyball practice 8:00 PM at West Gym"',
                "",
                'if __name__ == "__main__":',
                '    mcp.run(transport="stdio")',
                "",
            ]
        ),
        encoding="utf-8",
    )

    test_settings.mcp_config_path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "demo_calendar": {
                        "command": sys.executable,
                        "args": [str(server_script)],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    llm_client, sdk = make_mock_client(
        test_settings,
        [
            FakeMessage(
                stop_reason="tool_use",
                content=[
                    FakeBlock(
                        type="tool_use",
                        id="toolu_mcp_1",
                        name="mcp__demo_calendar__list_events",
                        input={"date": "2026-11-25"},
                    )
                ],
            ),
            FakeMessage(
                stop_reason="end_turn",
                content=[
                    FakeBlock(
                        type="text",
                        text="Nina has volleyball practice on 2026-11-25 at 8:00 PM at West Gym.",
                    )
                ],
            ),
        ],
    )

    daemon = AlwaysOnDaemon(test_settings, embedder=embedder, llm_client=llm_client)
    await daemon.start(run_background_loop=False)
    try:
        snap = await daemon.get_telemetry_snapshot()
        assert snap["mcp"]["connected_count"] == 1
        assert snap["mcp"]["tool_count"] == 1
        assert "mcp__demo_calendar__list_events" in snap["mcp"]["servers"]["demo_calendar"]["tools"]

        reply = await daemon.send_chat("What is on the calendar for 2026-11-25?")
        assert reply.status == "ok"
        assert "West Gym" in reply.reply
        assert len(reply.tool_calls) == 1
        assert reply.tool_calls[0].name == "mcp__demo_calendar__list_events"
        assert reply.tool_calls[0].is_error is False
        assert '<untrusted_mcp_content server="demo_calendar" tool="list_events">' in reply.tool_calls[0].result
        assert "8:00 PM at West Gym" in reply.tool_calls[0].result

        # Confirm tool schema was registered cleanly on the first LLM request
        call0_tools = {t["name"]: t for t in sdk.messages.calls[0]["tools"]}
        assert "mcp__demo_calendar__list_events" in call0_tools
        assert call0_tools["mcp__demo_calendar__list_events"]["description"].startswith(
            "[MCP:demo_calendar]"
        )

        # Confirm background webhook wake-ups do NOT expose chat-only MCP tools
        bg_tool_names = {
            t["name"] for t in daemon.tools.tool_definitions(event_source="webhook")
        }
        assert "mcp__demo_calendar__list_events" not in bg_tool_names
    finally:
        await daemon.stop()
