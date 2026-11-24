"""External Model Context Protocol (MCP) client manager with task-scoped transports."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import re
import shutil
from collections.abc import Callable, Mapping
from contextlib import AbstractAsyncContextManager, AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from always_on.config import Settings

logger = logging.getLogger("always_on.mcp")

_ENV_PLACEHOLDER_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")
_SAFE_TOOL_PART_RE = re.compile(r"[^a-zA-Z0-9_-]+")


class MCPServerConfig(BaseModel):
    """Configuration for a single external MCP server (`stdio`, `streamable_http`, or `sse`)."""

    name: str
    enabled: bool = True
    transport: Literal["stdio", "streamable_http", "sse"] = "stdio"
    command: str = ""
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    cwd: str | None = None
    url: str = ""
    headers: dict[str, str] = Field(default_factory=dict)
    allow_on_background: bool | None = None
    allowed_tools: list[str] = Field(default_factory=list)
    blocked_tools: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _infer_and_normalize_transport(cls, raw: Any) -> Any:
        if not isinstance(raw, Mapping):
            return raw
        data = dict(raw)
        explicit_transport = data.get("transport")
        if isinstance(explicit_transport, str):
            norm = explicit_transport.strip().lower().replace("-", "_")
            if norm == "http":
                norm = "streamable_http"
            data["transport"] = norm
        elif not explicit_transport and str(data.get("url") or "").strip():
            data["transport"] = "streamable_http"
        return data


@dataclass(frozen=True)
class MCPToolBinding:
    """Mapped remote MCP tool exposed to Anthropic under a safe `mcp__*` name."""

    anthropic_name: str
    server_name: str
    remote_tool_name: str
    description: str
    input_schema: dict[str, Any]
    allow_on_background: bool | None


@dataclass
class _ServerRuntime:
    config: MCPServerConfig
    effective_allow_on_background: bool
    status: str = "stopped"
    last_error: str | None = None
    tool_names: list[str] | None = None
    client: Any | None = None
    ready_event: asyncio.Event | None = None
    stop_event: asyncio.Event | None = None
    task: asyncio.Task[None] | None = None


def load_dotenv_lookup(dotenv_path: Path = Path(".env")) -> dict[str, str]:
    """Read `.env` KEY=VALUE pairs merged with `os.environ` (process env wins)."""
    merged: dict[str, str] = {}
    if dotenv_path.is_file():
        try:
            for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                if line.startswith("export "):
                    line = line[len("export ") :].strip()
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip()
                if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
                    val = val[1:-1]
                if key:
                    merged[key] = val
        except OSError as exc:
            logger.warning("Could not read %s for MCP env expansion: %s", dotenv_path, exc)
    merged.update(os.environ)
    return merged


def expand_env_placeholders(raw: str, env_lookup: Mapping[str, str]) -> str:
    """Expand `${VAR}` and `${VAR:-default}` placeholders using `env_lookup`."""
    if "${" not in raw:
        return raw

    def _replace(match: re.Match[str]) -> str:
        var_name = match.group(1)
        default_val = match.group(2)
        if var_name in env_lookup:
            return str(env_lookup[var_name])
        return default_val if default_val is not None else ""

    return _ENV_PLACEHOLDER_RE.sub(_replace, raw)


def _expand_entry_fields(entry: dict[str, Any], env_lookup: Mapping[str, str]) -> dict[str, Any]:
    out = dict(entry)
    for key in ("command", "cwd", "url"):
        if isinstance(out.get(key), str):
            out[key] = expand_env_placeholders(out[key], env_lookup)
    if isinstance(out.get("args"), list):
        out["args"] = [
            expand_env_placeholders(str(arg), env_lookup) for arg in out["args"]
        ]
    for map_key in ("env", "headers"):
        if isinstance(out.get(map_key), Mapping):
            out[map_key] = {
                str(k): expand_env_placeholders(str(v), env_lookup)
                for k, v in out[map_key].items()
            }
    return out


def parse_mcp_servers_config(
    *,
    config_path: Path | None,
    inline_json: str = "",
    env_lookup: Mapping[str, str] | None = None,
) -> list[MCPServerConfig]:
    """Parse Claude-Desktop-style `mcpServers` JSON config with `${VAR}` env expansion."""
    raw_text = (inline_json or "").strip()
    if not raw_text:
        if config_path is None or not config_path.is_file():
            return []
        raw_text = config_path.read_text(encoding="utf-8").strip()
        if not raw_text:
            return []

    parsed = json.loads(raw_text)
    if not isinstance(parsed, Mapping):
        raise ValueError("MCP config root must be a JSON object.")

    lookup = env_lookup if env_lookup is not None else load_dotenv_lookup()
    raw_entries: list[tuple[str, Mapping[str, Any]]] = []

    mcp_servers = parsed.get("mcpServers")
    if isinstance(mcp_servers, Mapping):
        for name, body in mcp_servers.items():
            if not isinstance(body, Mapping):
                raise ValueError(f"MCP server '{name}' configuration must be a JSON object.")
            raw_entries.append((str(name), body))
    elif isinstance(parsed.get("servers"), list):
        for idx, body in enumerate(parsed["servers"]):
            if not isinstance(body, Mapping) or not body.get("name"):
                raise ValueError(f"MCP servers[{idx}] must be an object with a 'name' field.")
            raw_entries.append((str(body["name"]), body))
    elif parsed:
        raise ValueError("MCP config must contain an 'mcpServers' object (or 'servers' list).")

    configs: list[MCPServerConfig] = []
    for name, body in raw_entries:
        expanded = _expand_entry_fields({"name": name, **dict(body)}, lookup)
        configs.append(MCPServerConfig.model_validate(expanded))
    return configs


def resolve_stdio_command(command: str) -> str:
    """Resolve `npx`/`uvx`/executable paths on Windows so subprocess launch succeeds."""
    cleaned = command.strip()
    if not cleaned:
        return cleaned
    if os.name == "nt":
        resolved = shutil.which(cleaned)
        if resolved:
            return resolved
    return cleaned


def build_anthropic_tool_name(server_name: str, tool_name: str) -> str:
    """Map `(server_name, tool_name)` to a unique Anthropic-safe name `<= 64` chars."""
    s_clean = _SAFE_TOOL_PART_RE.sub("_", server_name).strip("_") or "server"
    t_clean = _SAFE_TOOL_PART_RE.sub("_", tool_name).strip("_") or "tool"
    base = f"mcp__{s_clean}__{t_clean}"
    if len(base) <= 64:
        return base
    digest = hashlib.sha256(f"{server_name}:{tool_name}".encode("utf-8")).hexdigest()[:8]
    return f"{base[:55]}_{digest}"


def normalize_mcp_input_schema(raw_schema: Any) -> dict[str, Any]:
    """Normalize an MCP tool's JSON Schema for Anthropic's `input_schema` constraint."""
    if raw_schema is None:
        schema: dict[str, Any] = {}
    elif hasattr(raw_schema, "model_dump"):
        schema = dict(raw_schema.model_dump())
    elif isinstance(raw_schema, Mapping):
        schema = dict(raw_schema)
    else:
        schema = {}

    schema.pop("$schema", None)
    schema["type"] = "object"
    props = schema.get("properties")
    if not isinstance(props, Mapping):
        schema["properties"] = {}
    else:
        schema["properties"] = dict(props)
    return schema


def _extract_content_text(result: Any) -> str:
    pieces: list[str] = []
    content = getattr(result, "content", None) or ()
    for item in content:
        text = getattr(item, "text", None)
        if text is not None:
            pieces.append(str(text))
        elif isinstance(item, Mapping):
            if item.get("text") is not None:
                pieces.append(str(item["text"]))
            else:
                pieces.append(json.dumps(dict(item), ensure_ascii=False, default=str))
        elif hasattr(item, "model_dump"):
            pieces.append(json.dumps(item.model_dump(), ensure_ascii=False, default=str))
        else:
            pieces.append(str(item))

    joined = "\n".join(p for p in pieces if p).strip()
    if joined:
        return joined

    structured = getattr(result, "structured_content", None)
    if structured is None:
        structured = getattr(result, "structuredContent", None)
    if structured is not None:
        return json.dumps(structured, indent=2, ensure_ascii=False, default=str)

    return "(empty tool result)"


def format_mcp_tool_result(
    result: Any,
    *,
    server_name: str,
    tool_name: str,
    max_chars: int = 8000,
) -> tuple[str, bool]:
    """Extract, truncate, and untrusted-frame an MCP `call_tool` result."""
    is_error = bool(getattr(result, "is_error", False) or getattr(result, "isError", False))
    body = _extract_content_text(result)
    marker = "\n...[truncated]"
    if max_chars > 0 and len(body) > max_chars:
        keep = max(1, max_chars - len(marker))
        body = body[:keep] + marker

    framed = (
        f'<untrusted_mcp_content server="{server_name}" tool="{tool_name}">\n'
        f"{body}\n"
        "</untrusted_mcp_content>"
    )
    return framed, is_error


def _format_mcp_error(exc: BaseException) -> str:
    """Unwrap anyio BaseExceptionGroup wrappers into concrete leaf error messages."""
    if isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        parts = [_format_mcp_error(sub) for sub in exc.exceptions]
        return "; ".join(p for p in parts if p) or str(exc)
    detail = str(exc).strip()
    return detail or type(exc).__name__


class MCPClientManager:
    """Manages task-scoped MCP client connections, tool routing, and safety gating."""

    def __init__(
        self,
        settings: Settings,
        *,
        session_factory: Callable[[MCPServerConfig], AbstractAsyncContextManager[Any]]
        | None = None,
    ) -> None:
        self.settings = settings
        self._session_factory = session_factory
        self._runtimes: dict[str, _ServerRuntime] = {}
        self._bindings: dict[str, MCPToolBinding] = {}
        self._config_error: str | None = None
        self._started = False

    def has_tool(self, anthropic_name: str) -> bool:
        return anthropic_name in self._bindings

    def _effective_background_flag(self, cfg: MCPServerConfig) -> bool:
        if cfg.allow_on_background is not None:
            return cfg.allow_on_background
        return self.settings.allow_mcp_on_background

    @contextlib.asynccontextmanager
    async def _open_session(self, cfg: MCPServerConfig) -> Any:
        if self._session_factory is not None:
            async with self._session_factory(cfg) as client:
                yield client
            return

        from mcp import Client, StdioServerParameters
        from mcp.client.sse import sse_client
        from mcp.client.streamable_http import streamable_http_client

        read_timeout = self.settings.mcp_tool_timeout_seconds
        async with AsyncExitStack() as stack:
            if cfg.transport == "stdio":
                cmd = resolve_stdio_command(cfg.command)
                if not cmd:
                    raise ValueError(f"MCP server '{cfg.name}' (stdio) requires 'command'.")
                merged_env = {**os.environ, **cfg.env} if cfg.env else None
                params = StdioServerParameters(
                    command=cmd,
                    args=cfg.args,
                    env=merged_env,
                    cwd=cfg.cwd,
                )
                client = await stack.enter_async_context(
                    Client(params, read_timeout_seconds=read_timeout)
                )
                yield client
                return

            if cfg.transport == "streamable_http":
                url = cfg.url.strip()
                if not url:
                    raise ValueError(f"MCP server '{cfg.name}' (streamable_http) requires 'url'.")
                if cfg.headers:
                    import httpx2

                    http_client = await stack.enter_async_context(
                        httpx2.AsyncClient(
                            headers=cfg.headers,
                            timeout=httpx2.Timeout(read_timeout, read=300.0),
                        )
                    )
                    transport = streamable_http_client(url, http_client=http_client)
                    client = await stack.enter_async_context(
                        Client(transport, read_timeout_seconds=read_timeout)
                    )
                else:
                    client = await stack.enter_async_context(
                        Client(url, read_timeout_seconds=read_timeout)
                    )
                yield client
                return

            if cfg.transport == "sse":
                url = cfg.url.strip()
                if not url:
                    raise ValueError(f"MCP server '{cfg.name}' (sse) requires 'url'.")
                transport = sse_client(
                    url,
                    headers=cfg.headers or None,
                    timeout=min(read_timeout, 30.0),
                )
                client = await stack.enter_async_context(
                    Client(transport, read_timeout_seconds=read_timeout)
                )
                yield client
                return

            raise ValueError(f"Unsupported MCP transport '{cfg.transport}' for server '{cfg.name}'.")

    def _register_server_tools(self, runtime: _ServerRuntime, tools: list[Any]) -> list[str]:
        cfg = runtime.config
        allowed_set = set(cfg.allowed_tools)
        blocked_set = set(cfg.blocked_tools)
        registered: list[str] = []

        for tool in tools:
            remote_name = str(getattr(tool, "name", "") or "")
            if not remote_name:
                continue
            if allowed_set and remote_name not in allowed_set:
                continue
            if remote_name in blocked_set:
                continue

            base_name = build_anthropic_tool_name(cfg.name, remote_name)
            candidate = base_name
            suffix = 2
            while candidate in self._bindings:
                tag = f"_{suffix}"
                candidate = f"{base_name[: 64 - len(tag)]}{tag}"
                suffix += 1

            raw_schema = getattr(tool, "input_schema", None)
            if raw_schema is None:
                raw_schema = getattr(tool, "inputSchema", None)

            desc = str(getattr(tool, "description", "") or "").strip()
            self._bindings[candidate] = MCPToolBinding(
                anthropic_name=candidate,
                server_name=cfg.name,
                remote_tool_name=remote_name,
                description=desc,
                input_schema=normalize_mcp_input_schema(raw_schema),
                allow_on_background=cfg.allow_on_background,
            )
            registered.append(candidate)

        return registered

    async def _list_all_tools(self, client: Any) -> list[Any]:
        collected: list[Any] = []
        cursor: str | None = None
        for _ in range(20):
            if cursor is not None:
                res = await client.list_tools(cursor=cursor)
            else:
                res = await client.list_tools()
            collected.extend(getattr(res, "tools", None) or ())
            next_cursor = getattr(res, "next_cursor", None) or getattr(res, "nextCursor", None)
            if not next_cursor:
                break
            cursor = str(next_cursor)
        return collected

    async def _server_worker(self, runtime: _ServerRuntime) -> None:
        cfg = runtime.config
        assert runtime.ready_event is not None
        assert runtime.stop_event is not None

        try:
            async with self._open_session(cfg) as client:
                tools = await self._list_all_tools(client)
                runtime.client = client
                runtime.tool_names = self._register_server_tools(runtime, tools)
                runtime.status = "connected"
                runtime.last_error = None
                logger.info(
                    "Connected MCP server '%s' (%s): %d tool(s)",
                    cfg.name,
                    cfg.transport,
                    len(runtime.tool_names),
                )
                runtime.ready_event.set()
                await runtime.stop_event.wait()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            err_msg = _format_mcp_error(exc)
            runtime.status = "error"
            runtime.last_error = err_msg
            runtime.client = None
            logger.warning("MCP server '%s' failed (%s): %s", cfg.name, cfg.transport, err_msg)
            runtime.ready_event.set()

    async def start(self) -> dict[str, Any]:
        """Load MCP configuration and connect enabled servers in dedicated tasks."""
        if self._started:
            return self.status()

        self._runtimes.clear()
        self._bindings.clear()
        self._config_error = None
        self._started = True

        if not self.settings.enable_mcp:
            return self.status()

        try:
            configs = parse_mcp_servers_config(
                config_path=self.settings.mcp_config_path,
                inline_json=self.settings.mcp_servers_json,
            )
        except Exception as exc:
            self._config_error = f"Failed parsing MCP config: {exc}"
            logger.warning(self._config_error)
            return self.status()

        if not configs:
            return self.status()

        pending_waiters: list[tuple[_ServerRuntime, asyncio.Event]] = []
        for cfg in configs:
            rt = _ServerRuntime(
                config=cfg,
                effective_allow_on_background=self._effective_background_flag(cfg),
                tool_names=[],
            )
            self._runtimes[cfg.name] = rt
            if not cfg.enabled:
                rt.status = "disabled"
                continue

            rt.status = "connecting"
            rt.ready_event = asyncio.Event()
            rt.stop_event = asyncio.Event()
            rt.task = asyncio.create_task(
                self._server_worker(rt),
                name=f"mcp-{cfg.name}",
            )
            pending_waiters.append((rt, rt.ready_event))

        if pending_waiters:
            await asyncio.gather(
                *(
                    self._await_server_ready(rt, ev)
                    for rt, ev in pending_waiters
                )
            )

        return self.status()

    async def _await_server_ready(self, runtime: _ServerRuntime, ready_event: asyncio.Event) -> None:
        timeout = self.settings.mcp_startup_timeout_seconds
        try:
            await asyncio.wait_for(ready_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            runtime.status = "error"
            runtime.last_error = f"Startup timed out after {timeout:.1f}s"
            logger.warning("MCP server '%s' timed out during startup.", runtime.config.name)
            if runtime.stop_event is not None:
                runtime.stop_event.set()
            if runtime.task is not None and not runtime.task.done():
                runtime.task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception, asyncio.TimeoutError):
                    await asyncio.wait_for(runtime.task, timeout=1.5)

    def tool_definitions(
        self,
        *,
        event_source: str = "chat",
        default_allow_on_background: bool | None = None,
    ) -> list[dict[str, Any]]:
        """Return Anthropic Messages API tool specs gated by `event_source` policy."""
        if not self.settings.enable_mcp:
            return []

        fallback_bg = (
            self.settings.allow_mcp_on_background
            if default_allow_on_background is None
            else default_allow_on_background
        )
        defs: list[dict[str, Any]] = []
        for binding in self._bindings.values():
            allow_bg = (
                binding.allow_on_background
                if binding.allow_on_background is not None
                else fallback_bg
            )
            if event_source != "chat" and not allow_bg:
                continue
            prefix = f"[MCP:{binding.server_name}] "
            desc = (prefix + binding.description).strip()
            defs.append(
                {
                    "name": binding.anthropic_name,
                    "description": desc,
                    "input_schema": binding.input_schema,
                }
            )
        return defs

    async def call_tool(
        self, anthropic_name: str, args: dict[str, Any] | None = None
    ) -> tuple[str, bool]:
        """Invoke a connected remote MCP tool with timeout and untrusted framing."""
        binding = self._bindings.get(anthropic_name)
        if binding is None:
            return f"Error: unknown MCP tool '{anthropic_name}'.", True

        runtime = self._runtimes.get(binding.server_name)
        if runtime is None or runtime.status != "connected" or runtime.client is None:
            detail = runtime.last_error if runtime and runtime.last_error else "not connected"
            return f"MCP error ({anthropic_name}): server '{binding.server_name}' is {detail}.", True

        try:
            raw_res = await asyncio.wait_for(
                runtime.client.call_tool(binding.remote_tool_name, args or {}),
                timeout=self.settings.mcp_tool_timeout_seconds,
            )
            return format_mcp_tool_result(
                raw_res,
                server_name=binding.server_name,
                tool_name=binding.remote_tool_name,
                max_chars=self.settings.mcp_max_result_chars,
            )
        except Exception as exc:
            return f"MCP error ({anthropic_name}): {_format_mcp_error(exc)}", True

    def status(self) -> dict[str, Any]:
        """Return telemetry-friendly snapshot of configured MCP servers and tools."""
        servers_dict: dict[str, dict[str, Any]] = {}
        connected_count = 0
        for name, rt in self._runtimes.items():
            if rt.status == "connected":
                connected_count += 1
            servers_dict[name] = {
                "enabled": rt.config.enabled,
                "transport": rt.config.transport,
                "status": rt.status,
                "tool_count": len(rt.tool_names or ()),
                "tools": list(rt.tool_names or ()),
                "allow_on_background": rt.effective_allow_on_background,
                "last_error": rt.last_error,
            }

        return {
            "enabled": self.settings.enable_mcp,
            "config_path": str(self.settings.mcp_config_path),
            "config_error": self._config_error,
            "server_count": len(self._runtimes),
            "connected_count": connected_count,
            "tool_count": len(self._bindings),
            "servers": servers_dict,
        }

    async def stop(self) -> None:
        """Signal all MCP server worker tasks to exit inside their owning tasks."""
        tasks: list[asyncio.Task[None]] = []
        for rt in self._runtimes.values():
            rt.client = None
            if rt.stop_event is not None:
                rt.stop_event.set()
            if rt.task is not None:
                tasks.append(rt.task)

        if tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=3.0,
                )
            except asyncio.TimeoutError:
                for t in tasks:
                    if not t.done():
                        t.cancel()
                with contextlib.suppress(asyncio.CancelledError, asyncio.TimeoutError, Exception):
                    await asyncio.wait_for(
                        asyncio.gather(*tasks, return_exceptions=True),
                        timeout=1.5,
                    )

        self._runtimes.clear()
        self._bindings.clear()
        self._started = False
