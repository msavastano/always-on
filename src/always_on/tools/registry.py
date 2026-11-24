"""Safe, local-first tool registry exposed to Claude Haiku 5.5 during wake-ups."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from always_on.config import Settings
from always_on.db.store import AgentStore
from always_on.safety.circuit_breaker import AgentCircuitBreaker
from always_on.tools.mcp_client import MCPClientManager
from always_on.tools.web_search import (
    WebSearchProvider,
    WebSearchService,
    safe_fetch_webpage,
)


def _safe_resolve(base_dir: Path, filename: str) -> Path:
    """Resolve filename strictly inside base_dir, rejecting path traversal."""
    base_dir.mkdir(parents=True, exist_ok=True)
    cleaned = filename.strip().lstrip("/\\")
    if not cleaned:
        raise ValueError("Filename must not be empty.")
    target = (base_dir / cleaned).resolve()
    base_resolved = base_dir.resolve()
    if base_resolved not in target.parents:
        raise ValueError(f"Path traversal blocked: '{filename}' escapes {base_dir.name}/.")
    return target


class ToolRegistry:
    """Executes safe local & web tools and exports Anthropic-compatible tool definitions."""

    def __init__(
        self,
        *,
        store: AgentStore,
        inbox_dir: Path,
        notes_dir: Path,
        settings: Settings | None = None,
        circuit_breaker: AgentCircuitBreaker | None = None,
        web_search_provider: WebSearchProvider | None = None,
        mcp_manager: MCPClientManager | None = None,
    ) -> None:
        self.store = store
        self.inbox_dir = inbox_dir.resolve()
        self.notes_dir = notes_dir.resolve()
        self.settings = settings or Settings()
        self.circuit_breaker = circuit_breaker
        self.web_search_service: WebSearchService | None = (
            WebSearchService(
                store=store,
                circuit_breaker=circuit_breaker,
                settings=self.settings,
                provider=web_search_provider,
            )
            if circuit_breaker is not None
            and self.settings.web_search_provider not in ("anthropic", "disabled")
            else None
        )
        self.mcp_manager: MCPClientManager | None = (
            mcp_manager
            if mcp_manager is not None
            else (MCPClientManager(self.settings) if self.settings.enable_mcp else None)
        )
        self.inbox_dir.mkdir(parents=True, exist_ok=True)
        self.notes_dir.mkdir(parents=True, exist_ok=True)

    async def start(self) -> dict[str, Any]:
        """Connect configured external MCP servers (no-op if disabled or unconfigured)."""
        if self.mcp_manager is None:
            return self.mcp_status()
        return await self.mcp_manager.start()

    async def stop(self) -> None:
        """Close external MCP server sessions cleanly."""
        if self.mcp_manager is not None:
            await self.mcp_manager.stop()

    def mcp_status(self) -> dict[str, Any]:
        """Return current MCP client connection and tool summary."""
        if self.mcp_manager is None:
            return {
                "enabled": False,
                "config_path": str(self.settings.mcp_config_path),
                "config_error": None,
                "server_count": 0,
                "connected_count": 0,
                "tool_count": 0,
                "servers": {},
            }
        return self.mcp_manager.status()

    def tool_definitions(
        self,
        *,
        event_source: str = "chat",
        search_quota_remaining: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return Anthropic Messages API tool schemas (with optional search/fetch tools)."""
        defs: list[dict[str, Any]] = [
            {
                "name": "get_current_time",
                "description": "Get the current UTC and local system time.",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            },
            {
                "name": "list_inbox_files",
                "description": "List files currently in the watched local inbox directory.",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            },
            {
                "name": "read_inbox_file",
                "description": "Read UTF-8 text content of a file inside the local inbox directory (max 8000 chars).",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "description": "Relative filename inside inbox/ to read.",
                        }
                    },
                    "required": ["filename"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "list_notes",
                "description": "List saved personal notes in the local notes directory.",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            },
            {
                "name": "read_note",
                "description": "Read a saved personal note from the local notes directory.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "description": "Filename of the note inside notes/.",
                        }
                    },
                    "required": ["filename"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "write_note",
                "description": "Create or overwrite a markdown/text note in the local notes directory.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "description": "Filename inside notes/ (e.g. 'project-plan.md').",
                        },
                        "content": {
                            "type": "string",
                            "description": "UTF-8 text content to write.",
                        },
                    },
                    "required": ["filename", "content"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "search_core_memory",
                "description": "Search active Core Memory items by keyword or category (read-only).",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Optional case-insensitive keyword filter.",
                        },
                        "category": {
                            "type": "string",
                            "enum": ["fact", "preference", "project", "lesson"],
                            "description": "Optional category filter.",
                        },
                    },
                    "additionalProperties": False,
                },
            },
        ]

        search_allowed = (
            self.settings.web_search_provider != "disabled"
            and (event_source != "heartbeat" or self.settings.allow_search_on_heartbeat)
        )

        if search_allowed:
            if self.settings.web_search_provider == "anthropic":
                max_uses = self.settings.anthropic_web_search_max_uses
                if search_quota_remaining is not None:
                    max_uses = min(max_uses, max(0, search_quota_remaining))
                if max_uses > 0:
                    defs.append(
                        {
                            "type": "web_search_20260318",
                            "name": "web_search",
                            "max_uses": max_uses,
                        }
                    )
            elif self.web_search_service is not None:
                defs.append(
                    {
                        "name": "web_search",
                        "description": (
                            "Search the public web for up-to-date information and return compact "
                            "snippets with source URLs (cached locally in SQLite and quota-guarded)."
                        ),
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "Focused search query.",
                                },
                                "max_results": {
                                    "type": "integer",
                                    "description": "Maximum results to return (1 to 5).",
                                },
                                "recency_days": {
                                    "type": "integer",
                                    "description": "Optional recency filter in days (e.g. 1, 7, 30).",
                                },
                            },
                            "required": ["query"],
                            "additionalProperties": False,
                        },
                    }
                )

            if self.settings.enable_fetch_webpage:
                defs.append(
                    {
                        "name": "fetch_webpage",
                        "description": (
                            "Fetch readable plain text from a public http/https URL when search "
                            "snippets are insufficient (SSRF-protected and truncated to protect memory budget)."
                        ),
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "url": {
                                    "type": "string",
                                    "description": "Public http:// or https:// URL to fetch.",
                                }
                            },
                            "required": ["url"],
                            "additionalProperties": False,
                        },
                    }
                )

        # Anthropic native server-side sandboxed code execution (isolated cloud container)
        code_exec_allowed = self.settings.enable_code_execution and (
            event_source == "chat" or self.settings.allow_code_execution_on_background
        )
        if code_exec_allowed:
            defs.append(
                {
                    "type": self.settings.code_execution_tool_version,
                    "name": "code_execution",
                }
            )

        if self.mcp_manager is not None:
            defs.extend(
                self.mcp_manager.tool_definitions(
                    event_source=event_source,
                    default_allow_on_background=self.settings.allow_mcp_on_background,
                )
            )

        return defs

    async def execute(self, name: str, args: dict[str, Any]) -> tuple[str, bool]:
        """Run a tool by name, returning (result_text, is_error)."""
        try:
            if name == "get_current_time":
                now_utc = datetime.now(timezone.utc)
                now_local = datetime.now().astimezone()
                return (
                    json.dumps(
                        {
                            "utc": now_utc.isoformat(),
                            "local": now_local.isoformat(),
                        }
                    ),
                    False,
                )

            if name == "list_inbox_files":
                files = []
                for p in sorted(self.inbox_dir.rglob("*")):
                    if p.is_file():
                        stat = p.stat()
                        files.append(
                            {
                                "filename": p.relative_to(self.inbox_dir).as_posix(),
                                "size_bytes": stat.st_size,
                                "modified_utc": datetime.fromtimestamp(
                                    stat.st_mtime, tz=timezone.utc
                                ).isoformat(),
                            }
                        )
                return json.dumps({"files": files}), False

            if name == "read_inbox_file":
                filename = str(args.get("filename", ""))
                target = _safe_resolve(self.inbox_dir, filename)
                if not target.is_file():
                    return f"Error: inbox file '{filename}' does not exist.", True
                text = target.read_text(encoding="utf-8", errors="replace")[:8000]
                return text, False

            if name == "list_notes":
                notes = [
                    p.relative_to(self.notes_dir).as_posix()
                    for p in sorted(self.notes_dir.rglob("*"))
                    if p.is_file()
                ]
                return json.dumps({"notes": notes}), False

            if name == "read_note":
                filename = str(args.get("filename", ""))
                target = _safe_resolve(self.notes_dir, filename)
                if not target.is_file():
                    return f"Error: note '{filename}' does not exist.", True
                return target.read_text(encoding="utf-8", errors="replace")[:8000], False

            if name == "write_note":
                filename = str(args.get("filename", ""))
                content = str(args.get("content", ""))
                target = _safe_resolve(self.notes_dir, filename)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
                return json.dumps(
                    {
                        "status": "written",
                        "filename": target.relative_to(self.notes_dir.resolve()).as_posix(),
                        "bytes": len(content.encode("utf-8")),
                    }
                ), False

            if name == "search_core_memory":
                query = str(args.get("query", "")).strip().lower()
                category = args.get("category")
                items = await self.store.list_core_memory()
                matches = []
                for item in items:
                    if category and item.category != category:
                        continue
                    if query and query not in item.content.lower():
                        continue
                    matches.append(
                        {
                            "id": item.id,
                            "category": item.category,
                            "content": item.content,
                            "pinned": item.pinned,
                            "confidence": item.confidence,
                        }
                    )
                return json.dumps({"matches": matches}), False

            if name == "web_search":
                if self.web_search_service is None:
                    return (
                        f"Web search tool is unavailable (provider={self.settings.web_search_provider}).",
                        True,
                    )
                query = str(args.get("query", ""))
                max_results = args.get("max_results")
                recency_days = args.get("recency_days")
                return await self.web_search_service.search(
                    query,
                    max_results=int(max_results) if max_results is not None else None,
                    recency_days=int(recency_days) if recency_days is not None else None,
                )

            if name == "fetch_webpage":
                if not self.settings.enable_fetch_webpage:
                    return "Error: fetch_webpage tool is disabled in configuration.", True
                url = str(args.get("url", ""))
                return await safe_fetch_webpage(
                    url,
                    max_chars=self.settings.fetch_webpage_max_chars,
                    timeout_seconds=self.settings.web_search_timeout_seconds,
                )

            if self.mcp_manager is not None and self.mcp_manager.has_tool(name):
                return await self.mcp_manager.call_tool(name, args)

            return f"Error: unknown tool '{name}'.", True
        except Exception as exc:
            return f"Tool error ({name}): {exc}", True
