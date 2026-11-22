"""Strict Claude Haiku 5.5 client enforcing adaptive thinking, caching, and API constraints."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import anthropic

from always_on.config import EffortLevel, Settings
from always_on.models import TokenUsage, ToolCallRecord

FORBIDDEN_SAMPLING_PARAMS = frozenset({"temperature", "top_p", "top_k"})


class MissingAPIKeyError(RuntimeError):
    """Raised when ANTHROPIC_API_KEY is unset or still set to a placeholder."""


@dataclass
class ToolUseItem:
    """Normalized tool_use content block extracted by block.type."""

    id: str
    name: str
    input: dict[str, Any]


@dataclass
class ParsedLLMResponse:
    """Structured response parsed strictly by content block `type`."""

    text: str
    thinking: list[str] = field(default_factory=list)
    tool_uses: list[ToolUseItem] = field(default_factory=list)
    server_web_searches: list[dict[str, Any]] = field(default_factory=list)
    server_tool_calls: list[ToolCallRecord] = field(default_factory=list)
    raw_content: list[dict[str, Any]] = field(default_factory=list)
    stop_reason: str | None = None
    container_id: str | None = None
    usage: TokenUsage = field(default_factory=TokenUsage)
    model: str = ""

    @property
    def is_refusal(self) -> bool:
        return self.stop_reason == "refusal"

    @property
    def is_context_exceeded(self) -> bool:
        return self.stop_reason == "model_context_window_exceeded"

    @property
    def is_pause_turn(self) -> bool:
        return self.stop_reason == "pause_turn"


def _block_attr(block: Any, key: str, default: Any = None) -> Any:
    if isinstance(block, dict):
        return block.get(key, default)
    return getattr(block, key, default)


def serialize_content_block(block: Any) -> dict[str, Any]:
    """Convert an SDK content block (or dict) into a JSON-serializable message block."""
    if isinstance(block, dict):
        return dict(block)
    if hasattr(block, "model_dump"):
        return block.model_dump(exclude_none=True)
    btype = _block_attr(block, "type")
    if btype == "text":
        return {"type": "text", "text": str(_block_attr(block, "text", ""))}
    if btype == "thinking":
        out: dict[str, Any] = {
            "type": "thinking",
            "thinking": str(_block_attr(block, "thinking", "")),
        }
        sig = _block_attr(block, "signature")
        if sig is not None:
            out["signature"] = sig
        return out
    if btype == "redacted_thinking":
        return {"type": "redacted_thinking", "data": _block_attr(block, "data", "")}
    if btype == "tool_use":
        return {
            "type": "tool_use",
            "id": str(_block_attr(block, "id", "")),
            "name": str(_block_attr(block, "name", "")),
            "input": dict(_block_attr(block, "input", {}) or {}),
        }
    return {"type": str(btype)}


def _format_server_tool_result(block: Any) -> tuple[str, bool]:
    """Summarize an Anthropic server-side code execution result block into (text, is_error)."""
    content = _block_attr(block, "content", None)
    if content is None:
        return "", False

    ctype = str(_block_attr(content, "type", "") or "")
    if "error" in ctype:
        err_code = _block_attr(content, "error_code", None)
        err_msg = _block_attr(content, "message", None) or err_code or ctype
        return f"Server tool error: {err_msg}", True

    stdout = str(_block_attr(content, "stdout", "") or "").strip()
    stderr = str(_block_attr(content, "stderr", "") or "").strip()
    rc = _block_attr(content, "return_code", None)

    parts: list[str] = []
    if rc is not None:
        parts.append(f"exit_code={rc}")
    if stdout:
        parts.append(f"stdout:\n{stdout[:1600]}")
    if stderr:
        parts.append(f"stderr:\n{stderr[:800]}")
    if not parts:
        # Text editor result blocks or generic dicts
        summary = serialize_content_block(content)
        parts.append(str(summary)[:1200])

    is_error = rc is not None and int(rc) != 0
    return "\n".join(parts), is_error


def parse_anthropic_response(response: Any) -> ParsedLLMResponse:
    """Extract thinking, text, tool_use, and server_tool_use blocks strictly by `block.type`."""
    content_blocks = _block_attr(response, "content", []) or []
    text_parts: list[str] = []
    thinking_parts: list[str] = []
    tool_uses: list[ToolUseItem] = []
    server_web_searches: list[dict[str, Any]] = []
    server_calls_by_id: dict[str, ToolCallRecord] = {}
    server_tool_calls: list[ToolCallRecord] = []
    raw_content: list[dict[str, Any]] = []

    for block in content_blocks:
        raw_content.append(serialize_content_block(block))
        btype = _block_attr(block, "type")
        if btype == "thinking":
            thought = str(_block_attr(block, "thinking", "") or "").strip()
            if thought:
                thinking_parts.append(thought)
        elif btype == "text":
            txt = str(_block_attr(block, "text", "") or "")
            if txt:
                text_parts.append(txt)
        elif btype == "tool_use":
            tool_uses.append(
                ToolUseItem(
                    id=str(_block_attr(block, "id", "")),
                    name=str(_block_attr(block, "name", "")),
                    input=dict(_block_attr(block, "input", {}) or {}),
                )
            )
        elif btype == "server_tool_use":
            sid = str(_block_attr(block, "id", "") or "")
            sname = str(_block_attr(block, "name", "") or "server_tool")
            sinput = dict(_block_attr(block, "input", {}) or {})
            if sname == "web_search":
                server_web_searches.append(sinput)
            rec = ToolCallRecord(
                name=f"{sname} (anthropic)",
                args=sinput,
                result="Executed server-side by Anthropic.",
                is_error=False,
            )
            server_tool_calls.append(rec)
            if sid:
                server_calls_by_id[sid] = rec
        elif btype in (
            "code_execution_tool_result",
            "bash_code_execution_tool_result",
            "text_editor_code_execution_tool_result",
        ):
            tid = str(_block_attr(block, "tool_use_id", "") or "")
            res_text, is_err = _format_server_tool_result(block)
            target_rec = server_calls_by_id.get(tid)
            if target_rec is not None:
                target_rec.result = res_text or target_rec.result
                target_rec.is_error = is_err
            else:
                server_tool_calls.append(
                    ToolCallRecord(
                        name=f"{btype} (anthropic)",
                        args={"tool_use_id": tid},
                        result=res_text,
                        is_error=is_err,
                    )
                )

    usage_obj = _block_attr(response, "usage", None)
    server_tool_usage = _block_attr(usage_obj, "server_tool_use", None)
    web_search_requests = int(
        _block_attr(server_tool_usage, "web_search_requests", 0) or 0
    )
    usage = TokenUsage(
        input_tokens=int(_block_attr(usage_obj, "input_tokens", 0) or 0),
        cache_read_tokens=int(_block_attr(usage_obj, "cache_read_input_tokens", 0) or 0),
        cache_creation_tokens=int(
            _block_attr(usage_obj, "cache_creation_input_tokens", 0) or 0
        ),
        output_tokens=int(_block_attr(usage_obj, "output_tokens", 0) or 0),
        web_search_requests=web_search_requests,
    )

    container_obj = _block_attr(response, "container", None)
    container_id = _block_attr(container_obj, "id", None) if container_obj is not None else None

    return ParsedLLMResponse(
        text="\n".join(text_parts).strip(),
        thinking=thinking_parts,
        tool_uses=tool_uses,
        server_web_searches=server_web_searches,
        server_tool_calls=server_tool_calls,
        raw_content=raw_content,
        stop_reason=_block_attr(response, "stop_reason", None),
        container_id=str(container_id) if container_id else None,
        usage=usage,
        model=str(_block_attr(response, "model", "") or ""),
    )


def build_haiku_request_kwargs(
    *,
    model: str,
    max_tokens: int,
    system: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    effort: EffortLevel,
    tools: list[dict[str, Any]] | None = None,
    json_schema: dict[str, Any] | None = None,
    container: str | None = None,
) -> dict[str, Any]:
    """Construct and validate API kwargs obeying every Claude Haiku 5.5 constraint.

    Enforces:
      - Never sends temperature, top_p, or top_k.
      - Always uses thinking={"type": "adaptive"} (no legacy budget_tokens).
      - Controls reasoning depth via output_config={"effort": ...} (and optional format).
      - Always requires messages to end with a `user` turn (no assistant prefill).
    """
    if not messages:
        raise ValueError("messages must be non-empty.")
    last_role = messages[-1].get("role")
    if last_role != "user":
        raise ValueError(
            f"Assistant prefill is forbidden on Claude Haiku 5.5: "
            f"messages must end with role='user' (got role={last_role!r})."
        )

    output_config: dict[str, Any] = {"effort": effort}
    if json_schema is not None:
        output_config["format"] = {
            "type": "json_schema",
            "schema": json_schema,
        }

    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": messages,
        "thinking": {"type": "adaptive"},
        "output_config": output_config,
    }
    if tools:
        kwargs["tools"] = tools
    if container:
        kwargs["container"] = container

    # Hard runtime assertion: forbidden sampling params must never appear
    forbidden_found = FORBIDDEN_SAMPLING_PARAMS.intersection(kwargs.keys())
    if forbidden_found:
        raise ValueError(
            f"Forbidden Haiku 5.5 parameters detected: {sorted(forbidden_found)}"
        )

    return kwargs


class HaikuClient:
    """AsyncAnthropic wrapper enforcing Claude Haiku 5.5 API constraints."""

    def __init__(
        self,
        settings: Settings,
        *,
        sdk_client: Any | None = None,
    ) -> None:
        self.settings = settings
        self._sdk_client = sdk_client

    def _get_client(self) -> Any:
        if self._sdk_client is not None:
            return self._sdk_client
        key = (self.settings.anthropic_api_key or "").strip()
        if not key or key == "sk-ant-replace-me":
            raise MissingAPIKeyError(
                "ANTHROPIC_API_KEY is not configured. Copy .env.example to .env and set a valid key."
            )
        self._sdk_client = anthropic.AsyncAnthropic(api_key=key)
        return self._sdk_client

    async def create_message(
        self,
        *,
        system: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        effort: EffortLevel,
        max_tokens: int,
        tools: list[dict[str, Any]] | None = None,
        json_schema: dict[str, Any] | None = None,
        container: str | None = None,
    ) -> ParsedLLMResponse:
        """Send a strictly compliant Claude Haiku 5.5 Messages API call."""
        kwargs = build_haiku_request_kwargs(
            model=self.settings.anthropic_model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
            effort=effort,
            tools=tools,
            json_schema=json_schema,
            container=container,
        )
        client = self._get_client()
        raw_response = await client.messages.create(**kwargs)
        return parse_anthropic_response(raw_response)

    async def close(self) -> None:
        if self._sdk_client is not None and hasattr(self._sdk_client, "close"):
            await self._sdk_client.close()
