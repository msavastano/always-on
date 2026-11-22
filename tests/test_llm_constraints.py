"""Unit tests verifying strict Claude Haiku 5.5 API constraints and daemon wake handling."""

from __future__ import annotations

import pytest

from always_on.config import Settings
from always_on.daemon import AlwaysOnDaemon
from always_on.embeddings.provider import HashEmbeddingProvider
from always_on.llm.client import (
    FORBIDDEN_SAMPLING_PARAMS,
    build_haiku_request_kwargs,
    parse_anthropic_response,
)
from always_on.llm.prompts import (
    MIN_CACHE_PREFIX_TOKENS,
    build_cached_system_blocks,
    estimate_token_count,
)
from always_on.models import consolidation_json_schema
from tests.conftest import FakeBlock, FakeMessage, make_mock_client


def test_consolidation_schema_uses_anthropic_compatible_nullable_anyof() -> None:
    import anthropic

    schema = consolidation_json_schema()
    op_props = schema["properties"]["operations"]["items"]["properties"]

    # Nullable enum must use anyOf rather than mixing type=['string', 'null'] with enum
    assert "enum" not in op_props["category"]
    assert op_props["category"] == {
        "anyOf": [
            {"type": "string", "enum": ["fact", "preference", "project", "lesson"]},
            {"type": "null"},
        ]
    }
    assert op_props["target_id"] == {"anyOf": [{"type": "string"}, {"type": "null"}]}
    assert op_props["content"] == {"anyOf": [{"type": "string"}, {"type": "null"}]}

    # Verify Anthropic SDK's official schema transformer accepts the schema cleanly
    transformed = anthropic.transform_schema(schema)
    assert transformed["type"] == "object"
    assert transformed["additionalProperties"] is False


def test_haiku_request_kwargs_enforce_all_constraints() -> None:
    system = build_cached_system_blocks(
        agent_name="Aion",
        core_memory=[],
        active_priorities=[],
    )
    kwargs = build_haiku_request_kwargs(
        model="claude-haiku-5-5",
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": "Hello"}],
        effort="low",
    )

    assert kwargs["model"] == "claude-haiku-5-5"
    assert kwargs["thinking"] == {"type": "adaptive"}
    assert kwargs["output_config"] == {"effort": "low"}
    for forbidden in FORBIDDEN_SAMPLING_PARAMS:
        assert forbidden not in kwargs

    # Verify prompt caching breakpoint and >= 512 token prefix even with empty Core Memory
    assert system[-1]["cache_control"] == {"type": "ephemeral"}
    total_prefix_tokens = sum(estimate_token_count(b["text"]) for b in system)
    assert total_prefix_tokens >= MIN_CACHE_PREFIX_TOKENS


def test_assistant_prefill_is_strictly_rejected() -> None:
    system = [{"type": "text", "text": "sys"}]
    with pytest.raises(ValueError, match="Assistant prefill is forbidden"):
        build_haiku_request_kwargs(
            model="claude-haiku-5-5",
            max_tokens=512,
            system=system,
            messages=[
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Sure, here is"},
            ],
            effort="low",
        )


def test_response_parser_selects_blocks_by_type_not_index() -> None:
    msg = FakeMessage(
        content=[
            FakeBlock(type="thinking", thinking="Internal reasoning step", signature="sig_1"),
            FakeBlock(type="tool_use", id="tu_1", name="get_current_time", input={}),
            FakeBlock(type="text", text="Checking the clock now."),
        ],
        stop_reason="tool_use",
    )
    parsed = parse_anthropic_response(msg)
    assert parsed.thinking == ["Internal reasoning step"]
    assert parsed.text == "Checking the clock now."
    assert len(parsed.tool_uses) == 1
    assert parsed.tool_uses[0].name == "get_current_time"
    assert parsed.stop_reason == "tool_use"


@pytest.mark.asyncio
async def test_daemon_handles_tool_loop_refusal_and_context_exceeded(
    test_settings: Settings,
    embedder: HashEmbeddingProvider,
) -> None:
    llm_client, sdk = make_mock_client(
        test_settings,
        [
            # Turn 1: thinking + tool_use -> then final answer
            FakeMessage(
                content=[
                    FakeBlock(type="thinking", thinking="Let me check the current time first."),
                    FakeBlock(type="tool_use", id="tu_time", name="get_current_time", input={}),
                ],
                stop_reason="tool_use",
            ),
            FakeMessage(
                content=[
                    FakeBlock(type="thinking", thinking="Got the UTC timestamp cleanly."),
                    FakeBlock(type="text", text="Here is the current time from the local clock."),
                ],
                stop_reason="end_turn",
            ),
            # Turn 2: model refusal stop_reason
            FakeMessage(
                content=[FakeBlock(type="text", text="I cannot assist with that request.")],
                stop_reason="refusal",
            ),
            # Turn 3: context window exceeded -> daemon retries compact prompt once -> succeeds
            FakeMessage(
                content=[],
                stop_reason="model_context_window_exceeded",
            ),
            FakeMessage(
                content=[FakeBlock(type="text", text="Recovered cleanly with compact context.")],
                stop_reason="end_turn",
            ),
        ],
    )

    daemon = AlwaysOnDaemon(test_settings, embedder=embedder, llm_client=llm_client)
    await daemon.start(run_background_loop=False)
    try:
        # 1. Multi-step tool call wake-up
        r1 = await daemon.send_chat("What time is it?")
        assert r1.actor_id == "agent"
        assert r1.status == "ok"
        assert len(r1.tool_calls) == 1
        assert r1.tool_calls[0].name == "get_current_time"
        assert r1.tool_calls[0].is_error is False
        assert "local clock" in r1.reply

        # Every API call must end with a user turn and obey Haiku 5.5 constraints
        for call in sdk.messages.calls:
            assert call["messages"][-1]["role"] == "user"
            assert call["thinking"] == {"type": "adaptive"}
            assert call["output_config"]["effort"] == "low"
            for forbidden in FORBIDDEN_SAMPLING_PARAMS:
                assert forbidden not in call

        # 2. Refusal stop_reason handled cleanly without tripping backoff errors
        r2 = await daemon.send_chat("Do something disallowed")
        assert r2.status == "refusal"
        snap = await daemon.circuit_breaker.snapshot()
        assert snap.consecutive_errors == 0

        # 3. model_context_window_exceeded recovers cleanly via compact retry
        r3 = await daemon.send_chat("Summarize my day")
        assert r3.status == "ok"
        assert "Recovered cleanly" in r3.reply
    finally:
        await daemon.stop()


@pytest.mark.asyncio
async def test_anthropic_server_code_execution_gating_parsing_and_pause_turn(
    test_settings: Settings,
    embedder: HashEmbeddingProvider,
) -> None:
    from dataclasses import dataclass

    @dataclass
    class FakeContainer:
        id: str

    @dataclass
    class FakeExecResultContent:
        type: str = "bash_code_execution_result"
        stdout: str = "120\n"
        stderr: str = ""
        return_code: int = 0

    @dataclass
    class FakeServerResultBlock:
        type: str
        tool_use_id: str
        content: FakeExecResultContent

    @dataclass
    class MessageWithContainer(FakeMessage):
        container: FakeContainer | None = None

    test_settings.enable_code_execution = True
    test_settings.allow_code_execution_on_background = False

    llm_client, sdk = make_mock_client(
        test_settings,
        [
            # Turn 1: server-side code execution + pause_turn with container ID
            MessageWithContainer(
                content=[
                    FakeBlock(type="thinking", thinking="Let me compute 5! in Python."),
                    FakeBlock(
                        type="server_tool_use",
                        id="srv_ce_1",
                        name="bash_code_execution",
                        input={"command": "python3 -c 'import math; print(math.factorial(5))'"},
                    ),
                    FakeServerResultBlock(
                        type="bash_code_execution_tool_result",
                        tool_use_id="srv_ce_1",
                        content=FakeExecResultContent(stdout="120\n", stderr="", return_code=0),
                    ),
                ],
                stop_reason="pause_turn",
                container=FakeContainer(id="cntr_abc123"),
            ),
            # Turn 2: continuation reuses container_id and finishes cleanly
            MessageWithContainer(
                content=[
                    FakeBlock(type="text", text="The factorial of 5 is **120**."),
                ],
                stop_reason="end_turn",
                container=FakeContainer(id="cntr_abc123"),
            ),
        ],
    )

    daemon = AlwaysOnDaemon(test_settings, embedder=embedder, llm_client=llm_client)
    await daemon.start(run_background_loop=False)
    try:
        # Source gating: chat has code_execution; webhook & heartbeat do NOT by default
        chat_tools = daemon.tools.tool_definitions(event_source="chat")
        wh_tools = daemon.tools.tool_definitions(event_source="webhook")
        hb_tools = daemon.tools.tool_definitions(event_source="heartbeat")
        assert any(t.get("name") == "code_execution" for t in chat_tools)
        assert not any(t.get("name") == "code_execution" for t in wh_tools)
        assert not any(t.get("name") == "code_execution" for t in hb_tools)

        # Execute chat wake-up with pause_turn continuation
        reply = await daemon.send_chat("Compute 5 factorial with Python")
        assert reply.status == "ok"
        assert "120" in reply.reply
        assert len(reply.tool_calls) == 1
        assert reply.tool_calls[0].name == "bash_code_execution (anthropic)"
        assert "120" in reply.tool_calls[0].result
        assert reply.tool_calls[0].is_error is False

        # Verify two calls made: 2nd call reused container ID AND ended on a user turn (no assistant prefill)
        assert len(sdk.messages.calls) == 2
        assert sdk.messages.calls[0].get("container") is None
        assert sdk.messages.calls[1].get("container") == "cntr_abc123"
        assert sdk.messages.calls[1]["messages"][-1]["role"] == "user"
    finally:
        await daemon.stop()
