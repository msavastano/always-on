"""Unit & integration tests for persistent SQLite chat history and 2-way Discord bridge."""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import httpx
import pytest

from always_on.channels.discord_bot import (
    chunk_discord_message,
    classify_proactive_alert,
    detect_bot_mention,
    is_in_quiet_hours,
    should_accept_discord_message,
    strip_bot_mention,
)
from always_on.config import Settings
from always_on.daemon import AlwaysOnDaemon
from always_on.embeddings.provider import HashEmbeddingProvider
from always_on.models import ChatReply, IncomingEvent, MemoryItem
from always_on.web.app import create_app
from tests.conftest import FakeBlock, FakeMessage, make_mock_client


@pytest.mark.asyncio
async def test_chat_history_persists_across_daemon_restarts_and_delete_endpoint(
    test_settings: Settings,
    embedder: HashEmbeddingProvider,
) -> None:
    llm_client1, _ = make_mock_client(
        test_settings,
        [
            FakeMessage(
                content=[
                    FakeBlock(type="thinking", thinking="Checking user's question carefully."),
                    FakeBlock(type="text", text="Persistent answer from daemon 1."),
                ]
            ),
            FakeMessage(
                content=[
                    FakeBlock(type="text", text="Proactive webhook summary persisted too."),
                ]
            ),
        ],
    )

    daemon1 = AlwaysOnDaemon(test_settings, embedder=embedder, llm_client=llm_client1)
    await daemon1.start(run_background_loop=False)
    try:
        await daemon1.store.insert_core_memory_item(
            MemoryItem(
                id="mem_keep",
                category="fact",
                content="Core memory must not be deleted when chat history clears.",
                source_episodes=["ep_seed"],
            )
        )
        r1 = await daemon1.send_chat(
            "Hello from Discord!",
            actor_id="discord:mike",
            source="discord",
        )
        assert r1.reply == "Persistent answer from daemon 1."

        await daemon1.push_webhook(
            content="Vercel deploy finished",
            actor_id="vercel_poller",
            priority="high",
            hard_override=True,
        )
        await daemon1.tick_once()
        assert len(daemon1.chat_history) == 3  # user + assistant + proactive webhook reply
    finally:
        await daemon1.stop()

    # Boot a brand-new daemon2 against the same SQLite DB -> chat_history MUST hydrate automatically
    llm_client2, _ = make_mock_client(test_settings, [])
    daemon2 = AlwaysOnDaemon(test_settings, embedder=embedder, llm_client=llm_client2)
    app2 = create_app(settings=test_settings, daemon=daemon2, run_background_loop=False)
    await daemon2.start(run_background_loop=False)
    try:
        snap = await daemon2.get_telemetry_snapshot()
        hist = snap["chat_history"]
        assert len(hist) == 3
        assert hist[0]["role"] == "user"
        assert hist[0]["source"] == "discord"
        assert hist[0]["actor_id"] == "discord:mike"
        assert hist[0]["content"] == "Hello from Discord!"

        assert hist[1]["role"] == "assistant"
        assert hist[1]["thoughts"] == ["Checking user's question carefully."]
        assert hist[1]["usd_cost"] > 0.0

        assert hist[2]["role"] == "assistant"
        assert hist[2]["source"] == "webhook"
        assert hist[2]["trigger_actor"] == "vercel_poller"

        # Verify DELETE /api/chat clears chat_messages without touching core_memory or fast_buffer
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app2), base_url="http://test"
        ) as client:
            del_res = await client.delete("/api/chat")
            assert del_res.status_code == 200
            assert del_res.json()["cleared"] == 3

        assert await daemon2.store.list_chat_messages() == []
        assert len(await daemon2.store.list_core_memory()) == 1
        assert len(await daemon2.store.list_fast_buffer()) == 2
    finally:
        await daemon2.stop()


def test_discord_allowlist_and_self_trigger_filtering() -> None:
    allowed = {111222333}
    passive = {777888999}
    notify_cid = 555666777
    bot_uid = 999000111

    # Bot / self messages always ignored
    assert (
        should_accept_discord_message(
            author_id=111222333,
            author_is_bot=True,
            bot_user_id=bot_uid,
            is_dm=True,
            channel_id=1,
            allowed_user_ids=allowed,
            passive_channel_ids=passive,
            mentioned_bot=True,
            notify_channel_id=notify_cid,
        )
        == "ignore"
    )
    assert (
        should_accept_discord_message(
            author_id=bot_uid,
            author_is_bot=False,
            bot_user_id=bot_uid,
            is_dm=True,
            channel_id=1,
            allowed_user_ids={bot_uid},
            passive_channel_ids=passive,
            mentioned_bot=False,
            notify_channel_id=notify_cid,
        )
        == "ignore"
    )

    # Empty allowlist fails closed
    assert (
        should_accept_discord_message(
            author_id=111222333,
            author_is_bot=False,
            bot_user_id=bot_uid,
            is_dm=True,
            channel_id=1,
            allowed_user_ids=set(),
            passive_channel_ids=passive,
            mentioned_bot=True,
            notify_channel_id=notify_cid,
        )
        == "ignore"
    )

    # Unauthorized user ignored everywhere
    assert (
        should_accept_discord_message(
            author_id=444555666,
            author_is_bot=False,
            bot_user_id=bot_uid,
            is_dm=True,
            channel_id=notify_cid,
            allowed_user_ids=allowed,
            passive_channel_ids=passive,
            mentioned_bot=True,
            notify_channel_id=notify_cid,
        )
        == "ignore"
    )

    # Authorized user: DM, notify channel, and @mention -> direct
    assert (
        should_accept_discord_message(
            author_id=111222333,
            author_is_bot=False,
            bot_user_id=bot_uid,
            is_dm=True,
            channel_id=99,
            allowed_user_ids=allowed,
            passive_channel_ids=passive,
            mentioned_bot=False,
            notify_channel_id=notify_cid,
        )
        == "direct"
    )
    assert (
        should_accept_discord_message(
            author_id=111222333,
            author_is_bot=False,
            bot_user_id=bot_uid,
            is_dm=False,
            channel_id=notify_cid,
            allowed_user_ids=allowed,
            passive_channel_ids=passive,
            mentioned_bot=False,
            notify_channel_id=notify_cid,
        )
        == "direct"
    )

    # Passive channel without @mention -> passive; with @mention -> direct
    assert (
        should_accept_discord_message(
            author_id=111222333,
            author_is_bot=False,
            bot_user_id=bot_uid,
            is_dm=False,
            channel_id=777888999,
            allowed_user_ids=allowed,
            passive_channel_ids=passive,
            mentioned_bot=False,
            notify_channel_id=notify_cid,
        )
        == "passive"
    )
    assert (
        should_accept_discord_message(
            author_id=111222333,
            author_is_bot=False,
            bot_user_id=bot_uid,
            is_dm=False,
            channel_id=777888999,
            allowed_user_ids=allowed,
            passive_channel_ids=passive,
            mentioned_bot=True,
            notify_channel_id=notify_cid,
        )
        == "direct"
    )
    assert strip_bot_mention("<@!999000111> check status", bot_uid) == "check status"


def test_detect_and_strip_bot_user_and_managed_role_mentions() -> None:
    bot_uid = 999000111
    managed_role_id = 888777666
    other_role_id = 111000999

    msg_user_mention = SimpleNamespace(
        mentions=[SimpleNamespace(id=bot_uid)],
        raw_mentions=[],
        role_mentions=[],
        content=f"<@{bot_uid}> status",
    )
    assert detect_bot_mention(msg_user_mention, bot_uid) == (True, set())

    msg_raw_mention = SimpleNamespace(
        mentions=[],
        raw_mentions=[bot_uid],
        role_mentions=[],
        content="status",
    )
    assert detect_bot_mention(msg_raw_mention, bot_uid) == (True, set())

    msg_nick_text = SimpleNamespace(
        mentions=[],
        raw_mentions=[],
        role_mentions=[],
        content=f"<@!{bot_uid}> ping",
    )
    assert detect_bot_mention(msg_nick_text, bot_uid) == (True, set())

    msg_managed_role = SimpleNamespace(
        mentions=[],
        raw_mentions=[],
        role_mentions=[
            SimpleNamespace(id=managed_role_id, tags=SimpleNamespace(bot_id=bot_uid))
        ],
        content=f"<@&{managed_role_id}> summarize this",
    )
    mentioned, matched_roles = detect_bot_mention(msg_managed_role, bot_uid)
    assert mentioned is True
    assert matched_roles == {managed_role_id}
    assert (
        strip_bot_mention(msg_managed_role.content, bot_uid, matched_roles)
        == "summarize this"
    )

    msg_unrelated_role = SimpleNamespace(
        mentions=[],
        raw_mentions=[],
        role_mentions=[
            SimpleNamespace(id=other_role_id, tags=None),
            SimpleNamespace(id=other_role_id + 1, tags=SimpleNamespace(bot_id=123456)),
        ],
        content=f"<@&{other_role_id}> hello",
    )
    assert detect_bot_mention(msg_unrelated_role, bot_uid) == (False, set())

    msg_plain_text = SimpleNamespace(
        mentions=[],
        raw_mentions=[],
        role_mentions=[],
        content="@always-on (app) hello",
    )
    assert detect_bot_mention(msg_plain_text, bot_uid) == (False, set())


def test_discord_markdown_chunking_preserves_code_fences() -> None:
    lines = ["Intro paragraph.\n", "\n", "```python\n"]
    lines.extend(f"print('line number {i:03d} with extra padding text')\n" for i in range(60))
    lines.append("```\n\nClosing note.")
    text = "".join(lines)

    chunks = chunk_discord_message(text, limit=600)
    assert len(chunks) >= 3
    for chunk in chunks:
        assert len(chunk) <= 600
        # Every chunk with code must have balanced ``` fences!
        assert chunk.count("```") % 2 == 0
    assert "Closing note." in chunks[-1]


def test_discord_quiet_hours_overnight_and_same_day() -> None:
    assert is_in_quiet_hours("22:00-08:00", now_local=datetime(2026, 11, 22, 23, 30)) is True
    assert is_in_quiet_hours("22:00-08:00", now_local=datetime(2026, 11, 22, 3, 15)) is True
    assert is_in_quiet_hours("22:00-08:00", now_local=datetime(2026, 11, 22, 12, 0)) is False

    assert is_in_quiet_hours("13:00-14:30", now_local=datetime(2026, 11, 22, 13, 45)) is True
    assert is_in_quiet_hours("13:00-14:30", now_local=datetime(2026, 11, 22, 15, 0)) is False
    assert is_in_quiet_hours("", now_local=datetime(2026, 11, 22, 23, 30)) is False


@pytest.mark.asyncio
async def test_proactive_listener_invoked_on_webhook_wake_and_alert_classification(
    test_settings: Settings,
    embedder: HashEmbeddingProvider,
) -> None:
    llm_client, _ = make_mock_client(
        test_settings,
        [
            FakeMessage(content=[FakeBlock(type="text", text="Build failed on missing module.")]),
            FakeMessage(content=[FakeBlock(type="text", text="Claude Code v2.1.335 released.")]),
        ],
    )
    daemon = AlwaysOnDaemon(test_settings, embedder=embedder, llm_client=llm_client)

    received: list[tuple[str, int, bool, str]] = []

    async def healthy_listener(event: IncomingEvent, reply: ChatReply) -> None:
        title, color, is_crit = classify_proactive_alert(event, reply)
        received.append((title, color, is_crit, reply.reply))

    async def failing_listener(event: IncomingEvent, reply: ChatReply) -> None:
        raise RuntimeError("Simulated Discord network glitch")

    # Failing listener registered first to verify it never stops healthy_listener or wake persistence
    daemon.register_proactive_listener(failing_listener)
    daemon.register_proactive_listener(healthy_listener)

    await daemon.start(run_background_loop=False)
    try:
        await daemon.push_webhook(
            content="Vercel ERROR",
            actor_id="vercel_poller",
            priority="high",
            hard_override=True,
            payload={
                "provider": "vercel",
                "project_name": "my-app",
                "target": "production",
                "state": "ERROR",
            },
        )
        await daemon.push_webhook(
            content="New Claude Code release v2.1.335",
            actor_id="claude_code_watcher",
            priority="high",
            hard_override=True,
            payload={"tag": "v2.1.335"},
        )
        await daemon.tick_once()

        assert len(received) == 2
        assert received[0] == (
            "Vercel Build Failed: my-app [production]",
            0xE53935,
            True,
            "Build failed on missing module.",
        )
        assert received[1] == (
            "Claude Code Release v2.1.335",
            0x7C4DFF,
            False,
            "Claude Code v2.1.335 released.",
        )
        assert len(await daemon.store.list_fast_buffer()) == 2
    finally:
        await daemon.stop()
