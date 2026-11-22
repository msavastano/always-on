"""Unit & integration tests for the Claude Code release watcher and webhook auth."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from always_on.config import Settings
from always_on.daemon import AlwaysOnDaemon
from always_on.embeddings.provider import HashEmbeddingProvider
from always_on.watchers.claude_code import (
    MAX_BODY_CHARS,
    ReleaseInfo,
    WatcherState,
    build_webhook_payload,
    check_and_notify_once,
    fetch_latest_claude_code_release,
    parse_latest_changelog_section,
)
from always_on.web.app import create_app
from tests.conftest import FakeBlock, FakeMessage, make_mock_client


def test_parse_latest_changelog_section_extracts_top_version() -> None:
    md = """# Changelog

## 2.1.330

- Added enterprise sign-in
- Fixed Windows Git Bash latency

## 2.1.329

- Older fix
"""
    rel = parse_latest_changelog_section(md)
    assert rel is not None
    assert rel.tag == "v2.1.330"
    assert rel.source == "github_changelog"
    assert "Windows Git Bash latency" in rel.body
    assert "Older fix" not in rel.body


def test_build_webhook_payload_truncates_large_release_notes() -> None:
    rel = ReleaseInfo(
        tag="v2.1.330",
        published_at="2026-11-20T22:32:56Z",
        url="https://github.com/anthropics/claude-code/releases/tag/v2.1.330",
        body="X" * (MAX_BODY_CHARS + 1000),
    )
    payload = build_webhook_payload(rel, priority="high")
    assert payload["actor_id"] == "claude_code_watcher"
    assert payload["priority"] == "high"
    assert payload["hard_override"] is True
    assert payload["payload"]["tag"] == "v2.1.330"
    assert "...[truncated]..." in payload["content"]
    assert len(payload["content"]) < MAX_BODY_CHARS + 600


@pytest.mark.asyncio
async def test_fetch_handles_etag_304_and_changelog_fallback_on_rate_limit() -> None:
    # 1. HTTP 304 Not Modified returns not_modified=True cleanly
    transport_304 = httpx.MockTransport(
        lambda req: httpx.Response(304, headers={"etag": '"abc123"'})
        if req.headers.get("if-none-match") == '"abc123"'
        else httpx.Response(500)
    )
    async with httpx.AsyncClient(transport=transport_304) as client:
        rel, new_etag, not_mod = await fetch_latest_claude_code_release(
            etag='"abc123"', http_client=client
        )
    assert rel is None
    assert new_etag == '"abc123"'
    assert not_mod is True

    # 2. HTTP 403 rate limit falls back cleanly to raw CHANGELOG.md
    def rate_limit_handler(req: httpx.Request) -> httpx.Response:
        if "api.github.com" in str(req.url):
            return httpx.Response(403, json={"message": "API rate limit exceeded"})
        return httpx.Response(
            200,
            text="# Changelog\n\n## 2.1.331\n\n- Fallback changelog item\n\n## 2.1.330\n\n- Old",
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(rate_limit_handler)) as client:
        rel2, _, not_mod2 = await fetch_latest_claude_code_release(http_client=client)
    assert not_mod2 is False
    assert rel2 is not None
    assert rel2.tag == "v2.1.331"
    assert rel2.source == "github_changelog"
    assert "Fallback changelog item" in rel2.body


@pytest.mark.asyncio
async def test_watcher_end_to_end_webhook_auth_dedup_and_chat_surfacing(
    tmp_path: Path,
    test_settings: Settings,
    embedder: HashEmbeddingProvider,
) -> None:
    test_settings.webhook_secret = "super-secret-token"
    state_file = tmp_path / "watcher_state.json"

    github_calls = 0

    def github_handler(req: httpx.Request) -> httpx.Response:
        nonlocal github_calls
        github_calls += 1
        if req.headers.get("if-none-match") == '"etag-330"':
            return httpx.Response(304, headers={"etag": '"etag-330"'})
        return httpx.Response(
            200,
            headers={"etag": '"etag-330"'},
            json={
                "tag_name": "v2.1.330",
                "published_at": "2026-11-20T22:32:56Z",
                "html_url": "https://github.com/anthropics/claude-code/releases/tag/v2.1.330",
                "body": "- Fixed Windows Git Bash startup latency\n- Added easel charts",
            },
        )

    llm_client, _ = make_mock_client(
        test_settings,
        [
            FakeMessage(
                content=[
                    FakeBlock(
                        type="text",
                        text="Claude Code v2.1.330 fixes Windows Git Bash startup latency and adds easel charts.",
                    )
                ]
            )
        ],
    )

    daemon = AlwaysOnDaemon(test_settings, embedder=embedder, llm_client=llm_client)
    app = create_app(settings=test_settings, daemon=daemon, run_background_loop=False)
    await daemon.start(run_background_loop=False)

    try:
        async with (
            httpx.AsyncClient(transport=httpx.MockTransport(github_handler)) as gh_client,
            httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as d_client,
        ):
            # 1. Wrong/missing webhook secret -> rejected with 401 and state NOT advanced!
            bad_res = await check_and_notify_once(
                daemon_url="http://test",
                state_path=state_file,
                webhook_secret="wrong-secret",
                process_now=True,
                github_client=gh_client,
                daemon_client=d_client,
            )
            assert bad_res.status == "error"
            assert "401" in bad_res.message
            assert WatcherState.load(state_file).last_seen_tag is None

            # 2. Valid webhook secret -> delivers release, wakes daemon, surfaces summary in dashboard chat feed!
            ok_res = await check_and_notify_once(
                daemon_url="http://test",
                state_path=state_file,
                webhook_secret="super-secret-token",
                process_now=True,
                github_client=gh_client,
                daemon_client=d_client,
            )
            assert ok_res.status == "notified"
            assert ok_res.tag == "v2.1.330"
            assert ok_res.event_id

            saved = WatcherState.load(state_file)
            assert saved.last_seen_tag == "v2.1.330"
            assert saved.etag == '"etag-330"'

            # Verify daemon woke up and surfaced the summary in chat_history & fast_buffer
            telem = (await d_client.get("/api/telemetry")).json()
            assert len(telem["fast_buffer"]) == 1
            assert len(telem["chat_history"]) == 1
            assert telem["chat_history"][0]["source"] == "webhook"
            assert telem["chat_history"][0]["trigger_actor"] == "claude_code_watcher"
            assert "v2.1.330" in telem["chat_history"][0]["content"]

            # 3. Next poll hits HTTP 304 Not Modified -> zero duplicate webhooks sent!
            second_res = await check_and_notify_once(
                daemon_url="http://test",
                state_path=state_file,
                webhook_secret="super-secret-token",
                process_now=True,
                github_client=gh_client,
                daemon_client=d_client,
            )
            assert second_res.status == "not_modified"
            assert second_res.tag == "v2.1.330"
    finally:
        await daemon.stop()
