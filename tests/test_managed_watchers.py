"""Unit & integration tests for managed background watchers inside AlwaysOnDaemon."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

from always_on.config import Settings
from always_on.daemon import AlwaysOnDaemon
from always_on.embeddings.provider import HashEmbeddingProvider
from always_on.watchers.claude_code import WatcherResult, WatcherState, check_and_notify_once
from always_on.watchers.vercel import (
    VercelPollResult,
    VercelWatcherState,
    poll_vercel_and_notify_once,
)
from tests.conftest import FakeBlock, FakeMessage, make_mock_client


@pytest.mark.asyncio
async def test_managed_watchers_disabled_by_default(
    test_settings: Settings,
    embedder: HashEmbeddingProvider,
) -> None:
    llm_client, _ = make_mock_client(test_settings, [])
    daemon = AlwaysOnDaemon(test_settings, embedder=embedder, llm_client=llm_client)
    await daemon.start(run_background_loop=True)
    try:
        assert daemon._watcher_tasks == []
        snap = await daemon.get_telemetry_snapshot()
        assert snap["watchers"] == {}
    finally:
        await daemon.stop()


@pytest.mark.asyncio
async def test_managed_watchers_start_deliver_and_stop_cleanly(
    monkeypatch: pytest.MonkeyPatch,
    test_settings: Settings,
    embedder: HashEmbeddingProvider,
) -> None:
    test_settings.enable_claude_code_watcher = True
    test_settings.claude_code_watcher_interval_seconds = 300
    test_settings.enable_vercel_watcher = True
    test_settings.vercel_watcher_interval_seconds = 300
    test_settings.loop_tick_seconds = 0.05

    cc_called = asyncio.Event()
    vercel_called = asyncio.Event()

    async def fake_cc_check(**kwargs: Any) -> WatcherResult:
        deliver_fn = kwargs["deliver_fn"]
        assert deliver_fn is not None
        res = await deliver_fn(
            {
                "content": "New Claude Code release detected: v2.1.335",
                "actor_id": "claude_code_watcher",
                "priority": "high",
                "hard_override": True,
                "payload": {"tag": "v2.1.335"},
            }
        )
        cc_called.set()
        return WatcherResult(
            status="notified",
            tag="v2.1.335",
            source="github_releases",
            event_id=res["event_id"],
            message="Delivered Claude Code v2.1.335.",
        )

    async def fake_vercel_poll(**kwargs: Any) -> VercelPollResult:
        deliver_fn = kwargs["deliver_fn"]
        assert deliver_fn is not None
        await deliver_fn(
            {
                "content": "Vercel deployment ERROR (deployment.error): web [production]",
                "actor_id": "vercel_poller",
                "priority": "high",
                "hard_override": True,
                "payload": {"deployment_id": "dpl_999", "state": "ERROR"},
            }
        )
        vercel_called.set()
        return VercelPollResult(
            status="notified",
            notified_count=1,
            deployments=["web [production] ERROR (dpl_999)"],
            message="Delivered 1 Vercel deployment.",
        )

    monkeypatch.setattr("always_on.daemon.check_and_notify_once", fake_cc_check)
    monkeypatch.setattr("always_on.daemon.poll_vercel_and_notify_once", fake_vercel_poll)

    llm_client, _ = make_mock_client(
        test_settings,
        [
            FakeMessage(content=[FakeBlock(type="text", text="Handled watcher event 1.")]),
            FakeMessage(content=[FakeBlock(type="text", text="Handled watcher event 2.")]),
        ],
    )

    daemon = AlwaysOnDaemon(test_settings, embedder=embedder, llm_client=llm_client)
    await daemon.start(run_background_loop=True)
    try:
        assert len(daemon._watcher_tasks) == 2
        await asyncio.wait_for(cc_called.wait(), timeout=2.0)
        await asyncio.wait_for(vercel_called.wait(), timeout=2.0)

        # Wait briefly for outer loop to ingest both webhook events into fast_buffer
        deadline = time.monotonic() + 2.5
        while time.monotonic() < deadline:
            if len(await daemon.store.list_fast_buffer()) >= 2:
                break
            await asyncio.sleep(0.05)

        episodes = await daemon.store.list_fast_buffer()
        assert len(episodes) == 2

        snap = await daemon.get_telemetry_snapshot()
        assert snap["watchers"]["claude_code"]["last_status"] == "notified"
        assert snap["watchers"]["claude_code"]["last_tag"] == "v2.1.335"
        assert snap["watchers"]["vercel"]["last_status"] == "notified"
        assert snap["watchers"]["vercel"]["notified_count"] == 1
    finally:
        t0 = time.monotonic()
        daemon.request_shutdown()
        await daemon.stop()
        elapsed = time.monotonic() - t0
        assert elapsed < 1.0
        assert daemon._watcher_tasks == []


@pytest.mark.asyncio
async def test_deliver_fn_bypass_on_claude_code_and_vercel_watchers(tmp_path: Path) -> None:
    # 1. Claude Code watcher with deliver_fn (no daemon_url HTTP call)
    cc_state = tmp_path / "cc_state.json"
    cc_delivered: list[dict[str, Any]] = []

    async def cc_deliver(body: dict[str, Any]) -> dict[str, Any]:
        cc_delivered.append(body)
        return {"accepted": True, "event_id": "evt_cc_1"}

    gh_transport = httpx.MockTransport(
        lambda req: httpx.Response(
            200,
            headers={"etag": '"e1"'},
            json={
                "tag_name": "v2.1.340",
                "published_at": "2026-11-22T15:00:00Z",
                "html_url": "https://github.com/anthropics/claude-code/releases/tag/v2.1.340",
                "body": "- Added feature X",
            },
        )
    )
    async with httpx.AsyncClient(transport=gh_transport) as gh_client:
        cc_res = await check_and_notify_once(
            daemon_url="http://unreachable.invalid:9999",
            state_path=cc_state,
            github_client=gh_client,
            deliver_fn=cc_deliver,
        )

    assert cc_res.status == "notified"
    assert cc_res.event_id == "evt_cc_1"
    assert len(cc_delivered) == 1
    assert cc_delivered[0]["actor_id"] == "claude_code_watcher"
    assert WatcherState.load(cc_state).last_seen_tag == "v2.1.340"

    # 2. Vercel watcher with deliver_fn (seed then notify, no daemon_url HTTP call)
    v_state = tmp_path / "v_state.json"
    v_delivered: list[dict[str, Any]] = []

    async def v_deliver(body: dict[str, Any]) -> dict[str, Any]:
        v_delivered.append(body)
        return {"accepted": True, "event_id": "evt_v_1"}

    deployments = [
        {
            "uid": "dpl_1",
            "name": "app",
            "projectId": "prj_1",
            "url": "app.vercel.app",
            "target": "production",
            "readyState": "READY",
            "createdAt": 1732280000000,
        }
    ]
    v_transport = httpx.MockTransport(
        lambda req: httpx.Response(200, json={"deployments": deployments})
    )
    async with httpx.AsyncClient(transport=v_transport) as v_client:
        seed = await poll_vercel_and_notify_once(
            api_token="tok_123",
            daemon_url="http://unreachable.invalid:9999",
            state_path=v_state,
            vercel_client=v_client,
            deliver_fn=v_deliver,
        )
        assert seed.status == "seeded"
        assert v_delivered == []

        deployments.insert(
            0,
            {
                "uid": "dpl_2",
                "name": "app",
                "projectId": "prj_1",
                "url": "app-2.vercel.app",
                "target": "production",
                "readyState": "READY",
                "createdAt": 1732285000000,
            },
        )
        res = await poll_vercel_and_notify_once(
            api_token="tok_123",
            daemon_url="http://unreachable.invalid:9999",
            state_path=v_state,
            vercel_client=v_client,
            deliver_fn=v_deliver,
        )

    assert res.status == "notified"
    assert res.notified_count == 1
    assert len(v_delivered) == 1
    assert v_delivered[0]["actor_id"] == "vercel_poller"
    assert VercelWatcherState.load(v_state).seen_deployments["dpl_2"] == "READY"
