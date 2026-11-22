"""Integration tests for FastAPI endpoints, dashboard serving, and FileWatcherSource."""

from __future__ import annotations

import json

import httpx
import pytest

from always_on.config import Settings
from always_on.daemon import AlwaysOnDaemon
from always_on.embeddings.provider import HashEmbeddingProvider
from always_on.sources.file_watcher import FileWatcherSource
from always_on.web.app import create_app
from tests.conftest import FakeBlock, FakeMessage, make_mock_client


@pytest.mark.asyncio
async def test_file_watcher_detects_new_and_modified_files(test_settings: Settings) -> None:
    watcher = FileWatcherSource(test_settings.inbox_dir)
    await watcher.start()
    assert await watcher.poll() == []

    sample = test_settings.inbox_dir / "notes.txt"
    sample.write_text("Project deadline moved to Friday.", encoding="utf-8")

    created_events = await watcher.poll()
    assert len(created_events) == 1
    assert created_events[0].source == "file_watcher"
    assert created_events[0].actor_id == "filesystem"
    assert created_events[0].payload["change"] == "created"
    assert "Friday" in created_events[0].content

    # Polling again without change emits nothing
    assert await watcher.poll() == []

    # Modifying the file emits a modified event
    sample.write_text("Project deadline moved to Monday.", encoding="utf-8")
    mod_events = await watcher.poll()
    assert len(mod_events) == 1
    assert mod_events[0].payload["change"] == "modified"
    assert "Monday" in mod_events[0].content


@pytest.mark.asyncio
async def test_fastapi_chat_webhook_sleep_and_pin_endpoints(
    test_settings: Settings,
    embedder: HashEmbeddingProvider,
) -> None:
    sleep_plan = {
        "operations": [
            {
                "op": "ADD",
                "target_id": None,
                "category": "preference",
                "content": "User prefers concise bullet summaries.",
                "confidence": 0.9,
                "pinned": False,
                "source_episode_ids": [],  # Will be patched dynamically below
                "reason": "Direct user preference.",
            }
        ],
        "active_priorities": [
            "Deliver concise bullet summaries.",
            "Monitor inbox notes.",
            "Track project deadlines.",
        ],
        "reflection": "Stored summary preference.",
    }

    # Build a small dynamic mock so the sleep call references the real created episode ID
    class DynamicSDK:
        def __init__(self) -> None:
            self.turn = 0
            self.messages = self

        async def create(self, **kwargs):  # type: ignore[no-untyped-def]
            self.turn += 1
            if self.turn == 1:
                return FakeMessage(
                    content=[FakeBlock(type="text", text="Got it—I'll keep summaries concise!")]
                )
            # Extract valid episode ID from consolidation user prompt
            prompt_text = kwargs["messages"][0]["content"]
            marker = "Valid source_episode_ids: ['"
            start = prompt_text.index(marker) + len(marker)
            ep_id = prompt_text[start : prompt_text.index("'", start)]
            plan = json.loads(json.dumps(sleep_plan))
            plan["operations"][0]["source_episode_ids"] = [ep_id]
            return FakeMessage(content=[FakeBlock(type="text", text=json.dumps(plan))])

        async def close(self) -> None:
            return None

    from always_on.llm.client import HaikuClient

    llm_client = HaikuClient(test_settings, sdk_client=DynamicSDK())
    daemon = AlwaysOnDaemon(test_settings, embedder=embedder, llm_client=llm_client)
    app = create_app(settings=test_settings, daemon=daemon, run_background_loop=False)

    await daemon.start(run_background_loop=False)
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            # 1. Dashboard HTML & health
            home = await client.get("/")
            assert home.status_code == 200
            assert "Always-On CLS Agent" in home.text

            health = await client.get("/api/health")
            assert health.status_code == 200
            assert health.json()["model"] == "claude-haiku-5-5"

            # 2. Direct chat wake-up
            chat_res = await client.post(
                "/api/chat", json={"message": "Please keep summaries concise."}
            )
            assert chat_res.status_code == 200
            assert chat_res.json()["actor_id"] == "agent"

            # 3. Webhook self-trigger prevention via API
            wh_res = await client.post(
                "/api/events/webhook?process_now=true",
                json={"content": "Self echo", "actor_id": "agent"},
            )
            assert wh_res.status_code == 200
            telem = (await client.get("/api/telemetry")).json()
            assert telem["ingestor"]["dropped_self_triggers"] == 1
            assert len(telem["fast_buffer"]) == 1

            # 4. Manual sleep consolidation via API
            sleep_res = await client.post("/api/sleep")
            assert sleep_res.status_code == 200
            report = sleep_res.json()
            assert report["status"] == "committed"
            assert len(report["added_ids"]) == 1
            mem_id = report["added_ids"][0]

            # 5. Pin toggle via API
            pin_res = await client.post(f"/api/memory/{mem_id}/pin", json={"pinned": True})
            assert pin_res.status_code == 200
            assert pin_res.json()["pinned"] is True

            # Verify buffer flushed and fatigue reset to 0.0
            telem_after = (await client.get("/api/telemetry")).json()
            assert telem_after["fast_buffer"] == []
            assert telem_after["fatigue"]["score"] == 0.0
    finally:
        await daemon.stop()
