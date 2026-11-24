"""Integration tests for FastAPI endpoints, dashboard serving, and FileWatcherSource."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from always_on.config import Settings
from always_on.daemon import AlwaysOnDaemon
from always_on.db.store import AgentStore
from always_on.embeddings.provider import HashEmbeddingProvider
from always_on.sources.file_watcher import FileWatcherSource
from always_on.tools.registry import ToolRegistry
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


@pytest.mark.asyncio
async def test_notes_and_inbox_tools_work_with_relative_dirs_and_block_traversal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    test_settings: Settings,
) -> None:
    monkeypatch.chdir(tmp_path)
    store = AgentStore(test_settings.db_path)
    await store.connect()
    try:
        registry = ToolRegistry(
            store=store,
            inbox_dir=Path("data/inbox"),
            notes_dir=Path("data/notes"),
            settings=test_settings,
        )

        res1_txt, is_err1 = await registry.execute(
            "write_note",
            {"filename": "volleyball-practices.md", "content": "Wed 8pm"},
        )
        assert is_err1 is False
        res1 = json.loads(res1_txt)
        assert res1["status"] == "written"
        assert res1["filename"] == "volleyball-practices.md"

        res2_txt, is_err2 = await registry.execute(
            "write_note",
            {"filename": "family/nina.md", "content": "Practice"},
        )
        assert is_err2 is False
        res2 = json.loads(res2_txt)
        assert res2["status"] == "written"
        assert res2["filename"] == "family/nina.md"

        notes_txt, list_err = await registry.execute("list_notes", {})
        assert list_err is False
        assert json.loads(notes_txt)["notes"] == [
            "family/nina.md",
            "volleyball-practices.md",
        ]

        read1_txt, read1_err = await registry.execute(
            "read_note", {"filename": "volleyball-practices.md"}
        )
        assert read1_err is False
        assert read1_txt == "Wed 8pm"

        read2_txt, read2_err = await registry.execute(
            "read_note", {"filename": "family/nina.md"}
        )
        assert read2_err is False
        assert read2_txt == "Practice"

        (tmp_path / "data" / "inbox" / "schedule.txt").write_text(
            "Gym at 6pm", encoding="utf-8"
        )
        inbox_list_txt, inbox_list_err = await registry.execute(
            "list_inbox_files", {}
        )
        assert inbox_list_err is False
        assert [f["filename"] for f in json.loads(inbox_list_txt)["files"]] == [
            "schedule.txt"
        ]
        inbox_read_txt, inbox_read_err = await registry.execute(
            "read_inbox_file", {"filename": "schedule.txt"}
        )
        assert inbox_read_err is False
        assert inbox_read_txt == "Gym at 6pm"

        esc_txt, esc_err = await registry.execute(
            "write_note", {"filename": "../escape.md", "content": "nope"}
        )
        assert esc_err is True
        assert "Path traversal blocked" in esc_txt
        assert not (tmp_path / "data" / "escape.md").exists()
    finally:
        await store.close()
