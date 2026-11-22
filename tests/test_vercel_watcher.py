"""Unit & integration tests for Vercel push webhook (HMAC-SHA1) and zero-tunnel API poller."""

from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path

import httpx
import pytest

from always_on.config import Settings
from always_on.daemon import AlwaysOnDaemon
from always_on.embeddings.provider import HashEmbeddingProvider
from always_on.watchers.vercel import (
    VercelDeploymentEvent,
    VercelWatcherState,
    classify_vercel_priority,
    parse_vercel_webhook_envelope,
    poll_vercel_and_notify_once,
    verify_vercel_signature,
)
from always_on.web.app import create_app
from tests.conftest import FakeBlock, FakeMessage, make_mock_client


def _sign_vercel(raw_body: bytes, secret: str) -> str:
    return hmac.new(secret.encode("utf-8"), raw_body, hashlib.sha1).hexdigest()


def test_verify_vercel_hmac_sha1_signature() -> None:
    body = b'{"id":"evt_1","type":"deployment.succeeded"}'
    secret = "whsec_vercel_test_123"
    sig = _sign_vercel(body, secret)

    assert verify_vercel_signature(body, sig, secret) is True
    assert verify_vercel_signature(body, sig.upper(), secret) is True
    assert verify_vercel_signature(body + b" ", sig, secret) is False
    assert verify_vercel_signature(body, "deadbeef", secret) is False
    assert verify_vercel_signature(body, None, secret) is False


def test_smart_priority_policy_classifications() -> None:
    prod_ok = VercelDeploymentEvent(
        deployment_id="dpl_1", project_name="web", target="production", state="READY", event_type="deployment.succeeded"
    )
    preview_ok = VercelDeploymentEvent(
        deployment_id="dpl_2", project_name="web", target="preview", state="READY", event_type="deployment.succeeded"
    )
    preview_err = VercelDeploymentEvent(
        deployment_id="dpl_3", project_name="web", target="preview", state="ERROR", event_type="deployment.error"
    )
    prod_created = VercelDeploymentEvent(
        deployment_id="dpl_4", project_name="web", target="production", state="BUILDING", event_type="deployment.created"
    )

    # Smart policy: failures & prod READY = high; preview READY & created = normal (salience-gated)
    assert classify_vercel_priority(prod_ok, "smart") == ("high", True)
    assert classify_vercel_priority(preview_ok, "smart") == ("normal", False)
    assert classify_vercel_priority(preview_err, "smart") == ("high", True)
    assert classify_vercel_priority(prod_created, "smart") == ("normal", False)

    # Failures only
    assert classify_vercel_priority(prod_ok, "failures_only") == ("normal", False)
    assert classify_vercel_priority(preview_err, "failures_only") == ("high", True)

    # All high
    assert classify_vercel_priority(preview_ok, "all_high") == ("high", True)


def test_parse_vercel_webhook_envelope() -> None:
    env = {
        "id": "evt_abc",
        "type": "deployment.error",
        "createdAt": 1732280000000,
        "payload": {
            "target": "production",
            "deployment": {
                "id": "dpl_999",
                "name": "my-saas",
                "url": "my-saas-xyz.vercel.app",
                "meta": {
                    "githubCommitRef": "main",
                    "githubCommitSha": "abcdef1234567890",
                    "githubCommitMessage": "Fix checkout middleware",
                },
            },
            "project": {"id": "prj_123", "name": "my-saas"},
            "user": {"username": "mike"},
        },
    }
    evt = parse_vercel_webhook_envelope(env)
    assert evt.deployment_id == "dpl_999"
    assert evt.project_name == "my-saas"
    assert evt.target == "production"
    assert evt.state == "ERROR"
    assert evt.url == "https://my-saas-xyz.vercel.app"
    assert evt.git_branch == "main"
    assert evt.git_commit_sha == "abcdef1234"
    assert evt.creator == "mike"


@pytest.mark.asyncio
async def test_native_vercel_push_webhook_signature_filter_and_wake(
    test_settings: Settings,
    embedder: HashEmbeddingProvider,
) -> None:
    test_settings.vercel_webhook_secret = "vercel_secret_xyz"
    test_settings.vercel_project_filter = "my-saas"
    test_settings.vercel_priority_policy = "smart"

    llm_client, _ = make_mock_client(
        test_settings,
        [
            FakeMessage(
                content=[
                    FakeBlock(
                        type="text",
                        text="Alert: production deployment dpl_fail for my-saas failed.",
                    )
                ]
            )
        ],
    )

    daemon = AlwaysOnDaemon(test_settings, embedder=embedder, llm_client=llm_client)
    app = create_app(settings=test_settings, daemon=daemon, run_background_loop=False)
    await daemon.start(run_background_loop=False)

    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            error_envelope = {
                "id": "evt_err_1",
                "type": "deployment.error",
                "createdAt": 1732280000000,
                "payload": {
                    "target": "production",
                    "deployment": {
                        "id": "dpl_fail",
                        "name": "my-saas",
                        "url": "my-saas-bad.vercel.app",
                        "meta": {"githubCommitRef": "main"},
                    },
                    "project": {"id": "prj_1", "name": "my-saas"},
                },
            }
            raw = json.dumps(error_envelope).encode("utf-8")

            # 1. Invalid / missing x-vercel-signature -> 401
            bad = await client.post(
                "/api/events/webhook/vercel",
                content=raw,
                headers={"Content-Type": "application/json", "x-vercel-signature": "bad"},
            )
            assert bad.status_code == 401

            # 2. Filtered project returns 200 accepted=False (prevents Vercel retries)
            other_raw = json.dumps(
                {
                    "id": "evt_other",
                    "type": "deployment.succeeded",
                    "payload": {
                        "deployment": {"id": "dpl_other", "name": "unrelated-site"},
                        "project": {"id": "prj_other", "name": "unrelated-site"},
                    },
                }
            ).encode("utf-8")
            filtered = await client.post(
                "/api/events/webhook/vercel",
                content=other_raw,
                headers={
                    "Content-Type": "application/json",
                    "x-vercel-signature": _sign_vercel(other_raw, "vercel_secret_xyz"),
                },
            )
            assert filtered.status_code == 200
            assert filtered.json()["accepted"] is False
            assert filtered.json()["reason"] == "filtered_project"

            # 3. Valid signed production error -> priority=high, wakes agent, appears in dashboard chat!
            good = await client.post(
                "/api/events/webhook/vercel?process_now=true",
                content=raw,
                headers={
                    "Content-Type": "application/json",
                    "x-vercel-signature": _sign_vercel(raw, "vercel_secret_xyz"),
                },
            )
            assert good.status_code == 200
            body = good.json()
            assert body["accepted"] is True
            assert body["state"] == "ERROR"
            assert body["priority"] == "high"

            telem = (await client.get("/api/telemetry")).json()
            assert len(telem["chat_history"]) == 1
            assert telem["chat_history"][0]["trigger_actor"] == "vercel_push"
            assert "dpl_fail" in telem["chat_history"][0]["content"]

            # 4. Preview success -> smart policy sets priority=normal (salience-gated, NOT hard override)
            prev_raw = json.dumps(
                {
                    "id": "evt_prev",
                    "type": "deployment.succeeded",
                    "payload": {
                        "target": "preview",
                        "deployment": {
                            "id": "dpl_prev",
                            "name": "my-saas",
                            "url": "my-saas-git-feat.vercel.app",
                        },
                        "project": {"id": "prj_1", "name": "my-saas"},
                    },
                }
            ).encode("utf-8")
            prev_res = await client.post(
                "/api/events/webhook/vercel",
                content=prev_raw,
                headers={
                    "Content-Type": "application/json",
                    "x-vercel-signature": _sign_vercel(prev_raw, "vercel_secret_xyz"),
                },
            )
            assert prev_res.status_code == 200
            assert prev_res.json()["priority"] == "normal"
    finally:
        await daemon.stop()


@pytest.mark.asyncio
async def test_zero_tunnel_vercel_poller_seeds_detects_error_and_fetches_logs(
    tmp_path: Path,
    test_settings: Settings,
    embedder: HashEmbeddingProvider,
) -> None:
    state_file = tmp_path / "vercel_state.json"

    deployments_list = [
        {
            "uid": "dpl_old_1",
            "name": "portfolio",
            "projectId": "prj_p1",
            "url": "portfolio-old.vercel.app",
            "target": "production",
            "readyState": "READY",
            "createdAt": 1732200000000,
        }
    ]

    def vercel_api_handler(req: httpx.Request) -> httpx.Response:
        assert req.headers.get("authorization") == "Bearer vcl_test_token"
        if "/v6/deployments" in str(req.url):
            return httpx.Response(200, json={"deployments": deployments_list})
        if "/v3/deployments/dpl_new_err/events" in str(req.url):
            return httpx.Response(
                200,
                json=[
                    {"payload": {"text": "TypeError: Cannot read properties of undefined", "info": {"type": "stderr"}}},
                    {"payload": {"text": "Error: Command \"npm run build\" exited with 1", "info": {"type": "stderr"}}},
                ],
            )
        return httpx.Response(404)

    llm_client, _ = make_mock_client(
        test_settings,
        [
            FakeMessage(
                content=[
                    FakeBlock(
                        type="text",
                        text="Your portfolio build failed in npm run build due to a TypeError.",
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
            httpx.AsyncClient(transport=httpx.MockTransport(vercel_api_handler)) as v_client,
            httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as d_client,
        ):
            # 1. First poll without --force seeds existing history cleanly (0 LLM calls!)
            seed_res = await poll_vercel_and_notify_once(
                api_token="vcl_test_token",
                daemon_url="http://test",
                state_path=state_file,
                process_now=True,
                vercel_client=v_client,
                daemon_client=d_client,
            )
            assert seed_res.status == "seeded"
            assert VercelWatcherState.load(state_file).seen_deployments == {"dpl_old_1": "READY"}

            # 2. A new failed deployment appears -> poller fetches build error logs & wakes Aion!
            deployments_list.insert(
                0,
                {
                    "uid": "dpl_new_err",
                    "name": "portfolio",
                    "projectId": "prj_p1",
                    "url": "portfolio-broken.vercel.app",
                    "target": "production",
                    "readyState": "ERROR",
                    "meta": {"githubCommitRef": "main", "githubCommitMessage": "Upgrade Next.js"},
                    "createdAt": 1732290000000,
                },
            )
            err_res = await poll_vercel_and_notify_once(
                api_token="vcl_test_token",
                daemon_url="http://test",
                state_path=state_file,
                process_now=True,
                vercel_client=v_client,
                daemon_client=d_client,
            )
            assert err_res.status == "notified"
            assert err_res.notified_count == 1

            telem = (await d_client.get("/api/telemetry")).json()
            assert len(telem["fast_buffer"]) == 1
            assert "npm run build" in telem["fast_buffer"][0]["summary"] or "TypeError" in telem["fast_buffer"][0]["summary"]
            assert len(telem["chat_history"]) == 1
            assert telem["chat_history"][0]["trigger_actor"] == "vercel_poller"

            # 3. Third poll with no changes -> up_to_date (zero duplicate notifications)
            idle_res = await poll_vercel_and_notify_once(
                api_token="vcl_test_token",
                daemon_url="http://test",
                state_path=state_file,
                process_now=True,
                vercel_client=v_client,
                daemon_client=d_client,
            )
            assert idle_res.status == "up_to_date"
    finally:
        await daemon.stop()
