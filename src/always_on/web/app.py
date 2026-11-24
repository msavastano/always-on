"""FastAPI server exposing chat, webhook ingestion, memory controls, sleep trigger, and SSE telemetry."""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Literal

from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from always_on.config import Settings
from always_on.daemon import AlwaysOnDaemon
from always_on.watchers.vercel import (
    build_vercel_webhook_payload,
    fetch_deployment_error_logs,
    matches_project_filter,
    parse_vercel_webhook_envelope,
    verify_vercel_signature,
)

STATIC_DIR = Path(__file__).with_name("static")


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=16000)
    is_correction: bool = False


class WebhookRequest(BaseModel):
    content: str = Field(min_length=1, max_length=16000)
    actor_id: str = "webhook"
    payload: dict[str, Any] = Field(default_factory=dict)
    priority: Literal["normal", "high"] = "normal"
    hard_override: bool = False


class PinRequest(BaseModel):
    pinned: bool


def create_app(
    *,
    settings: Settings | None = None,
    daemon: AlwaysOnDaemon | None = None,
    run_background_loop: bool = True,
) -> FastAPI:
    """Create and configure the FastAPI application bound to an AlwaysOnDaemon."""
    resolved_settings = settings or Settings()
    agent_daemon = daemon or AlwaysOnDaemon(resolved_settings)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        await agent_daemon.start(run_background_loop=run_background_loop)
        try:
            yield
        finally:
            await agent_daemon.stop()

    app = FastAPI(
        title=f"{resolved_settings.agent_name} — Always-On CLS Agent",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.daemon = agent_daemon

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        html_path = STATIC_DIR / "index.html"
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "agent_name": agent_daemon.settings.agent_name,
            "model": agent_daemon.settings.anthropic_model,
        }

    @app.get("/api/telemetry")
    async def telemetry() -> dict[str, Any]:
        return await agent_daemon.get_telemetry_snapshot()

    @app.get("/api/stream")
    async def stream_telemetry(request: Request) -> StreamingResponse:
        queue = agent_daemon.subscribe_sse()

        async def event_generator() -> AsyncIterator[str]:
            try:
                if agent_daemon.is_stopping:
                    return
                # Emit initial snapshot immediately
                initial = await agent_daemon.get_telemetry_snapshot()
                yield f"event: telemetry\ndata: {json.dumps(initial)}\n\n"

                while not agent_daemon.is_stopping:
                    if await request.is_disconnected():
                        break
                    get_task = asyncio.create_task(queue.get())
                    stop_task = asyncio.create_task(agent_daemon._stop_event.wait())
                    done, pending = await asyncio.wait(
                        {get_task, stop_task},
                        timeout=5.0,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for task in pending:
                        task.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await task

                    if stop_task in done or agent_daemon.is_stopping:
                        break
                    if await request.is_disconnected():
                        break

                    if get_task in done:
                        payload = get_task.result()
                    else:
                        payload = await agent_daemon.get_telemetry_snapshot()

                    yield f"event: telemetry\ndata: {json.dumps(payload)}\n\n"
            except asyncio.CancelledError:
                return
            finally:
                agent_daemon.unsubscribe_sse(queue)

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post("/api/chat")
    async def post_chat(req: ChatRequest) -> dict[str, Any]:
        reply = await agent_daemon.send_chat(
            req.message,
            is_correction=req.is_correction,
        )
        return reply.model_dump()

    @app.delete("/api/chat")
    async def delete_chat() -> dict[str, Any]:
        cleared = await agent_daemon.clear_chat_history()
        return {"cleared": cleared}

    @app.post("/api/events/webhook")
    async def post_webhook(
        req: WebhookRequest,
        process_now: bool = Query(default=False),
        x_webhook_secret: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        expected_secret = (agent_daemon.settings.webhook_secret or "").strip()
        if expected_secret:
            supplied = (x_webhook_secret or "").strip()
            if not supplied and authorization and authorization.lower().startswith("bearer "):
                supplied = authorization[7:].strip()
            if not supplied or not hmac.compare_digest(supplied, expected_secret):
                raise HTTPException(status_code=401, detail="Invalid or missing webhook secret.")

        event = await agent_daemon.push_webhook(
            content=req.content,
            actor_id=req.actor_id,
            payload=req.payload,
            priority=req.priority,
            hard_override=req.hard_override,
        )
        if process_now:
            await agent_daemon.tick_once()
        return {
            "accepted": True,
            "event_id": event.id,
            "source": event.source,
            "actor_id": event.actor_id,
        }

    @app.post("/api/events/webhook/vercel")
    async def post_vercel_webhook(
        request: Request,
        process_now: bool = Query(default=False),
        x_vercel_signature: str | None = Header(default=None),
    ) -> dict[str, Any]:
        raw_body = await request.body()
        secret = (agent_daemon.settings.vercel_webhook_secret or "").strip()
        if secret and not verify_vercel_signature(raw_body, x_vercel_signature, secret):
            raise HTTPException(status_code=401, detail="Invalid or missing x-vercel-signature.")

        try:
            envelope = json.loads(raw_body.decode("utf-8"))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid JSON payload: {exc}") from exc

        if not isinstance(envelope, dict):
            raise HTTPException(status_code=400, detail="Vercel webhook body must be a JSON object.")

        vercel_evt = parse_vercel_webhook_envelope(envelope)

        # Return 200 OK cleanly for filtered projects so Vercel does not retry endlessly
        if not matches_project_filter(
            vercel_evt.project_name,
            vercel_evt.project_id,
            agent_daemon.settings.vercel_project_filter,
        ):
            return {
                "accepted": False,
                "reason": "filtered_project",
                "project": vercel_evt.project_name,
            }

        # Best-effort build error log enrichment on failed deploys (bounded 4s timeout)
        if vercel_evt.state == "ERROR" and agent_daemon.settings.vercel_api_token.strip():
            vercel_evt.error_logs = await fetch_deployment_error_logs(
                vercel_evt.deployment_id,
                agent_daemon.settings.vercel_api_token,
                team_id=agent_daemon.settings.vercel_team_id,
                timeout_seconds=4.0,
            )

        normalized = build_vercel_webhook_payload(
            vercel_evt,
            policy=agent_daemon.settings.vercel_priority_policy,
        )
        event = await agent_daemon.push_webhook(
            content=normalized["content"],
            actor_id=normalized["actor_id"],
            payload=normalized["payload"],
            priority=normalized["priority"],
            hard_override=normalized["hard_override"],
        )
        if process_now:
            await agent_daemon.tick_once()

        return {
            "accepted": True,
            "event_id": event.id,
            "deployment_id": vercel_evt.deployment_id,
            "project": vercel_evt.project_name,
            "target": vercel_evt.target,
            "state": vercel_evt.state,
            "priority": normalized["priority"],
        }

    @app.post("/api/sleep")
    async def post_sleep() -> dict[str, Any]:
        report = await agent_daemon.trigger_sleep(reason="manual")
        return report.model_dump()

    @app.get("/api/memory")
    async def get_memory() -> dict[str, Any]:
        core = await agent_daemon.store.list_core_memory()
        archive = await agent_daemon.store.list_memory_archive(limit=100)
        priorities = await agent_daemon.store.list_active_priorities()
        return {
            "core_memory": [m.model_dump() for m in core],
            "memory_archive": [a.model_dump() for a in archive],
            "active_priorities": [p.model_dump() for p in priorities],
        }

    @app.post("/api/memory/{memory_id}/pin")
    async def set_pin(memory_id: str, req: PinRequest) -> dict[str, Any]:
        updated = await agent_daemon.store.set_memory_pinned(memory_id, req.pinned)
        if updated is None:
            raise HTTPException(status_code=404, detail=f"Memory item '{memory_id}' not found.")
        await agent_daemon._broadcast_telemetry()
        return updated.model_dump()

    return app
