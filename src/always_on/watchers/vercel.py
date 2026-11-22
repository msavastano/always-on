"""Vercel deployment webhook receiver helpers (HMAC-SHA1) and zero-tunnel REST API poller."""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import httpx

from always_on.config import VercelPriorityPolicy

VERCEL_API_BASE = "https://api.vercel.com"
DEFAULT_VERCEL_STATE_PATH = Path("data/vercel_watcher_state.json")
MAX_ERROR_LOG_CHARS = 1600
MAX_SEEN_DEPLOYMENTS = 200
_USER_AGENT = "AlwaysOnCLSAgent-VercelWatcher/0.1"

_WEBHOOK_TYPE_TO_STATE: dict[str, str] = {
    "deployment.created": "BUILDING",
    "deployment.succeeded": "READY",
    "deployment.ready": "READY",
    "deployment.error": "ERROR",
    "deployment.canceled": "CANCELED",
    "deployment.check-rerequested": "BUILDING",
}


@dataclass
class VercelDeploymentEvent:
    """Normalized Vercel deployment lifecycle event from either push webhook or REST API."""

    deployment_id: str
    project_name: str
    project_id: str = ""
    url: str = ""
    target: str = "preview"  # "production" | "preview"
    event_type: str = "deployment.succeeded"
    state: str = "READY"  # READY | ERROR | BUILDING | CANCELED
    git_branch: str = ""
    git_commit_sha: str = ""
    git_commit_message: str = ""
    creator: str = ""
    created_at_ms: int = 0
    error_logs: str = ""
    source_mode: Literal["vercel_push", "vercel_poller"] = "vercel_poller"


@dataclass
class VercelWatcherState:
    """Persistent state tracking seen deployment IDs and their last notified state."""

    seen_deployments: dict[str, str] = field(default_factory=dict)
    initialized: bool = False
    last_checked_at: str | None = None

    @classmethod
    def load(cls, path: Path) -> "VercelWatcherState":
        if not path.is_file():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            seen = data.get("seen_deployments")
            return cls(
                seen_deployments=dict(seen) if isinstance(seen, dict) else {},
                initialized=bool(data.get("initialized", False)),
                last_checked_at=data.get("last_checked_at"),
            )
        except Exception:
            return cls()

    def save(self, path: Path) -> None:
        # Bound history to most recent MAX_SEEN_DEPLOYMENTS entries
        if len(self.seen_deployments) > MAX_SEEN_DEPLOYMENTS:
            items = list(self.seen_deployments.items())[-MAX_SEEN_DEPLOYMENTS:]
            self.seen_deployments = dict(items)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


@dataclass
class VercelPollResult:
    """Summary of one poll_vercel_and_notify_once() cycle."""

    status: Literal["notified", "seeded", "up_to_date", "error"]
    notified_count: int = 0
    deployments: list[str] = field(default_factory=list)
    message: str = ""


def verify_vercel_signature(
    raw_body: bytes, signature_header: str | None, secret: str
) -> bool:
    """Verify Vercel's `x-vercel-signature` HMAC-SHA1 hex digest over raw request bytes."""
    cleaned_secret = secret.strip()
    if not cleaned_secret or not signature_header:
        return False
    expected = hmac.new(
        cleaned_secret.encode("utf-8"),
        raw_body,
        hashlib.sha1,
    ).hexdigest()
    return hmac.compare_digest(expected, signature_header.strip().lower())


def _normalize_url(raw_url: str | None) -> str:
    cleaned = str(raw_url or "").strip()
    if not cleaned:
        return ""
    if cleaned.startswith(("http://", "https://")):
        return cleaned
    return f"https://{cleaned}"


def _extract_git_meta(meta: dict[str, Any] | None) -> tuple[str, str, str]:
    m = meta or {}
    branch = str(
        m.get("githubCommitRef")
        or m.get("gitlabCommitRef")
        or m.get("bitbucketCommitRef")
        or m.get("branchAlias")
        or ""
    ).strip()
    sha = str(
        m.get("githubCommitSha")
        or m.get("gitlabCommitSha")
        or m.get("bitbucketCommitSha")
        or ""
    ).strip()[:10]
    msg = str(
        m.get("githubCommitMessage")
        or m.get("gitlabCommitMessage")
        or m.get("bitbucketCommitMessage")
        or ""
    ).strip()
    if len(msg) > 160:
        msg = msg[:159].rstrip() + "…"
    return branch, sha, msg


def matches_project_filter(
    project_name: str, project_id: str, filter_csv: str
) -> bool:
    """Return True if project matches comma-separated filter (or if filter is empty)."""
    raw = (filter_csv or "").strip()
    if not raw:
        return True
    allowed = {part.strip().lower() for part in raw.split(",") if part.strip()}
    if not allowed:
        return True
    return (
        project_name.strip().lower() in allowed
        or (bool(project_id) and project_id.strip().lower() in allowed)
    )


def parse_vercel_webhook_envelope(envelope: dict[str, Any]) -> VercelDeploymentEvent:
    """Parse a native Vercel Webhook JSON envelope into a normalized VercelDeploymentEvent."""
    event_type = str(envelope.get("type") or "deployment.unknown").strip()
    payload = envelope.get("payload") or {}
    deployment = payload.get("deployment") or {}
    project = payload.get("project") or {}
    user = payload.get("user") or {}

    deployment_id = str(
        deployment.get("id") or envelope.get("id") or f"dpl_{uuid4().hex[:8]}"
    ).strip()
    project_name = str(
        project.get("name") or deployment.get("name") or "unknown-project"
    ).strip()
    project_id = str(project.get("id") or "").strip()
    raw_target = payload.get("target") or deployment.get("target")
    target = "production" if str(raw_target or "").lower() == "production" else "preview"
    state = _WEBHOOK_TYPE_TO_STATE.get(event_type, "UNKNOWN")
    branch, sha, commit_msg = _extract_git_meta(deployment.get("meta"))

    return VercelDeploymentEvent(
        deployment_id=deployment_id,
        project_name=project_name,
        project_id=project_id,
        url=_normalize_url(deployment.get("url")),
        target=target,
        event_type=event_type,
        state=state,
        git_branch=branch,
        git_commit_sha=sha,
        git_commit_message=commit_msg,
        creator=str(user.get("username") or user.get("id") or "").strip(),
        created_at_ms=int(envelope.get("createdAt") or 0),
        source_mode="vercel_push",
    )


def parse_vercel_api_deployment(item: dict[str, Any]) -> VercelDeploymentEvent:
    """Parse one deployment object from Vercel REST API `GET /v6/deployments`."""
    uid = str(item.get("uid") or item.get("id") or "").strip()
    name = str(item.get("name") or "unknown-project").strip()
    project_id = str(item.get("projectId") or "").strip()
    raw_target = item.get("target")
    target = "production" if str(raw_target or "").lower() == "production" else "preview"
    state = str(item.get("readyState") or item.get("state") or "UNKNOWN").upper().strip()

    state_to_event = {
        "READY": "deployment.succeeded",
        "ERROR": "deployment.error",
        "CANCELED": "deployment.canceled",
        "BUILDING": "deployment.created",
        "INITIALIZING": "deployment.created",
        "QUEUED": "deployment.created",
    }
    event_type = state_to_event.get(state, f"deployment.{state.lower()}")
    branch, sha, commit_msg = _extract_git_meta(item.get("meta"))
    creator_obj = item.get("creator") or {}
    creator = str(
        creator_obj.get("username") or creator_obj.get("email") or creator_obj.get("uid") or ""
    ).strip()

    return VercelDeploymentEvent(
        deployment_id=uid,
        project_name=name,
        project_id=project_id,
        url=_normalize_url(item.get("url")),
        target=target,
        event_type=event_type,
        state=state,
        git_branch=branch,
        git_commit_sha=sha,
        git_commit_message=commit_msg,
        creator=creator,
        created_at_ms=int(item.get("createdAt") or item.get("created") or 0),
        source_mode="vercel_poller",
    )


def classify_vercel_priority(
    event: VercelDeploymentEvent,
    policy: VercelPriorityPolicy = "smart",
) -> tuple[Literal["normal", "high"], bool]:
    """Determine (priority, hard_override) under the configured salience policy.

    - smart (default):
        * ERROR / deployment.error -> high (always wakes immediately)
        * READY / deployment.succeeded on production -> high (always wakes immediately)
        * previews, deployment.created, CANCELED -> normal (local salience gating)
    - failures_only:
        * ERROR / deployment.error -> high; all others -> normal
    - all_high:
        * every deployment event -> high
    """
    is_error = event.state == "ERROR" or event.event_type == "deployment.error"
    is_prod_success = (
        event.target == "production"
        and (event.state == "READY" or event.event_type in ("deployment.succeeded", "deployment.ready"))
    )

    if policy == "all_high":
        return "high", True
    if policy == "failures_only":
        return ("high", True) if is_error else ("normal", False)

    # Smart policy
    if is_error or is_prod_success:
        return "high", True
    return "normal", False


def _extract_event_log_line(entry: dict[str, Any]) -> tuple[str, bool]:
    """Extract (text, is_stderr) from a Vercel /v3/deployments/{id}/events item."""
    payload = entry.get("payload")
    if isinstance(payload, dict):
        text = str(payload.get("text") or entry.get("text") or "").strip()
        info = payload.get("info") if isinstance(payload.get("info"), dict) else {}
        stream_type = str(info.get("type") or entry.get("type") or "").lower()
    else:
        text = str(entry.get("text") or "").strip()
        stream_type = str(entry.get("type") or "").lower()

    lower_text = text.lower()
    is_err = (
        stream_type in ("stderr", "error", "fatal")
        or "error:" in lower_text
        or "err!" in lower_text
        or "failed" in lower_text
    )
    return text, is_err


async def fetch_deployment_error_logs(
    deployment_id: str,
    api_token: str,
    *,
    team_id: str | None = None,
    max_chars: int = MAX_ERROR_LOG_CHARS,
    timeout_seconds: float = 5.0,
    http_client: httpx.AsyncClient | None = None,
) -> str:
    """Pull a concise tail of build errors from Vercel `/v3/deployments/{id}/events`."""
    token = (api_token or "").strip()
    if not token or not deployment_id:
        return ""

    params: dict[str, Any] = {"builds": 1, "direction": "backward", "limit": 80}
    if team_id and team_id.strip():
        params["teamId"] = team_id.strip()

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "User-Agent": _USER_AGENT,
    }
    url = f"{VERCEL_API_BASE}/v3/deployments/{deployment_id}/events"

    owns_client = http_client is None
    client = http_client or httpx.AsyncClient(timeout=timeout_seconds)
    try:
        resp = await client.get(url, params=params, headers=headers)
        if resp.status_code != 200:
            return ""
        data = resp.json()
        if not isinstance(data, list):
            return ""

        error_lines: list[str] = []
        all_lines: list[str] = []
        # direction=backward returns newest first; reverse for chronological reading
        for entry in reversed(data):
            if not isinstance(entry, dict):
                continue
            line, is_err = _extract_event_log_line(entry)
            if not line:
                continue
            all_lines.append(line)
            if is_err:
                error_lines.append(line)

        chosen = error_lines[-20:] if error_lines else all_lines[-15:]
        joined = "\n".join(chosen).strip()
        if len(joined) > max_chars:
            joined = "...[truncated]...\n" + joined[-(max_chars - 20) :].lstrip()
        return joined
    except Exception:
        return ""
    finally:
        if owns_client:
            await client.aclose()


def build_vercel_webhook_payload(
    event: VercelDeploymentEvent,
    *,
    policy: VercelPriorityPolicy = "smart",
    force_nonce: str | None = None,
) -> dict[str, Any]:
    """Build a bounded, structured body for POST /api/events/webhook."""
    priority, hard_override = classify_vercel_priority(event, policy=policy)

    lines = [
        f"Vercel deployment {event.state} ({event.event_type}): {event.project_name} [{event.target}]",
        f"Deployment ID: {event.deployment_id}",
    ]
    if event.url:
        lines.append(f"URL: {event.url}")
    if event.git_branch or event.git_commit_sha:
        git_desc = " @ ".join(p for p in (event.git_branch, event.git_commit_sha) if p)
        lines.append(f"Git: {git_desc}")
    if event.git_commit_message:
        lines.append(f"Commit: {event.git_commit_message}")
    if event.creator:
        lines.append(f"Triggered by: {event.creator}")

    if event.error_logs:
        lines.append("")
        lines.append("Build Error Tail:")
        lines.append(event.error_logs)

    lines.append("")
    if event.state == "ERROR":
        lines.append(
            "This Vercel deployment FAILED. Summarize the failure and diagnose the likely root cause "
            "from the error logs (if included) along with concrete next steps."
        )
    else:
        lines.append(
            "Summarize this Vercel deployment status concisely and note anything relevant to active projects."
        )

    payload: dict[str, Any] = {
        "provider": "vercel",
        "source_mode": event.source_mode,
        "deployment_id": event.deployment_id,
        "project_name": event.project_name,
        "project_id": event.project_id,
        "target": event.target,
        "event_type": event.event_type,
        "state": event.state,
        "url": event.url,
        "git_branch": event.git_branch,
        "git_commit_sha": event.git_commit_sha,
    }
    if force_nonce:
        payload["force_nonce"] = force_nonce

    return {
        "content": "\n".join(lines),
        "actor_id": event.source_mode,
        "priority": priority,
        "hard_override": hard_override,
        "payload": payload,
    }


async def poll_vercel_and_notify_once(
    *,
    api_token: str,
    daemon_url: str = "http://127.0.0.1:8420",
    state_path: Path = DEFAULT_VERCEL_STATE_PATH,
    team_id: str = "",
    project_filter: str = "",
    target_filter: str = "",
    policy: VercelPriorityPolicy = "smart",
    force: bool = False,
    process_now: bool = False,
    webhook_secret: str = "",
    vercel_client: httpx.AsyncClient | None = None,
    daemon_client: httpx.AsyncClient | None = None,
    deliver_fn: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]] | None = None,
) -> VercelPollResult:
    """Poll Vercel `/v6/deployments` and POST unseen deployment transitions to the daemon webhook."""
    token = (api_token or "").strip()
    if not token:
        return VercelPollResult(
            status="error",
            message="VERCEL_API_TOKEN is not configured. Create a token at https://vercel.com/account/tokens and add it to .env.",
        )

    state = VercelWatcherState.load(state_path)
    now_iso = datetime.now(timezone.utc).isoformat()

    params: dict[str, Any] = {
        "state": "READY,ERROR,CANCELED",
        "limit": 20,
    }
    if team_id.strip():
        params["teamId"] = team_id.strip()
    if target_filter.strip():
        params["target"] = target_filter.strip()

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "User-Agent": _USER_AGENT,
    }

    owns_vercel = vercel_client is None
    v_client = vercel_client or httpx.AsyncClient(timeout=15.0)
    try:
        resp = await v_client.get(
            f"{VERCEL_API_BASE}/v6/deployments",
            params=params,
            headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()
        raw_deployments = data.get("deployments") or []
    except Exception as exc:
        return VercelPollResult(
            status="error",
            message=f"Vercel API request failed: {exc}",
        )

    try:
        events: list[VercelDeploymentEvent] = []
        for item in raw_deployments:
            if not isinstance(item, dict):
                continue
            evt = parse_vercel_api_deployment(item)
            if not evt.deployment_id:
                continue
            if not matches_project_filter(evt.project_name, evt.project_id, project_filter):
                continue
            events.append(evt)

        if not events:
            state.initialized = True
            state.last_checked_at = now_iso
            state.save(state_path)
            return VercelPollResult(
                status="up_to_date",
                message="No matching Vercel deployments found.",
            )

        # First-run baseline seeding: avoid flooding historical deploys unless --force is passed
        if not state.initialized and not force:
            for evt in events:
                state.seen_deployments[evt.deployment_id] = evt.state
            state.initialized = True
            state.last_checked_at = now_iso
            state.save(state_path)
            return VercelPollResult(
                status="seeded",
                message=(
                    f"Seeded {len(events)} existing Vercel deployment(s) "
                    f"(latest: {events[0].project_name} {events[0].state}); watching for new transitions."
                ),
            )

        # Pick deployments to deliver (oldest-first for natural chronological order)
        if force:
            to_notify = [events[0]]
        else:
            to_notify = [
                evt
                for evt in reversed(events)
                if state.seen_deployments.get(evt.deployment_id) != evt.state
            ]

        if not to_notify:
            state.last_checked_at = now_iso
            state.save(state_path)
            return VercelPollResult(
                status="up_to_date",
                message=f"No new Vercel deployment transitions (latest: {events[0].project_name} {events[0].state}).",
            )

        webhook_url = f"{daemon_url.rstrip('/')}/api/events/webhook"
        if process_now:
            webhook_url += "?process_now=true"
        wh_headers: dict[str, str] = {"Content-Type": "application/json"}
        if webhook_secret.strip():
            wh_headers["X-Webhook-Secret"] = webhook_secret.strip()

        owns_daemon = deliver_fn is None and daemon_client is None
        d_client = (
            None
            if deliver_fn is not None
            else (daemon_client or httpx.AsyncClient(timeout=60.0))
        )
        delivered: list[str] = []
        try:
            for evt in to_notify:
                if evt.state == "ERROR":
                    evt.error_logs = await fetch_deployment_error_logs(
                        evt.deployment_id,
                        token,
                        team_id=team_id,
                        http_client=v_client,
                    )
                body = build_vercel_webhook_payload(
                    evt,
                    policy=policy,
                    force_nonce=uuid4().hex[:8] if force else None,
                )
                if deliver_fn is not None:
                    await deliver_fn(body)
                else:
                    assert d_client is not None
                    w_resp = await d_client.post(webhook_url, json=body, headers=wh_headers)
                    w_resp.raise_for_status()
                state.seen_deployments[evt.deployment_id] = evt.state
                delivered.append(f"{evt.project_name} [{evt.target}] {evt.state} ({evt.deployment_id})")
        except Exception as exc:
            # Persist any earlier successful deliveries, but do NOT mark the failed one as seen
            if delivered:
                state.initialized = True
                state.last_checked_at = now_iso
                state.save(state_path)
            target_desc = "in-process daemon" if deliver_fn is not None else webhook_url
            return VercelPollResult(
                status="error",
                notified_count=len(delivered),
                deployments=delivered,
                message=f"Failed delivering Vercel deployment to {target_desc}: {exc}",
            )
        finally:
            if owns_daemon and d_client is not None:
                await d_client.aclose()

        # Mark remaining snapshot items seeded on --force first run so next poll only catches new deploys
        if not state.initialized:
            for evt in events:
                state.seen_deployments.setdefault(evt.deployment_id, evt.state)
            state.initialized = True

        state.last_checked_at = now_iso
        state.save(state_path)

        return VercelPollResult(
            status="notified",
            notified_count=len(delivered),
            deployments=delivered,
            message=f"Delivered {len(delivered)} Vercel deployment(s): " + "; ".join(delivered),
        )
    finally:
        if owns_vercel:
            await v_client.aclose()
