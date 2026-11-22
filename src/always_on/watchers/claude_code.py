"""External Claude Code release watcher posting new releases to POST /api/events/webhook."""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import httpx

GITHUB_REPO = "anthropics/claude-code"
GITHUB_LATEST_RELEASE_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
RAW_CHANGELOG_URL = f"https://raw.githubusercontent.com/{GITHUB_REPO}/main/CHANGELOG.md"
DEFAULT_STATE_PATH = Path("data/claude_code_watcher_state.json")
MAX_BODY_CHARS = 2800
_USER_AGENT = "AlwaysOnCLSAgent-ReleaseWatcher/0.1"

_CHANGELOG_HEADING_RE = re.compile(r"^##\s+v?(\d+\.\d+\.\d+(?:[-+][^\s]+)?)\s*$", re.MULTILINE)


def normalize_tag(version: str) -> str:
    """Normalize version strings (e.g. '2.1.330' and 'v2.1.330') to canonical 'vX.Y.Z'."""
    cleaned = version.strip()
    if not cleaned:
        return ""
    return cleaned if cleaned.startswith("v") else f"v{cleaned}"


@dataclass
class ReleaseInfo:
    """Normalized metadata for a Claude Code release."""

    tag: str
    published_at: str
    url: str
    body: str
    source: Literal["github_releases", "github_changelog"] = "github_releases"


@dataclass
class WatcherState:
    """Persistent state stored on disk between polls/runs."""

    last_seen_tag: str | None = None
    etag: str | None = None
    last_checked_at: str | None = None

    @classmethod
    def load(cls, path: Path) -> "WatcherState":
        if not path.is_file():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return cls(
                last_seen_tag=data.get("last_seen_tag"),
                etag=data.get("etag"),
                last_checked_at=data.get("last_checked_at"),
            )
        except Exception:
            return cls()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


@dataclass
class WatcherResult:
    """Outcome of a single watcher poll."""

    status: Literal["notified", "up_to_date", "not_modified", "error"]
    tag: str | None = None
    source: str | None = None
    event_id: str | None = None
    message: str = ""


def parse_latest_changelog_section(markdown: str) -> ReleaseInfo | None:
    """Parse the top version section from anthropics/claude-code CHANGELOG.md."""
    matches = list(_CHANGELOG_HEADING_RE.finditer(markdown))
    if not matches:
        return None
    first = matches[0]
    tag = normalize_tag(first.group(1))
    start = first.end()
    end = matches[1].start() if len(matches) > 1 else len(markdown)
    body = markdown[start:end].strip()
    return ReleaseInfo(
        tag=tag,
        published_at="",
        url=f"https://github.com/{GITHUB_REPO}/releases/tag/{tag}",
        body=body,
        source="github_changelog",
    )


async def fetch_latest_claude_code_release(
    *,
    etag: str | None = None,
    github_token: str | None = None,
    timeout_seconds: float = 12.0,
    http_client: httpx.AsyncClient | None = None,
) -> tuple[ReleaseInfo | None, str | None, bool]:
    """Fetch the latest Claude Code release from GitHub Releases API with ETag & CHANGELOG fallback.

    Returns:
        (release, new_etag, not_modified)
    """
    headers: dict[str, str] = {
        "Accept": "application/vnd.github+json",
        "User-Agent": _USER_AGENT,
    }
    if github_token and github_token.strip():
        headers["Authorization"] = f"Bearer {github_token.strip()}"
    if etag:
        headers["If-None-Match"] = etag

    owns_client = http_client is None
    client = http_client or httpx.AsyncClient(timeout=timeout_seconds, follow_redirects=True)
    try:
        resp = await client.get(GITHUB_LATEST_RELEASE_URL, headers=headers)
        if resp.status_code == 304:
            return None, etag, True

        if resp.status_code == 200:
            data = resp.json()
            tag = normalize_tag(str(data.get("tag_name") or ""))
            if not tag:
                raise ValueError("GitHub release response missing tag_name.")
            new_etag = resp.headers.get("etag") or etag
            release = ReleaseInfo(
                tag=tag,
                published_at=str(data.get("published_at") or ""),
                url=str(
                    data.get("html_url")
                    or f"https://github.com/{GITHUB_REPO}/releases/tag/{tag}"
                ),
                body=str(data.get("body") or "").strip(),
                source="github_releases",
            )
            return release, new_etag, False

        # Fallback to raw CHANGELOG.md on GitHub REST rate limit (403/429) or transient 5xx
        if resp.status_code in (403, 429, 500, 502, 503):
            raw_resp = await client.get(
                RAW_CHANGELOG_URL,
                headers={"User-Agent": _USER_AGENT},
            )
            raw_resp.raise_for_status()
            parsed = parse_latest_changelog_section(raw_resp.text)
            if parsed is None:
                raise ValueError("Failed to parse latest version heading from CHANGELOG.md.")
            return parsed, etag, False

        resp.raise_for_status()
        return None, etag, False
    finally:
        if owns_client:
            await client.aclose()


def build_webhook_payload(
    release: ReleaseInfo,
    *,
    priority: Literal["normal", "high"] = "high",
    force_nonce: str | None = None,
) -> dict[str, Any]:
    """Build a bounded, structured webhook request body for POST /api/events/webhook."""
    body = release.body.strip()
    if len(body) > MAX_BODY_CHARS:
        body = body[: MAX_BODY_CHARS - 24].rstrip() + "\n...[truncated]..."

    published_line = f"Published: {release.published_at}\n" if release.published_at else ""
    content = (
        f"New Claude Code release detected: {release.tag}\n"
        f"{published_line}"
        f"Release URL: {release.url}\n\n"
        f"Release Notes:\n{body}\n\n"
        "Please summarize the most notable new features, breaking changes, and bug fixes "
        "(especially anything relevant to my OS, active projects, or preferences in Core Memory)."
    )

    payload: dict[str, Any] = {
        "repo": GITHUB_REPO,
        "tag": release.tag,
        "published_at": release.published_at,
        "url": release.url,
        "source_kind": release.source,
    }
    if force_nonce:
        payload["force_nonce"] = force_nonce

    return {
        "content": content,
        "actor_id": "claude_code_watcher",
        "priority": priority,
        "hard_override": priority == "high",
        "payload": payload,
    }


async def check_and_notify_once(
    *,
    daemon_url: str = "http://127.0.0.1:8420",
    state_path: Path = DEFAULT_STATE_PATH,
    priority: Literal["normal", "high"] = "high",
    force: bool = False,
    process_now: bool = False,
    webhook_secret: str | None = None,
    github_token: str | None = None,
    github_client: httpx.AsyncClient | None = None,
    daemon_client: httpx.AsyncClient | None = None,
    deliver_fn: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]] | None = None,
) -> WatcherResult:
    """Check GitHub for a newer Claude Code release and POST it to the daemon webhook."""
    state = WatcherState.load(state_path)
    now_iso = datetime.now(timezone.utc).isoformat()

    try:
        release, new_etag, not_modified = await fetch_latest_claude_code_release(
            etag=None if force else state.etag,
            github_token=github_token,
            http_client=github_client,
        )
    except Exception as exc:
        return WatcherResult(
            status="error",
            tag=state.last_seen_tag,
            message=f"GitHub check failed: {exc}",
        )

    if not_modified:
        state.last_checked_at = now_iso
        state.save(state_path)
        return WatcherResult(
            status="not_modified",
            tag=state.last_seen_tag,
            source="github_releases",
            message=f"No change (HTTP 304, latest={state.last_seen_tag or 'unknown'}).",
        )

    if release is None:
        return WatcherResult(status="error", message="No release data returned from GitHub.")

    # Already seen and not forcing delivery
    if not force and state.last_seen_tag == release.tag:
        state.etag = new_etag
        state.last_checked_at = now_iso
        state.save(state_path)
        return WatcherResult(
            status="up_to_date",
            tag=release.tag,
            source=release.source,
            message=f"Already notified for {release.tag}.",
        )

    # Deliver new release to Always-On daemon webhook
    webhook_url = f"{daemon_url.rstrip('/')}/api/events/webhook"
    if process_now:
        webhook_url += "?process_now=true"

    req_headers: dict[str, str] = {"Content-Type": "application/json"}
    if webhook_secret and webhook_secret.strip():
        req_headers["X-Webhook-Secret"] = webhook_secret.strip()

    body = build_webhook_payload(
        release,
        priority=priority,
        force_nonce=uuid4().hex[:8] if force else None,
    )

    if deliver_fn is not None:
        try:
            resp_data = await deliver_fn(body)
        except Exception as exc:
            # Do NOT advance last_seen_tag if delivery failed, so we retry next check!
            return WatcherResult(
                status="error",
                tag=release.tag,
                source=release.source,
                message=f"Failed to deliver {release.tag} to in-process daemon: {exc}",
            )
    else:
        owns_daemon_client = daemon_client is None
        client = daemon_client or httpx.AsyncClient(timeout=60.0)
        try:
            resp = await client.post(webhook_url, json=body, headers=req_headers)
            resp.raise_for_status()
            resp_data = resp.json()
        except Exception as exc:
            # Do NOT advance last_seen_tag if webhook delivery failed, so we retry next check!
            return WatcherResult(
                status="error",
                tag=release.tag,
                source=release.source,
                message=f"Failed to deliver {release.tag} to daemon webhook ({webhook_url}): {exc}",
            )
        finally:
            if owns_daemon_client:
                await client.aclose()

    # Persist state only after confirmed delivery
    state.last_seen_tag = release.tag
    state.etag = new_etag
    state.last_checked_at = now_iso
    state.save(state_path)

    return WatcherResult(
        status="notified",
        tag=release.tag,
        source=release.source,
        event_id=str(resp_data.get("event_id") or ""),
        message=f"Delivered Claude Code {release.tag} ({release.source}) to daemon webhook.",
    )
