"""External webhook watchers feeding real-world events into Always-On."""

from .claude_code import (
    ReleaseInfo,
    WatcherResult,
    WatcherState,
    build_webhook_payload,
    check_and_notify_once,
    fetch_latest_claude_code_release,
    parse_latest_changelog_section,
)
from .vercel import (
    DEFAULT_VERCEL_STATE_PATH,
    VercelDeploymentEvent,
    VercelPollResult,
    VercelWatcherState,
    build_vercel_webhook_payload,
    classify_vercel_priority,
    fetch_deployment_error_logs,
    matches_project_filter,
    parse_vercel_api_deployment,
    parse_vercel_webhook_envelope,
    poll_vercel_and_notify_once,
    verify_vercel_signature,
)

__all__ = [
    "DEFAULT_VERCEL_STATE_PATH",
    "ReleaseInfo",
    "VercelDeploymentEvent",
    "VercelPollResult",
    "VercelWatcherState",
    "WatcherResult",
    "WatcherState",
    "build_vercel_webhook_payload",
    "build_webhook_payload",
    "check_and_notify_once",
    "classify_vercel_priority",
    "fetch_deployment_error_logs",
    "fetch_latest_claude_code_release",
    "matches_project_filter",
    "parse_latest_changelog_section",
    "parse_vercel_api_deployment",
    "parse_vercel_webhook_envelope",
    "poll_vercel_and_notify_once",
    "verify_vercel_signature",
]
