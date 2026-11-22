"""Pluggable event sources and ingestion gatekeeper (self-trigger + dedup)."""

from .base import EventIngestor, EventSource, IngestionDecision
from .chat import ChatSource, PendingChatRequest
from .file_watcher import FileWatcherSource
from .heartbeat import HeartbeatSource
from .webhook import WebhookSource

__all__ = [
    "ChatSource",
    "EventIngestor",
    "EventSource",
    "FileWatcherSource",
    "HeartbeatSource",
    "IngestionDecision",
    "PendingChatRequest",
    "WebhookSource",
]
