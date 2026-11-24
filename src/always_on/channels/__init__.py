"""Bidirectional communication channels (Discord bot bridge)."""

from .discord_bot import (
    DiscordBridge,
    chunk_discord_message,
    classify_proactive_alert,
    is_in_quiet_hours,
    parse_id_set,
    parse_optional_int,
    should_accept_discord_message,
    strip_bot_mention,
)

__all__ = [
    "DiscordBridge",
    "chunk_discord_message",
    "classify_proactive_alert",
    "is_in_quiet_hours",
    "parse_id_set",
    "parse_optional_int",
    "should_accept_discord_message",
    "strip_bot_mention",
]
