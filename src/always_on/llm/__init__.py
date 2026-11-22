"""Claude Haiku 5.5 API client and cache-optimized prompt builders."""

from .client import (
    FORBIDDEN_SAMPLING_PARAMS,
    HaikuClient,
    MissingAPIKeyError,
    ParsedLLMResponse,
    ToolUseItem,
    build_haiku_request_kwargs,
    parse_anthropic_response,
)
from .prompts import (
    MIN_CACHE_PREFIX_TOKENS,
    build_cached_system_blocks,
    build_consolidation_system_blocks,
    build_consolidation_user_message,
    build_wake_user_message,
    estimate_token_count,
)

__all__ = [
    "FORBIDDEN_SAMPLING_PARAMS",
    "HaikuClient",
    "MIN_CACHE_PREFIX_TOKENS",
    "MissingAPIKeyError",
    "ParsedLLMResponse",
    "ToolUseItem",
    "build_cached_system_blocks",
    "build_consolidation_system_blocks",
    "build_consolidation_user_message",
    "build_haiku_request_kwargs",
    "build_wake_user_message",
    "estimate_token_count",
    "parse_anthropic_response",
]
