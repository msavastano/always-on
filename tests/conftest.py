"""Shared pytest fixtures and deterministic Anthropic SDK mocks."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio

from always_on.config import Settings
from always_on.db.store import AgentStore
from always_on.embeddings.provider import HashEmbeddingProvider
from always_on.llm.client import HaikuClient


@dataclass
class FakeUsage:
    input_tokens: int = 100
    cache_read_input_tokens: int = 600
    cache_creation_input_tokens: int = 0
    output_tokens: int = 80


@dataclass
class FakeBlock:
    type: str
    text: str | None = None
    thinking: str | None = None
    signature: str | None = None
    id: str | None = None
    name: str | None = None
    input: dict[str, Any] | None = None


@dataclass
class FakeMessage:
    content: list[FakeBlock]
    stop_reason: str = "end_turn"
    usage: FakeUsage = field(default_factory=FakeUsage)
    model: str = "claude-haiku-5-5"


class FakeAsyncMessages:
    """Mock AsyncAnthropic.messages capturing every request payload."""

    def __init__(self, responses: list[FakeMessage | Exception]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> FakeMessage:
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("FakeAsyncMessages ran out of queued mock responses.")
        nxt = self._responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


class FakeAnthropicSDK:
    """Drop-in mock for anthropic.AsyncAnthropic."""

    def __init__(self, responses: list[FakeMessage | Exception]) -> None:
        self.messages = FakeAsyncMessages(responses)

    async def close(self) -> None:
        return None


@pytest.fixture
def test_settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        anthropic_api_key="sk-ant-test-key",
        anthropic_model="claude-haiku-5-5",
        db_path=tmp_path / "test_agent.db",
        inbox_dir=tmp_path / "inbox",
        notes_dir=tmp_path / "notes",
        mcp_config_path=tmp_path / "mcp_servers.json",
        embedding_provider="mock",
    )


@pytest.fixture
def embedder() -> HashEmbeddingProvider:
    return HashEmbeddingProvider(dim=128)


@pytest_asyncio.fixture
async def store(test_settings: Settings) -> AgentStore:
    s = AgentStore(test_settings.db_path)
    await s.connect()
    yield s
    await s.close()


def make_mock_client(
    settings: Settings, responses: list[FakeMessage | Exception]
) -> tuple[HaikuClient, FakeAnthropicSDK]:
    sdk = FakeAnthropicSDK(responses)
    return HaikuClient(settings, sdk_client=sdk), sdk
