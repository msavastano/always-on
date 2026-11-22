"""Unit & integration tests for web search providers, SQLite cache, quotas, SSRF, and SSE shutdown."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest

from always_on.config import Settings
from always_on.daemon import AlwaysOnDaemon
from always_on.db.store import AgentStore
from always_on.embeddings.provider import HashEmbeddingProvider
from always_on.models import utc_now
from always_on.safety.circuit_breaker import AgentCircuitBreaker, CircuitBreakerTripped
from always_on.tools.registry import ToolRegistry
from always_on.tools.web_search import (
    BraveSearchProvider,
    DDGSearchProvider,
    SearXNGSearchProvider,
    TavilySearchProvider,
    WebSearchProvider,
    WebSearchResult,
    WebSearchService,
    safe_fetch_webpage,
    validate_public_http_url,
)
from always_on.web.app import create_app
from tests.conftest import FakeBlock, FakeMessage, FakeUsage, make_mock_client


class CountingMockSearchProvider(WebSearchProvider):
    """Deterministic in-memory search provider for testing caching & quotas."""

    name = "mock_search"

    def __init__(self, results: list[WebSearchResult] | None = None) -> None:
        self.calls: list[tuple[str, int, int | None]] = []
        self._results = results or [
            WebSearchResult(
                title="Python 3.13 Release Notes",
                url="https://docs.python.org/3.13/whatsnew/",
                snippet="A" * 500,  # Intentionally > 300 chars to verify truncation
                published_date="2026-01-15",
            )
        ]

    async def search(
        self,
        query: str,
        *,
        max_results: int = 5,
        recency_days: int | None = None,
    ) -> list[WebSearchResult]:
        self.calls.append((query, max_results, recency_days))
        return self._results[:max_results]


@pytest.mark.asyncio
async def test_web_search_sqlite_cache_truncation_and_untrusted_framing(
    store: AgentStore,
    test_settings: Settings,
    embedder: HashEmbeddingProvider,
) -> None:
    test_settings.web_search_cache_ttl_seconds = 600
    test_settings.web_search_max_snippet_chars = 120
    cb = AgentCircuitBreaker(store, test_settings, embedder)
    provider = CountingMockSearchProvider()
    svc = WebSearchService(
        store=store, circuit_breaker=cb, settings=test_settings, provider=provider
    )
    t0 = utc_now()

    # First call hits provider, truncates snippet, wraps in untrusted delimiters, increments daily count
    out1, is_err1 = await svc.search("  Python   3.13  release ", now=t0)
    assert is_err1 is False
    assert len(provider.calls) == 1
    assert '<untrusted_web_content tool="web_search" provider="mock_search" cached="false">' in out1
    assert "</untrusted_web_content>" in out1

    inner_json = out1.split("\n", 1)[1].rsplit("\n", 1)[0]
    parsed = json.loads(inner_json)
    assert parsed["result_count"] == 1
    assert len(parsed["results"][0]["snippet"]) <= 120

    spend1 = await store.get_daily_spend(t0.strftime("%Y-%m-%d"))
    assert spend1.web_searches == 1

    # Second normalized identical call within TTL hits SQLite cache ($0, does NOT increment quota!)
    out2, is_err2 = await svc.search("python 3.13 release", now=t0 + timedelta(seconds=60))
    assert is_err2 is False
    assert len(provider.calls) == 1  # Provider NOT called again!
    assert 'cached="true"' in out2
    spend2 = await store.get_daily_spend(t0.strftime("%Y-%m-%d"))
    assert spend2.web_searches == 1

    # After TTL expires -> cache miss calls provider again
    out3, is_err3 = await svc.search("python 3.13 release", now=t0 + timedelta(seconds=601))
    assert is_err3 is False
    assert len(provider.calls) == 2
    assert 'cached="false"' in out3
    spend3 = await store.get_daily_spend(t0.strftime("%Y-%m-%d"))
    assert spend3.web_searches == 2


@pytest.mark.asyncio
async def test_web_search_daily_quota_persists_and_blocks(
    store: AgentStore,
    test_settings: Settings,
    embedder: HashEmbeddingProvider,
) -> None:
    test_settings.max_web_searches_per_day = 2
    cb = AgentCircuitBreaker(store, test_settings, embedder)
    provider = CountingMockSearchProvider()
    svc = WebSearchService(
        store=store, circuit_breaker=cb, settings=test_settings, provider=provider
    )
    t0 = utc_now()

    _, err1 = await svc.search("query one", now=t0)
    _, err2 = await svc.search("query two", now=t0)
    out3, err3 = await svc.search("query three", now=t0)

    assert err1 is False
    assert err2 is False
    assert err3 is True
    assert "Daily web search limit reached" in out3
    assert len(provider.calls) == 2


@pytest.mark.asyncio
async def test_heartbeat_gating_and_anthropic_native_search_cost_tracking(
    test_settings: Settings,
    embedder: HashEmbeddingProvider,
) -> None:
    test_settings.web_search_provider = "anthropic"
    test_settings.allow_search_on_heartbeat = False
    test_settings.anthropic_web_search_max_uses = 3
    test_settings.max_daily_usd = 0.025  # 3 server searches = $0.03 > $0.025 cap!

    @dataclass
    class FakeServerToolUsage:
        web_search_requests: int = 3
        web_fetch_requests: int = 0

    @dataclass
    class UsageWithSearch(FakeUsage):
        server_tool_use: FakeServerToolUsage = None  # type: ignore[assignment]

    usage = UsageWithSearch(
        input_tokens=100,
        cache_read_input_tokens=600,
        cache_creation_input_tokens=0,
        output_tokens=50,
        server_tool_use=FakeServerToolUsage(web_search_requests=3),
    )

    llm_client, sdk = make_mock_client(
        test_settings,
        [
            FakeMessage(
                content=[
                    FakeBlock(
                        type="server_tool_use",
                        id="srv_1",
                        name="web_search",
                        input={"query": "latest claude haiku release"},
                    ),
                    FakeBlock(type="text", text="Found the latest release details."),
                ],
                usage=usage,
            )
        ],
    )

    daemon = AlwaysOnDaemon(test_settings, embedder=embedder, llm_client=llm_client)
    await daemon.start(run_background_loop=False)
    try:
        # Heartbeat tools must omit web_search & fetch_webpage when allow_search_on_heartbeat=False
        hb_tools = daemon.tools.tool_definitions(event_source="heartbeat")
        hb_names = [t.get("name") for t in hb_tools]
        assert "web_search" not in hb_names
        assert "fetch_webpage" not in hb_names

        # Chat wake-up passes Anthropic native web_search_20260318 tool and bills 3 * $0.01 = $0.03!
        reply = await daemon.send_chat("What is the latest Claude Haiku release?")
        assert reply.status == "ok"
        assert reply.usage.web_search_requests == 3
        assert reply.usd_cost >= 0.03  # 3 * $0.01 + token cost
        assert any(tc.name == "web_search (anthropic)" for tc in reply.tool_calls)

        # Verify Anthropic tool declaration sent in request
        sent_tools = sdk.messages.calls[0]["tools"]
        native_ws = [t for t in sent_tools if t.get("type") == "web_search_20260318"]
        assert len(native_ws) == 1
        assert native_ws[0]["max_uses"] == 3

        # Verify persistent daily_spend recorded both web_searches=3 and usd_spent >= $0.03 -> trips daily_usd_cap!
        spend = await daemon.store.get_daily_spend()
        assert spend.web_searches == 3
        assert spend.usd_spent >= 0.03
        with pytest.raises(CircuitBreakerTripped) as exc:
            await daemon.circuit_breaker.check_before_call(is_wake_start=True)
        assert exc.value.reason == "daily_usd_cap"
    finally:
        await daemon.stop()


@pytest.mark.asyncio
async def test_brave_tavily_and_searxng_providers_parse_responses() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if "api.search.brave.com" in str(request.url):
            return httpx.Response(
                200,
                json={
                    "web": {
                        "results": [
                            {
                                "title": "Brave Hit",
                                "url": "https://example.com/brave",
                                "description": "Brave snippet text",
                                "age": "2 days ago",
                            }
                        ]
                    }
                },
            )
        if "api.tavily.com" in str(request.url):
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "title": "Tavily Hit",
                            "url": "https://example.com/tavily",
                            "content": "Tavily snippet text",
                            "published_date": "2026-11-20",
                        }
                    ]
                },
            )
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "SearXNG Hit",
                        "url": "https://example.com/searx",
                        "content": "SearXNG snippet text",
                    }
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        brave = BraveSearchProvider("test-brave-key", http_client=client)
        r_brave = await brave.search("test", max_results=3, recency_days=7)
        assert r_brave[0].title == "Brave Hit"
        assert r_brave[0].published_date == "2 days ago"

        tavily = TavilySearchProvider("test-tavily-key", http_client=client)
        r_tav = await tavily.search("test", max_results=3, recency_days=7)
        assert r_tav[0].title == "Tavily Hit"

        searx = SearXNGSearchProvider("http://localhost:8080", http_client=client)
        r_sx = await searx.search("test", max_results=3, recency_days=30)
        assert r_sx[0].title == "SearXNG Hit"


@pytest.mark.asyncio
async def test_ssrf_protection_blocks_private_and_local_urls() -> None:
    blocked_urls = [
        "http://127.0.0.1:8420/api/telemetry",
        "http://localhost:8420/",
        "http://169.254.169.254/latest/meta-data/",
        "http://10.0.0.1/admin",
        "http://192.168.1.1/",
        "file:///C:/Windows/win.ini",
        "ftp://example.com/file.txt",
    ]
    for bad in blocked_urls:
        out, is_err = await safe_fetch_webpage(bad)
        assert is_err is True, f"Expected SSRF block for {bad}"
        assert "Fetch blocked:" in out

    with pytest.raises(ValueError):
        await validate_public_http_url("http://[::1]:8420/")


@pytest.mark.asyncio
async def test_safe_fetch_webpage_strips_scripts_truncates_and_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_validate(url: str) -> str:
        return url

    monkeypatch.setattr("always_on.tools.web_search.validate_public_http_url", fake_validate)

    html_page = (
        "<html><head><title>Test Article</title><style>body{color:red}</style></head>"
        "<body><script>alert('xss')</script><main><h1>Heading</h1>"
        "<p>Clean article content.</p></main><footer>Footer noise</footer></body></html>"
    )
    transport = httpx.MockTransport(
        lambda req: httpx.Response(200, text=html_page, headers={"content-type": "text/html"})
    )
    async with httpx.AsyncClient(transport=transport) as client:
        out, is_err = await safe_fetch_webpage(
            "https://example.com/article", max_chars=2000, http_client=client
        )
    assert is_err is False
    assert '<untrusted_web_content tool="fetch_webpage" source_url="https://example.com/article">' in out
    assert "Test Article" in out
    assert "Clean article content." in out
    assert "alert('xss')" not in out
    assert "Footer noise" not in out


@pytest.mark.asyncio
async def test_store_migrates_legacy_daily_spend_table(tmp_path: Path) -> None:
    """Pre-existing agent.db without daily_spend.web_searches migrates seamlessly."""
    db_file = tmp_path / "legacy.db"
    conn = sqlite3.connect(db_file)
    conn.execute(
        """
        CREATE TABLE daily_spend (
            date_utc TEXT PRIMARY KEY,
            input_tokens INTEGER NOT NULL DEFAULT 0,
            cache_read_tokens INTEGER NOT NULL DEFAULT 0,
            cache_creation_tokens INTEGER NOT NULL DEFAULT 0,
            output_tokens INTEGER NOT NULL DEFAULT 0,
            usd_spent REAL NOT NULL DEFAULT 0.0,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        "INSERT INTO daily_spend VALUES ('2026-11-21', 500, 1000, 0, 200, 0.0123, '2026-11-21T00:00:00Z')"
    )
    conn.commit()
    conn.close()

    store = AgentStore(db_file)
    await store.connect()
    try:
        legacy = await store.get_daily_spend("2026-11-21")
        assert legacy.usd_spent == pytest.approx(0.0123)
        assert legacy.web_searches == 0
        updated = await store.record_web_search(count=2, date_utc="2026-11-21")
        assert updated.web_searches == 2
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_sse_stream_exits_immediately_on_request_shutdown(
    test_settings: Settings,
    embedder: HashEmbeddingProvider,
) -> None:
    """Opening /api/stream must close promptly when daemon.request_shutdown() fires (Ctrl+C fix)."""
    llm_client, _ = make_mock_client(test_settings, [])
    daemon = AlwaysOnDaemon(test_settings, embedder=embedder, llm_client=llm_client)
    app = create_app(settings=test_settings, daemon=daemon, run_background_loop=False)
    await daemon.start(run_background_loop=False)

    async def _shutdown_soon() -> None:
        await asyncio.sleep(0.1)
        daemon.request_shutdown()

    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            trigger = asyncio.create_task(_shutdown_soon())
            chunks: list[str] = []

            async def _read_stream() -> None:
                async with client.stream("GET", "/api/stream") as resp:
                    assert resp.status_code == 200
                    async for line in resp.aiter_lines():
                        if line:
                            chunks.append(line)

            # Must complete well under 1.5s once request_shutdown() triggers (NOT hang!)
            await asyncio.wait_for(_read_stream(), timeout=1.5)
            await trigger
            assert any(c.startswith("event: telemetry") for c in chunks)
    finally:
        await daemon.stop()


@pytest.mark.asyncio
async def test_client_web_search_tool_executes_inside_daemon_wake_loop(
    test_settings: Settings,
    embedder: HashEmbeddingProvider,
) -> None:
    provider = CountingMockSearchProvider()
    llm_client, _ = make_mock_client(
        test_settings,
        [
            FakeMessage(
                content=[
                    FakeBlock(type="thinking", thinking="I should search the web for this."),
                    FakeBlock(
                        type="tool_use",
                        id="tu_ws_1",
                        name="web_search",
                        input={"query": "Python 3.13 release notes", "max_results": 3},
                    ),
                ],
                stop_reason="tool_use",
            ),
            FakeMessage(
                content=[
                    FakeBlock(
                        type="text",
                        text="Python 3.13 release notes are at https://docs.python.org/3.13/whatsnew/.",
                    )
                ],
                stop_reason="end_turn",
            ),
        ],
    )

    daemon = AlwaysOnDaemon(
        test_settings,
        embedder=embedder,
        llm_client=llm_client,
        web_search_provider=provider,
    )
    await daemon.start(run_background_loop=False)
    try:
        reply = await daemon.send_chat("Find Python 3.13 release notes")
        assert reply.status == "ok"
        assert len(reply.tool_calls) == 1
        assert reply.tool_calls[0].name == "web_search"
        assert reply.tool_calls[0].is_error is False
        assert "<untrusted_web_content" in reply.tool_calls[0].result
        assert (await daemon.store.get_daily_spend()).web_searches == 1
    finally:
        await daemon.stop()


@pytest.mark.asyncio
async def test_ddg_provider_falls_back_across_backends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempted: list[str] = []

    class FakeDDGS:
        def __init__(self, timeout: int = 5) -> None:
            self.timeout = timeout

        def __enter__(self) -> "FakeDDGS":
            return self

        def __exit__(self, *args: Any) -> None:
            return None

        def text(
            self, query: str, *, max_results: int = 5, timelimit: str | None = None, backend: str = "auto"
        ) -> list[dict[str, str]]:
            attempted.append(backend)
            if backend == "first_empty":
                raise RuntimeError("No results found.")
            return [
                {
                    "title": f"Hit from {backend}",
                    "href": "https://docs.python.org/3.13/",
                    "body": "Python 3.13 notes",
                }
            ]

    import ddgs

    monkeypatch.setattr(ddgs, "DDGS", FakeDDGS)
    provider = DDGSearchProvider(backends=("first_empty", "yahoo"))
    results = await provider.search("Python 3.13 release notes", max_results=3)

    assert attempted == ["first_empty", "yahoo"]
    assert len(results) == 1
    assert results[0].title == "Hit from yahoo"
