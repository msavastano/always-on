"""Pluggable client-side web search providers, SQLite TTL cache, and SSRF-safe webpage fetcher."""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import json
import re
import socket
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx

from always_on.config import Settings
from always_on.db.store import AgentStore
from always_on.safety.circuit_breaker import AgentCircuitBreaker, CircuitBreakerTripped

_MAX_FETCH_BYTES = 512 * 1024  # 512 KB download cap
_MAX_REDIRECTS = 3
_USER_AGENT = "AlwaysOnCLSAgent/0.1 (+local-personal-agent)"


@dataclass
class WebSearchResult:
    """Compact normalized web search result."""

    title: str
    url: str
    snippet: str
    published_date: str | None = None


class WebSearchProvider(ABC):
    """Abstract client-side search backend."""

    name: str = "unknown"

    @abstractmethod
    async def search(
        self,
        query: str,
        *,
        max_results: int = 5,
        recency_days: int | None = None,
    ) -> list[WebSearchResult]:
        """Execute a search query and return normalized results."""


class DDGSearchProvider(WebSearchProvider):
    """Zero-key metasearch provider backed by `ddgs` (runs off the event loop)."""

    name = "ddg"
    # Explicit per-backend fallback using real ddgs 9.x backends avoids ddgs's
    # shuffled `auto` mode failing early when DuckDuckGo (202) or Brave (429) throttles.
    DEFAULT_BACKENDS: tuple[str, ...] = ("yahoo", "auto", "startpage", "duckduckgo", "brave")

    def __init__(
        self,
        timeout_seconds: float = 10.0,
        backends: tuple[str, ...] | None = None,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.backends = backends or self.DEFAULT_BACKENDS

    @staticmethod
    def _recency_to_timelimit(recency_days: int | None) -> str | None:
        if recency_days is None:
            return None
        if recency_days <= 1:
            return "d"
        if recency_days <= 7:
            return "w"
        if recency_days <= 31:
            return "m"
        return "y"

    def _sync_search(
        self, query: str, max_results: int, recency_days: int | None
    ) -> list[WebSearchResult]:
        from ddgs import DDGS

        timelimit = self._recency_to_timelimit(recency_days)
        per_backend_timeout = int(max(3, min(8, self.timeout_seconds)))
        last_exc: Exception | None = None

        with DDGS(timeout=per_backend_timeout) as ddgs:
            for backend in self.backends:
                try:
                    raw = ddgs.text(
                        query,
                        max_results=max_results,
                        timelimit=timelimit,
                        backend=backend,
                    )
                except Exception as exc:
                    last_exc = exc
                    continue

                results: list[WebSearchResult] = []
                for item in raw or []:
                    title = str(item.get("title", "") or "").strip()
                    url = str(item.get("href") or item.get("url") or "").strip()
                    snippet = str(item.get("body") or item.get("snippet") or "").strip()
                    if title or url:
                        results.append(
                            WebSearchResult(title=title, url=url, snippet=snippet)
                        )
                if results:
                    return results

        if last_exc is not None:
            raise last_exc
        return []

    async def search(
        self,
        query: str,
        *,
        max_results: int = 5,
        recency_days: int | None = None,
    ) -> list[WebSearchResult]:
        return await asyncio.wait_for(
            asyncio.to_thread(self._sync_search, query, max_results, recency_days),
            timeout=max(15.0, self.timeout_seconds * 2.0),
        )


class BraveSearchProvider(WebSearchProvider):
    """Brave Web Search REST API provider."""

    name = "brave"
    endpoint = "https://api.search.brave.com/res/v1/web/search"

    def __init__(
        self,
        api_key: str,
        *,
        timeout_seconds: float = 10.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.api_key = api_key.strip()
        self.timeout_seconds = timeout_seconds
        self._http_client = http_client

    @staticmethod
    def _recency_to_freshness(recency_days: int | None) -> str | None:
        if recency_days is None:
            return None
        if recency_days <= 1:
            return "pd"
        if recency_days <= 7:
            return "pw"
        if recency_days <= 31:
            return "pm"
        return "py"

    async def search(
        self,
        query: str,
        *,
        max_results: int = 5,
        recency_days: int | None = None,
    ) -> list[WebSearchResult]:
        if not self.api_key:
            raise ValueError("BRAVE_SEARCH_API_KEY is not configured in .env.")
        params: dict[str, Any] = {"q": query, "count": max_results}
        freshness = self._recency_to_freshness(recency_days)
        if freshness:
            params["freshness"] = freshness
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self.api_key,
            "User-Agent": _USER_AGENT,
        }
        data = await _http_get_json(
            self.endpoint,
            params=params,
            headers=headers,
            timeout=self.timeout_seconds,
            client=self._http_client,
        )
        raw_items = (data.get("web") or {}).get("results") or []
        return [
            WebSearchResult(
                title=str(item.get("title", "") or "").strip(),
                url=str(item.get("url", "") or "").strip(),
                snippet=str(item.get("description", "") or "").strip(),
                published_date=item.get("age") or item.get("page_age"),
            )
            for item in raw_items[:max_results]
        ]


class TavilySearchProvider(WebSearchProvider):
    """Tavily Search REST API provider (AI-snippet optimized)."""

    name = "tavily"
    endpoint = "https://api.tavily.com/search"

    def __init__(
        self,
        api_key: str,
        *,
        timeout_seconds: float = 10.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.api_key = api_key.strip()
        self.timeout_seconds = timeout_seconds
        self._http_client = http_client

    async def search(
        self,
        query: str,
        *,
        max_results: int = 5,
        recency_days: int | None = None,
    ) -> list[WebSearchResult]:
        if not self.api_key:
            raise ValueError("TAVILY_API_KEY is not configured in .env.")
        body: dict[str, Any] = {
            "api_key": self.api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",
            "include_answer": False,
        }
        if recency_days is not None:
            body["days"] = max(1, int(recency_days))
        data = await _http_post_json(
            self.endpoint,
            json_body=body,
            headers={"User-Agent": _USER_AGENT},
            timeout=self.timeout_seconds,
            client=self._http_client,
        )
        raw_items = data.get("results") or []
        return [
            WebSearchResult(
                title=str(item.get("title", "") or "").strip(),
                url=str(item.get("url", "") or "").strip(),
                snippet=str(item.get("content", "") or "").strip(),
                published_date=item.get("published_date"),
            )
            for item in raw_items[:max_results]
        ]


class SearXNGSearchProvider(WebSearchProvider):
    """Self-hosted SearXNG JSON API provider ($0.00, 100% local-first)."""

    name = "searxng"

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 10.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.strip().rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._http_client = http_client

    @staticmethod
    def _recency_to_time_range(recency_days: int | None) -> str | None:
        if recency_days is None:
            return None
        if recency_days <= 1:
            return "day"
        if recency_days <= 7:
            return "week"
        if recency_days <= 31:
            return "month"
        return "year"

    async def search(
        self,
        query: str,
        *,
        max_results: int = 5,
        recency_days: int | None = None,
    ) -> list[WebSearchResult]:
        if not self.base_url:
            raise ValueError("SEARXNG_URL is not configured in .env.")
        params: dict[str, Any] = {"q": query, "format": "json"}
        tr = self._recency_to_time_range(recency_days)
        if tr:
            params["time_range"] = tr
        data = await _http_get_json(
            f"{self.base_url}/search",
            params=params,
            headers={"Accept": "application/json", "User-Agent": _USER_AGENT},
            timeout=self.timeout_seconds,
            client=self._http_client,
        )
        raw_items = data.get("results") or []
        return [
            WebSearchResult(
                title=str(item.get("title", "") or "").strip(),
                url=str(item.get("url", "") or "").strip(),
                snippet=str(item.get("content", "") or "").strip(),
                published_date=item.get("publishedDate"),
            )
            for item in raw_items[:max_results]
        ]


def create_web_search_provider(settings: Settings) -> WebSearchProvider | None:
    """Instantiate the configured client-side search provider (None for anthropic/disabled)."""
    p = settings.web_search_provider
    if p in ("anthropic", "disabled"):
        return None
    if p == "ddg":
        return DDGSearchProvider(timeout_seconds=settings.web_search_timeout_seconds)
    if p == "brave":
        return BraveSearchProvider(
            settings.brave_search_api_key,
            timeout_seconds=settings.web_search_timeout_seconds,
        )
    if p == "tavily":
        return TavilySearchProvider(
            settings.tavily_api_key,
            timeout_seconds=settings.web_search_timeout_seconds,
        )
    if p == "searxng":
        return SearXNGSearchProvider(
            settings.searxng_url,
            timeout_seconds=settings.web_search_timeout_seconds,
        )
    raise ValueError(f"Unsupported WEB_SEARCH_PROVIDER: {p!r}")


def _truncate(text: str, limit: int) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(1, limit - 1)].rstrip() + "…"


def compute_search_cache_key(
    provider: str, query: str, max_results: int, recency_days: int | None
) -> str:
    canonical = json.dumps(
        {
            "provider": provider,
            "query": " ".join(query.strip().lower().split()),
            "max_results": max_results,
            "recency_days": recency_days,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def wrap_untrusted_web_content(
    body: str, *, tool_name: str, metadata: dict[str, Any]
) -> str:
    """Frame external web outputs explicitly as untrusted data to resist prompt injection."""
    attrs = " ".join(f'{k}="{v}"' for k, v in metadata.items())
    return (
        f"<untrusted_web_content tool=\"{tool_name}\" {attrs}>\n"
        f"{body}\n"
        f"</untrusted_web_content>"
    )


class WebSearchService:
    """Budget-safe, SQLite-cached web search orchestrator for client-side providers."""

    def __init__(
        self,
        *,
        store: AgentStore,
        circuit_breaker: AgentCircuitBreaker,
        settings: Settings,
        provider: WebSearchProvider | None = None,
    ) -> None:
        self.store = store
        self.circuit_breaker = circuit_breaker
        self.settings = settings
        self.provider = (
            provider if provider is not None else create_web_search_provider(settings)
        )

    async def search(
        self,
        query: str,
        *,
        max_results: int | None = None,
        recency_days: int | None = None,
        now: datetime | None = None,
    ) -> tuple[str, bool]:
        """Run a cached, quota-enforced web search returning (framed_output, is_error)."""
        cleaned_query = " ".join(query.strip().split())
        if not cleaned_query:
            return "Error: search query must not be empty.", True
        if self.provider is None:
            return (
                f"Web search client tool is inactive (WEB_SEARCH_PROVIDER={self.settings.web_search_provider}).",
                True,
            )

        count = max(
            1,
            min(
                self.settings.web_search_max_results,
                int(max_results or self.settings.web_search_max_results),
            ),
        )
        cache_key = compute_search_cache_key(
            self.provider.name, cleaned_query, count, recency_days
        )

        # 1. Check SQLite TTL cache first ($0, does not burn daily quota)
        cached_json = await self.store.get_cached_web_search(cache_key, now=now)
        if cached_json is not None:
            return (
                wrap_untrusted_web_content(
                    cached_json,
                    tool_name="web_search",
                    metadata={"provider": self.provider.name, "cached": "true"},
                ),
                False,
            )

        # 2. Enforce persistent daily search quota & daily USD cap
        try:
            await self.circuit_breaker.check_web_search_quota(now=now)
        except CircuitBreakerTripped as exc:
            return f"Web search blocked: {exc.detail}", True

        # 3. Execute live provider search
        try:
            raw_results = await self.provider.search(
                cleaned_query,
                max_results=count,
                recency_days=recency_days,
            )
        except Exception as exc:
            return f"Web search error ({self.provider.name}): {exc}", True

        # 4. Compact & truncate snippets to protect 16k fast_buffer budget
        compact: list[dict[str, Any]] = []
        for r in raw_results[:count]:
            entry = asdict(
                WebSearchResult(
                    title=_truncate(r.title, 160),
                    url=r.url.strip(),
                    snippet=_truncate(r.snippet, self.settings.web_search_max_snippet_chars),
                    published_date=r.published_date,
                )
            )
            compact.append(entry)

        payload_json = json.dumps(
            {
                "query": cleaned_query,
                "provider": self.provider.name,
                "result_count": len(compact),
                "results": compact,
            },
            ensure_ascii=False,
        )

        # 5. Record usage & cache cleanly
        await self.circuit_breaker.record_client_web_search(count=1, now=now)
        await self.store.put_cached_web_search(
            cache_key=cache_key,
            provider=self.provider.name,
            query=cleaned_query,
            results_json=payload_json,
            ttl_seconds=self.settings.web_search_cache_ttl_seconds,
            now=now,
        )

        return (
            wrap_untrusted_web_content(
                payload_json,
                tool_name="web_search",
                metadata={"provider": self.provider.name, "cached": "false"},
            ),
            False,
        )


# -----------------------------------------------------------------------------
# SSRF-Safe Webpage Fetcher (`fetch_webpage`)
# -----------------------------------------------------------------------------


def _is_blocked_ip(ip_str: str) -> bool:
    """Return True if an IP address is loopback, private, link-local, multicast, or reserved."""
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return True
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


async def validate_public_http_url(url: str) -> str:
    """Validate URL scheme and verify all resolved DNS IPs are public (SSRF protection)."""
    parsed = urlparse(url.strip())
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Unsupported URL scheme {parsed.scheme!r}; only http/https allowed.")
    hostname = parsed.hostname
    if not hostname:
        raise ValueError("URL must include a valid hostname.")

    lower_host = hostname.lower().rstrip(".")
    if lower_host in ("localhost", "localhost.localdomain") or lower_host.endswith(".local"):
        raise ValueError(f"Blocked private/local hostname: {hostname!r}.")

    # Direct IP literal check
    try:
        ipaddress.ip_address(lower_host)
    except ValueError:
        # Resolve hostname via DNS off the event loop
        try:
            addr_info = await asyncio.to_thread(
                socket.getaddrinfo, hostname, parsed.port or (443 if parsed.scheme == "https" else 80)
            )
        except socket.gaierror as exc:
            raise ValueError(f"DNS lookup failed for {hostname!r}: {exc}") from exc
        if not addr_info:
            raise ValueError(f"No DNS records resolved for {hostname!r}.")
        for entry in addr_info:
            resolved_ip = str(entry[4][0])
            if _is_blocked_ip(resolved_ip):
                raise ValueError(
                    f"SSRF blocked: {hostname!r} resolved to non-public IP {resolved_ip}."
                )
    else:
        if _is_blocked_ip(lower_host):
            raise ValueError(f"SSRF blocked: non-public IP address {hostname!r}.")

    return url.strip()


class _FallbackHTMLTextExtractor(HTMLParser):
    """Stdlib fallback HTML text extractor skipping script/style/nav/footer blocks."""

    _SKIP_TAGS = frozenset({"script", "style", "noscript", "svg", "nav", "footer", "head"})

    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self._chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in self._SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in self._SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            cleaned = data.strip()
            if cleaned:
                self._chunks.append(cleaned)

    def get_text(self) -> str:
        return "\n".join(self._chunks)


def extract_readable_text(html_content: str) -> str:
    """Strip scripts/styles/chrome and extract readable text from HTML."""
    try:
        from lxml import html as lxml_html

        doc = lxml_html.fromstring(html_content)
        for bad in doc.xpath("//script|//style|//noscript|//svg|//nav|//footer|//header"):
            parent = bad.getparent()
            if parent is not None:
                parent.remove(bad)
        title_nodes = doc.xpath("//title/text()")
        title = " ".join(title_nodes[0].split()) if title_nodes else ""
        body_nodes = doc.xpath("//main|//article|//body")
        target = body_nodes[0] if body_nodes else doc
        raw_text = target.text_content()
        lines = [" ".join(line.split()) for line in raw_text.splitlines()]
        non_empty = [line for line in lines if line]
        body_text = "\n".join(non_empty)
        return f"Title: {title}\n\n{body_text}" if title else body_text
    except Exception:
        parser = _FallbackHTMLTextExtractor()
        parser.feed(html_content)
        return re.sub(r"\n{3,}", "\n\n", parser.get_text())


async def safe_fetch_webpage(
    url: str,
    *,
    max_chars: int = 4000,
    timeout_seconds: float = 10.0,
    http_client: httpx.AsyncClient | None = None,
) -> tuple[str, bool]:
    """Fetch a public webpage with SSRF validation, redirect checks, and strict size capping."""
    try:
        current_url = await validate_public_http_url(url)
    except ValueError as exc:
        return f"Fetch blocked: {exc}", True

    owns_client = http_client is None
    client = http_client or httpx.AsyncClient(
        timeout=timeout_seconds,
        follow_redirects=False,
        headers={"User-Agent": _USER_AGENT, "Accept": "text/html,text/plain;q=0.9,*/*;q=0.1"},
    )
    try:
        resp: httpx.Response | None = None
        for _ in range(_MAX_REDIRECTS + 1):
            resp = await client.get(current_url)
            if resp.status_code in (301, 302, 303, 307, 308):
                location = resp.headers.get("location")
                if not location:
                    return "Fetch error: redirect missing Location header.", True
                next_url = urljoin(current_url, location)
                current_url = await validate_public_http_url(next_url)
                continue
            break

        if resp is None:
            return "Fetch error: no HTTP response received.", True
        if resp.status_code >= 400:
            return f"Fetch error: HTTP {resp.status_code} for {current_url}.", True

        content_type = resp.headers.get("content-type", "").lower()
        raw_bytes = resp.content[:_MAX_FETCH_BYTES]
        text_body = raw_bytes.decode(resp.encoding or "utf-8", errors="replace")

        if "html" in content_type or "<html" in text_body[:500].lower():
            extracted = extract_readable_text(text_body)
        else:
            extracted = text_body.strip()

        if len(extracted) > max_chars:
            extracted = extracted[: max_chars - 24].rstrip() + "\n...[truncated]..."

        return (
            wrap_untrusted_web_content(
                extracted,
                tool_name="fetch_webpage",
                metadata={"source_url": current_url},
            ),
            False,
        )
    except Exception as exc:
        return f"Fetch error: {exc}", True
    finally:
        if owns_client:
            await client.aclose()


async def _http_get_json(
    url: str,
    *,
    params: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
    client: httpx.AsyncClient | None,
) -> dict[str, Any]:
    if client is not None:
        resp = await client.get(url, params=params, headers=headers, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    async with httpx.AsyncClient(timeout=timeout) as c:
        resp = await c.get(url, params=params, headers=headers)
        resp.raise_for_status()
        return resp.json()


async def _http_post_json(
    url: str,
    *,
    json_body: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
    client: httpx.AsyncClient | None,
) -> dict[str, Any]:
    if client is not None:
        resp = await client.post(url, json=json_body, headers=headers, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    async with httpx.AsyncClient(timeout=timeout) as c:
        resp = await c.post(url, json=json_body, headers=headers)
        resp.raise_for_status()
        return resp.json()
