"""Async SQLite persistence layer with WAL mode and strict ACID transactions."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, AsyncIterator

import aiosqlite
import numpy as np

from always_on.embeddings.provider import bytes_to_vector, vector_to_bytes
from always_on.models import (
    ActivePriority,
    ArchivedMemoryItem,
    ConsolidationOperation,
    DailySpend,
    FastBufferEpisode,
    MemoryItem,
    TokenUsage,
    ToolCallRecord,
    WakeLogEntry,
    new_id,
    utc_now,
    utc_now_iso,
)

SCHEMA_PATH = Path(__file__).with_name("schema.sql")

DEFAULT_PRIORITIES: tuple[str, ...] = (
    "Respond helpfully and accurately to the user's active projects, tasks, and questions.",
    "Track important user preferences, workflow corrections, and recurring lessons.",
    "Surface urgent inbox files, alerts, or webhook notifications relevant to current goals.",
)


class AgentStore:
    """Persistent SQLite repository for memory, working buffer, spend, and breaker state."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self._conn: aiosqlite.Connection | None = None
        self._write_lock = asyncio.Lock()

    async def connect(self) -> None:
        """Open SQLite connection in WAL mode and apply DDL schema."""
        if self.db_path != Path(":memory:"):
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = await aiosqlite.connect(
            str(self.db_path),
            isolation_level=None,  # Explicit transaction control
        )
        conn.row_factory = aiosqlite.Row
        await conn.execute("PRAGMA journal_mode = WAL;")
        await conn.execute("PRAGMA synchronous = NORMAL;")
        await conn.execute("PRAGMA foreign_keys = ON;")
        await conn.execute("PRAGMA busy_timeout = 5000;")

        schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
        await conn.executescript(schema_sql)

        # Safe idempotent migration for pre-existing databases lacking daily_spend.web_searches
        async with conn.execute("PRAGMA table_info(daily_spend)") as cur:
            cols = {str(r["name"]) for r in await cur.fetchall()}
        if "web_searches" not in cols:
            await conn.execute(
                "ALTER TABLE daily_spend ADD COLUMN web_searches INTEGER NOT NULL DEFAULT 0"
            )

        # Ensure singleton circuit_breaker_state row exists
        now_iso = utc_now_iso()
        await conn.execute(
            """
            INSERT OR IGNORE INTO circuit_breaker_state (
                id, consecutive_errors, cooldown_until, last_sleep_at,
                last_wake_at, last_user_activity_at, updated_at
            ) VALUES (1, 0, NULL, ?, NULL, NULL, ?)
            """,
            (now_iso, now_iso),
        )
        self._conn = conn

    async def close(self) -> None:
        """Close the SQLite connection."""
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    @property
    def conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("AgentStore is not connected. Call connect() first.")
        return self._conn

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[aiosqlite.Connection]:
        """Acquire exclusive write lock and execute an atomic IMMEDIATE transaction."""
        async with self._write_lock:
            await self.conn.execute("BEGIN IMMEDIATE")
            try:
                yield self.conn
            except BaseException:
                await self.conn.execute("ROLLBACK")
                raise
            else:
                await self.conn.execute("COMMIT")

    # -------------------------------------------------------------------------
    # Core Memory & Archive
    # -------------------------------------------------------------------------

    async def list_core_memory(self) -> list[MemoryItem]:
        """Return all active Core Memory items in stable deterministic order (cache-friendly)."""
        async with self.conn.execute(
            """
            SELECT id, category, content, pinned, confidence, source_episodes,
                   version, created_at, updated_at
            FROM core_memory
            ORDER BY pinned DESC, category ASC, id ASC
            """
        ) as cur:
            rows = await cur.fetchall()
        return [self._row_to_memory_item(r) for r in rows]

    async def get_core_memory_map(self) -> dict[str, MemoryItem]:
        """Return active Core Memory keyed by item ID."""
        items = await self.list_core_memory()
        return {item.id: item for item in items}

    async def insert_core_memory_item(self, item: MemoryItem) -> MemoryItem:
        """Directly insert a Core Memory item (useful for seeding/admin/tests)."""
        async with self.transaction() as db:
            await db.execute(
                """
                INSERT INTO core_memory (
                    id, category, content, pinned, confidence,
                    source_episodes, version, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item.id,
                    item.category,
                    item.content,
                    1 if item.pinned else 0,
                    item.confidence,
                    json.dumps(item.source_episodes),
                    item.version,
                    item.created_at,
                    item.updated_at,
                ),
            )
        return item

    async def set_memory_pinned(self, memory_id: str, pinned: bool) -> MemoryItem | None:
        """Toggle pinned protection on an active Core Memory item."""
        now_iso = utc_now_iso()
        async with self.transaction() as db:
            cur = await db.execute(
                """
                UPDATE core_memory
                SET pinned = ?, updated_at = ?
                WHERE id = ?
                """,
                (1 if pinned else 0, now_iso, memory_id),
            )
            if cur.rowcount == 0:
                return None
        items = await self.get_core_memory_map()
        return items.get(memory_id)

    async def list_memory_archive(self, limit: int = 100) -> list[ArchivedMemoryItem]:
        """Return archived & superseded memory history newest-first."""
        async with self.conn.execute(
            """
            SELECT archive_id, memory_id, version, category, content, pinned,
                   confidence, source_episodes, action, reason, sleep_cycle_id, archived_at
            FROM memory_archive
            ORDER BY archive_id DESC
            LIMIT ?
            """,
            (limit,),
        ) as cur:
            rows = await cur.fetchall()
        return [
            ArchivedMemoryItem(
                archive_id=int(r["archive_id"]),
                memory_id=str(r["memory_id"]),
                version=int(r["version"]),
                category=r["category"],
                content=str(r["content"]),
                pinned=bool(r["pinned"]),
                confidence=float(r["confidence"]),
                source_episodes=json.loads(r["source_episodes"]),
                action=r["action"],
                reason=str(r["reason"]),
                sleep_cycle_id=str(r["sleep_cycle_id"]),
                archived_at=str(r["archived_at"]),
            )
            for r in rows
        ]

    # -------------------------------------------------------------------------
    # Active Priorities (Goal Relevance Embeddings)
    # -------------------------------------------------------------------------

    async def list_active_priorities(self) -> list[ActivePriority]:
        async with self.conn.execute(
            """
            SELECT id, rank, description, updated_at
            FROM active_priorities
            ORDER BY rank ASC, id ASC
            """
        ) as cur:
            rows = await cur.fetchall()
        return [
            ActivePriority(
                id=int(r["id"]),
                rank=int(r["rank"]),
                description=str(r["description"]),
                updated_at=str(r["updated_at"]),
            )
            for r in rows
        ]

    async def get_active_priority_vectors(self) -> list[tuple[ActivePriority, np.ndarray]]:
        async with self.conn.execute(
            """
            SELECT id, rank, description, embedding, updated_at
            FROM active_priorities
            ORDER BY rank ASC, id ASC
            """
        ) as cur:
            rows = await cur.fetchall()
        return [
            (
                ActivePriority(
                    id=int(r["id"]),
                    rank=int(r["rank"]),
                    description=str(r["description"]),
                    updated_at=str(r["updated_at"]),
                ),
                bytes_to_vector(r["embedding"]),
            )
            for r in rows
        ]

    async def replace_active_priorities(
        self, priorities: list[tuple[str, np.ndarray]]
    ) -> list[ActivePriority]:
        now_iso = utc_now_iso()
        async with self.transaction() as db:
            await db.execute("DELETE FROM active_priorities")
            for rank, (desc, vec) in enumerate(priorities, start=1):
                await db.execute(
                    """
                    INSERT INTO active_priorities (rank, description, embedding, updated_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (rank, desc, vector_to_bytes(vec), now_iso),
                )
        return await self.list_active_priorities()

    # -------------------------------------------------------------------------
    # Fast Working Buffer
    # -------------------------------------------------------------------------

    async def add_fast_buffer_episode(
        self, episode: FastBufferEpisode, embedding: np.ndarray
    ) -> FastBufferEpisode:
        async with self.transaction() as db:
            await db.execute(
                """
                INSERT INTO fast_buffer (
                    id, created_at, source, actor_id, event_payload,
                    salience_score, summary, thoughts, tool_calls,
                    agent_response, token_count, had_error, had_correction, embedding
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    episode.id,
                    episode.created_at,
                    episode.source,
                    episode.actor_id,
                    json.dumps(episode.event_payload),
                    episode.salience_score,
                    episode.summary,
                    json.dumps(episode.thoughts),
                    json.dumps([tc.model_dump() for tc in episode.tool_calls]),
                    episode.agent_response,
                    episode.token_count,
                    1 if episode.had_error else 0,
                    1 if episode.had_correction else 0,
                    vector_to_bytes(embedding),
                ),
            )
        return episode

    async def list_fast_buffer(self) -> list[FastBufferEpisode]:
        async with self.conn.execute(
            """
            SELECT id, created_at, source, actor_id, event_payload,
                   salience_score, summary, thoughts, tool_calls,
                   agent_response, token_count, had_error, had_correction
            FROM fast_buffer
            ORDER BY created_at ASC, id ASC
            """
        ) as cur:
            rows = await cur.fetchall()
        return [self._row_to_episode(r) for r in rows]

    async def list_fast_buffer_with_vectors(
        self, limit_recent: int | None = None
    ) -> list[tuple[FastBufferEpisode, np.ndarray]]:
        if limit_recent is not None:
            query = """
                SELECT * FROM (
                    SELECT id, created_at, source, actor_id, event_payload,
                           salience_score, summary, thoughts, tool_calls,
                           agent_response, token_count, had_error, had_correction, embedding
                    FROM fast_buffer
                    ORDER BY created_at DESC, id DESC
                    LIMIT ?
                ) ORDER BY created_at ASC, id ASC
            """
            params: tuple[Any, ...] = (limit_recent,)
        else:
            query = """
                SELECT id, created_at, source, actor_id, event_payload,
                       salience_score, summary, thoughts, tool_calls,
                       agent_response, token_count, had_error, had_correction, embedding
                FROM fast_buffer
                ORDER BY created_at ASC, id ASC
            """
            params = ()
        async with self.conn.execute(query, params) as cur:
            rows = await cur.fetchall()
        return [(self._row_to_episode(r), bytes_to_vector(r["embedding"])) for r in rows]

    async def get_fast_buffer_ids(self) -> set[str]:
        async with self.conn.execute("SELECT id FROM fast_buffer") as cur:
            rows = await cur.fetchall()
        return {str(r["id"]) for r in rows}

    # -------------------------------------------------------------------------
    # Transactional Sleep & Consolidation
    # -------------------------------------------------------------------------

    async def create_sleep_cycle(
        self,
        sleep_cycle_id: str,
        trigger_reason: str,
        fatigue_before: float,
        episodes_count: int,
    ) -> None:
        now_iso = utc_now_iso()
        async with self.transaction() as db:
            await db.execute(
                """
                INSERT INTO sleep_cycles (
                    id, started_at, completed_at, status, trigger_reason,
                    fatigue_before, episodes_count, ops_applied, abort_reason
                ) VALUES (?, ?, NULL, 'running', ?, ?, ?, '{}', NULL)
                """,
                (sleep_cycle_id, now_iso, trigger_reason, fatigue_before, episodes_count),
            )

    async def mark_sleep_cycle_aborted_or_failed(
        self,
        sleep_cycle_id: str,
        status: str,
        reason: str,
        ops_summary: dict[str, Any] | None = None,
    ) -> None:
        now_iso = utc_now_iso()
        async with self.transaction() as db:
            await db.execute(
                """
                UPDATE sleep_cycles
                SET completed_at = ?, status = ?, ops_applied = ?, abort_reason = ?
                WHERE id = ?
                """,
                (now_iso, status, json.dumps(ops_summary or {}), reason, sleep_cycle_id),
            )

    async def apply_consolidation_transaction(
        self,
        *,
        sleep_cycle_id: str,
        validated_ops: list[ConsolidationOperation],
        priorities_with_vectors: list[tuple[str, np.ndarray]],
        episode_ids_to_flush: list[str],
        ops_summary: dict[str, Any],
    ) -> tuple[list[str], list[str], list[str]]:
        """Atomically apply Core Memory diffs, soft-archives, priorities, and buffer flush.

        Any exception rolls back the entire consolidation cleanly, leaving Core Memory
        and fast_buffer completely unchanged.
        """
        now_iso = utc_now_iso()
        added_ids: list[str] = []
        updated_ids: list[str] = []
        archived_ids: list[str] = []

        async with self.transaction() as db:
            for op in validated_ops:
                if op.op == "ADD":
                    mem_id = new_id("mem")
                    assert op.category is not None and op.content is not None
                    await db.execute(
                        """
                        INSERT INTO core_memory (
                            id, category, content, pinned, confidence,
                            source_episodes, version, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?)
                        """,
                        (
                            mem_id,
                            op.category,
                            op.content,
                            1 if op.pinned else 0,
                            op.confidence,
                            json.dumps(op.source_episode_ids),
                            now_iso,
                            now_iso,
                        ),
                    )
                    added_ids.append(mem_id)

                elif op.op == "UPDATE":
                    assert op.target_id is not None and op.content is not None
                    async with db.execute(
                        "SELECT * FROM core_memory WHERE id = ?",
                        (op.target_id,),
                    ) as cur:
                        existing = await cur.fetchone()
                    if existing is None:
                        raise ValueError(f"UPDATE target {op.target_id} vanished mid-tx")
                    if bool(existing["pinned"]):
                        raise ValueError(f"Pinned protection violated for {op.target_id}")

                    # Soft-archive superseded version
                    await db.execute(
                        """
                        INSERT INTO memory_archive (
                            memory_id, version, category, content, pinned,
                            confidence, source_episodes, action, reason,
                            sleep_cycle_id, archived_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, 'UPDATE_SUPERSEDED', ?, ?, ?)
                        """,
                        (
                            existing["id"],
                            existing["version"],
                            existing["category"],
                            existing["content"],
                            existing["pinned"],
                            existing["confidence"],
                            existing["source_episodes"],
                            op.reason or "Updated during sleep consolidation",
                            sleep_cycle_id,
                            now_iso,
                        ),
                    )

                    merged_sources = list(
                        dict.fromkeys(
                            json.loads(existing["source_episodes"]) + op.source_episode_ids
                        )
                    )
                    new_category = op.category or existing["category"]
                    await db.execute(
                        """
                        UPDATE core_memory
                        SET category = ?, content = ?, confidence = ?,
                            source_episodes = ?, version = version + 1, updated_at = ?
                        WHERE id = ?
                        """,
                        (
                            new_category,
                            op.content,
                            op.confidence,
                            json.dumps(merged_sources),
                            now_iso,
                            op.target_id,
                        ),
                    )
                    updated_ids.append(op.target_id)

                elif op.op == "ARCHIVE":
                    assert op.target_id is not None
                    async with db.execute(
                        "SELECT * FROM core_memory WHERE id = ?",
                        (op.target_id,),
                    ) as cur:
                        existing = await cur.fetchone()
                    if existing is None:
                        raise ValueError(f"ARCHIVE target {op.target_id} vanished mid-tx")
                    if bool(existing["pinned"]):
                        raise ValueError(f"Pinned protection violated for {op.target_id}")

                    # Soft-archive into memory_archive (never hard-delete without audit)
                    await db.execute(
                        """
                        INSERT INTO memory_archive (
                            memory_id, version, category, content, pinned,
                            confidence, source_episodes, action, reason,
                            sleep_cycle_id, archived_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, 'ARCHIVE', ?, ?, ?)
                        """,
                        (
                            existing["id"],
                            existing["version"],
                            existing["category"],
                            existing["content"],
                            existing["pinned"],
                            existing["confidence"],
                            existing["source_episodes"],
                            op.reason or "Archived during sleep consolidation",
                            sleep_cycle_id,
                            now_iso,
                        ),
                    )
                    await db.execute(
                        "DELETE FROM core_memory WHERE id = ?",
                        (op.target_id,),
                    )
                    archived_ids.append(op.target_id)

            # Refresh active priorities if provided
            if priorities_with_vectors:
                await db.execute("DELETE FROM active_priorities")
                for rank, (desc, vec) in enumerate(priorities_with_vectors, start=1):
                    await db.execute(
                        """
                        INSERT INTO active_priorities (rank, description, embedding, updated_at)
                        VALUES (?, ?, ?, ?)
                        """,
                        (rank, desc, vector_to_bytes(vec), now_iso),
                    )

            # Flush ONLY the episodes processed in this sleep pass
            if episode_ids_to_flush:
                placeholders = ",".join("?" for _ in episode_ids_to_flush)
                await db.execute(
                    f"DELETE FROM fast_buffer WHERE id IN ({placeholders})",
                    tuple(episode_ids_to_flush),
                )

            # Reset last_sleep_at
            await db.execute(
                """
                UPDATE circuit_breaker_state
                SET last_sleep_at = ?, updated_at = ?
                WHERE id = 1
                """,
                (now_iso, now_iso),
            )

            # Finalize sleep cycle record
            final_summary = {
                **ops_summary,
                "added_ids": added_ids,
                "updated_ids": updated_ids,
                "archived_ids": archived_ids,
            }
            await db.execute(
                """
                UPDATE sleep_cycles
                SET completed_at = ?, status = 'committed', ops_applied = ?, abort_reason = NULL
                WHERE id = ?
                """,
                (now_iso, json.dumps(final_summary), sleep_cycle_id),
            )

        return added_ids, updated_ids, archived_ids

    async def list_sleep_cycles(self, limit: int = 30) -> list[dict[str, Any]]:
        async with self.conn.execute(
            """
            SELECT id, started_at, completed_at, status, trigger_reason,
                   fatigue_before, episodes_count, ops_applied, abort_reason
            FROM sleep_cycles
            ORDER BY started_at DESC
            LIMIT ?
            """,
            (limit,),
        ) as cur:
            rows = await cur.fetchall()
        return [
            {
                "id": str(r["id"]),
                "started_at": str(r["started_at"]),
                "completed_at": r["completed_at"],
                "status": str(r["status"]),
                "trigger_reason": str(r["trigger_reason"]),
                "fatigue_before": float(r["fatigue_before"]),
                "episodes_count": int(r["episodes_count"]),
                "ops_applied": json.loads(r["ops_applied"]),
                "abort_reason": r["abort_reason"],
            }
            for r in rows
        ]

    # -------------------------------------------------------------------------
    # Circuit Breaker, Spend & Wake Log
    # -------------------------------------------------------------------------

    async def get_daily_spend(self, date_utc: str | None = None) -> DailySpend:
        day = date_utc or utc_now().strftime("%Y-%m-%d")
        async with self.conn.execute(
            """
            SELECT date_utc, input_tokens, cache_read_tokens,
                   cache_creation_tokens, output_tokens, web_searches, usd_spent, updated_at
            FROM daily_spend
            WHERE date_utc = ?
            """,
            (day,),
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return DailySpend(date_utc=day)
        return DailySpend(
            date_utc=str(row["date_utc"]),
            input_tokens=int(row["input_tokens"]),
            cache_read_tokens=int(row["cache_read_tokens"]),
            cache_creation_tokens=int(row["cache_creation_tokens"]),
            output_tokens=int(row["output_tokens"]),
            web_searches=int(row["web_searches"]),
            usd_spent=float(row["usd_spent"]),
            updated_at=str(row["updated_at"]),
        )

    async def record_spend(
        self, usage: TokenUsage, usd_cost: float, date_utc: str | None = None
    ) -> DailySpend:
        day = date_utc or utc_now().strftime("%Y-%m-%d")
        now_iso = utc_now_iso()
        async with self.transaction() as db:
            await db.execute(
                """
                INSERT INTO daily_spend (
                    date_utc, input_tokens, cache_read_tokens,
                    cache_creation_tokens, output_tokens, web_searches, usd_spent, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(date_utc) DO UPDATE SET
                    input_tokens = input_tokens + excluded.input_tokens,
                    cache_read_tokens = cache_read_tokens + excluded.cache_read_tokens,
                    cache_creation_tokens = cache_creation_tokens + excluded.cache_creation_tokens,
                    output_tokens = output_tokens + excluded.output_tokens,
                    web_searches = web_searches + excluded.web_searches,
                    usd_spent = usd_spent + excluded.usd_spent,
                    updated_at = excluded.updated_at
                """,
                (
                    day,
                    usage.input_tokens,
                    usage.cache_read_tokens,
                    usage.cache_creation_tokens,
                    usage.output_tokens,
                    usage.web_search_requests,
                    usd_cost,
                    now_iso,
                ),
            )
        return await self.get_daily_spend(day)

    async def record_web_search(
        self,
        count: int = 1,
        usd_cost: float = 0.0,
        date_utc: str | None = None,
    ) -> DailySpend:
        """Increment persistent daily web search counter (and optional USD cost)."""
        day = date_utc or utc_now().strftime("%Y-%m-%d")
        now_iso = utc_now_iso()
        async with self.transaction() as db:
            await db.execute(
                """
                INSERT INTO daily_spend (
                    date_utc, input_tokens, cache_read_tokens,
                    cache_creation_tokens, output_tokens, web_searches, usd_spent, updated_at
                ) VALUES (?, 0, 0, 0, 0, ?, ?, ?)
                ON CONFLICT(date_utc) DO UPDATE SET
                    web_searches = web_searches + excluded.web_searches,
                    usd_spent = usd_spent + excluded.usd_spent,
                    updated_at = excluded.updated_at
                """,
                (day, max(0, int(count)), usd_cost, now_iso),
            )
        return await self.get_daily_spend(day)

    # -------------------------------------------------------------------------
    # Local Web Search TTL Cache
    # -------------------------------------------------------------------------

    async def get_cached_web_search(
        self, cache_key: str, *, now: datetime | None = None
    ) -> str | None:
        """Return cached JSON search results if present and unexpired."""
        current_iso = (now or utc_now()).isoformat()
        async with self.conn.execute(
            """
            SELECT results_json, expires_at
            FROM web_search_cache
            WHERE cache_key = ?
            """,
            (cache_key,),
        ) as cur:
            row = await cur.fetchone()
        if row is None or str(row["expires_at"]) <= current_iso:
            return None
        return str(row["results_json"])

    async def put_cached_web_search(
        self,
        *,
        cache_key: str,
        provider: str,
        query: str,
        results_json: str,
        ttl_seconds: int,
        now: datetime | None = None,
    ) -> None:
        """Persist search results in SQLite TTL cache and prune expired rows."""
        if ttl_seconds <= 0:
            return
        current = now or utc_now()
        now_iso = current.isoformat()
        expires_iso = (current + timedelta(seconds=ttl_seconds)).isoformat()
        async with self.transaction() as db:
            await db.execute("DELETE FROM web_search_cache WHERE expires_at <= ?", (now_iso,))
            await db.execute(
                """
                INSERT INTO web_search_cache (
                    cache_key, provider, query, results_json, created_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    provider = excluded.provider,
                    query = excluded.query,
                    results_json = excluded.results_json,
                    created_at = excluded.created_at,
                    expires_at = excluded.expires_at
                """,
                (cache_key, provider, query, results_json, now_iso, expires_iso),
            )

    async def get_breaker_state_row(self) -> dict[str, Any]:
        async with self.conn.execute(
            """
            SELECT consecutive_errors, cooldown_until, last_sleep_at,
                   last_wake_at, last_user_activity_at, updated_at
            FROM circuit_breaker_state
            WHERE id = 1
            """
        ) as cur:
            row = await cur.fetchone()
        assert row is not None
        return dict(row)

    async def update_breaker_error(
        self, consecutive_errors: int, cooldown_until: str | None
    ) -> None:
        now_iso = utc_now_iso()
        async with self.transaction() as db:
            await db.execute(
                """
                UPDATE circuit_breaker_state
                SET consecutive_errors = ?, cooldown_until = ?, updated_at = ?
                WHERE id = 1
                """,
                (consecutive_errors, cooldown_until, now_iso),
            )

    async def reset_breaker_errors(self) -> None:
        now_iso = utc_now_iso()
        async with self.transaction() as db:
            await db.execute(
                """
                UPDATE circuit_breaker_state
                SET consecutive_errors = 0, cooldown_until = NULL, updated_at = ?
                WHERE id = 1
                """,
                (now_iso,),
            )

    async def touch_last_wake(self, when: datetime | None = None) -> None:
        ts = (when or utc_now()).isoformat()
        async with self.transaction() as db:
            await db.execute(
                """
                UPDATE circuit_breaker_state
                SET last_wake_at = ?, updated_at = ?
                WHERE id = 1
                """,
                (ts, ts),
            )

    async def touch_last_user_activity(self, when: datetime | None = None) -> None:
        ts = (when or utc_now()).isoformat()
        async with self.transaction() as db:
            await db.execute(
                """
                UPDATE circuit_breaker_state
                SET last_user_activity_at = ?, updated_at = ?
                WHERE id = 1
                """,
                (ts, ts),
            )

    async def set_last_sleep_at(self, when: datetime) -> None:
        ts = when.isoformat()
        async with self.transaction() as db:
            await db.execute(
                """
                UPDATE circuit_breaker_state
                SET last_sleep_at = ?, updated_at = ?
                WHERE id = 1
                """,
                (ts, utc_now_iso()),
            )

    async def log_wake(
        self,
        *,
        source: str,
        salience: float,
        effort: str,
        usd_cost: float,
        steps: int,
        status: str,
        timestamp: datetime | None = None,
    ) -> None:
        ts = (timestamp or utc_now()).isoformat()
        async with self.transaction() as db:
            await db.execute(
                """
                INSERT INTO wake_log (timestamp, source, salience, effort, usd_cost, steps, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (ts, source, salience, effort, usd_cost, steps, status),
            )

    async def count_wakes_since(self, since: datetime) -> int:
        async with self.conn.execute(
            "SELECT COUNT(*) AS cnt FROM wake_log WHERE timestamp >= ?",
            (since.isoformat(),),
        ) as cur:
            row = await cur.fetchone()
        return int(row["cnt"]) if row else 0

    async def list_recent_wakes(self, limit: int = 30) -> list[WakeLogEntry]:
        async with self.conn.execute(
            """
            SELECT id, timestamp, source, salience, effort, usd_cost, steps, status
            FROM wake_log
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ) as cur:
            rows = await cur.fetchall()
        return [
            WakeLogEntry(
                id=int(r["id"]),
                timestamp=str(r["timestamp"]),
                source=str(r["source"]),
                salience=float(r["salience"]),
                effort=str(r["effort"]),
                usd_cost=float(r["usd_cost"]),
                steps=int(r["steps"]),
                status=str(r["status"]),
            )
            for r in rows
        ]

    # -------------------------------------------------------------------------
    # Event Deduplication Cooldown
    # -------------------------------------------------------------------------

    async def is_duplicate_payload(
        self, payload_hash: str, cooldown_seconds: int, now: datetime | None = None
    ) -> bool:
        """Check whether payload_hash was seen within cooldown_seconds; record if fresh."""
        current = now or utc_now()
        cutoff = (current - timedelta(seconds=cooldown_seconds)).isoformat()
        now_iso = current.isoformat()

        async with self.transaction() as db:
            # Prune expired rows lazily
            await db.execute("DELETE FROM event_dedup WHERE seen_at < ?", (cutoff,))
            async with db.execute(
                "SELECT seen_at FROM event_dedup WHERE payload_hash = ?",
                (payload_hash,),
            ) as cur:
                row = await cur.fetchone()
            if row is not None and str(row["seen_at"]) >= cutoff:
                return True
            await db.execute(
                """
                INSERT INTO event_dedup (payload_hash, seen_at)
                VALUES (?, ?)
                ON CONFLICT(payload_hash) DO UPDATE SET seen_at = excluded.seen_at
                """,
                (payload_hash, now_iso),
            )
            return False

    # -------------------------------------------------------------------------
    # Persistent Chat Transcript (Shared across Dashboard, CLI & Discord)
    # -------------------------------------------------------------------------

    async def append_chat_message(self, entry: dict[str, Any]) -> dict[str, Any]:
        """Persist a chat turn or proactive alert in SQLite and return copy with row ID."""
        timestamp = str(entry.get("timestamp") or utc_now_iso())
        role = str(entry["role"])
        actor_id = str(entry.get("actor_id") or ("user" if role == "user" else "agent"))
        source = str(entry.get("source") or "chat")
        trigger_actor = entry.get("trigger_actor")
        trigger_preview = entry.get("trigger_preview")
        episode_id = entry.get("episode_id")
        content = str(entry.get("content") or "")
        thoughts = list(entry.get("thoughts") or [])
        tool_calls = list(entry.get("tool_calls") or [])
        is_correction = bool(entry.get("is_correction", False))
        salience = entry.get("salience")
        status = str(entry.get("status") or "ok")
        usd_cost = float(entry.get("usd_cost") or 0.0)

        async with self.transaction() as db:
            cur = await db.execute(
                """
                INSERT INTO chat_messages (
                    timestamp, role, actor_id, source, trigger_actor, trigger_preview,
                    episode_id, content, thoughts, tool_calls, is_correction,
                    salience, status, usd_cost
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp,
                    role,
                    actor_id,
                    source,
                    trigger_actor,
                    trigger_preview,
                    episode_id,
                    content,
                    json.dumps(thoughts),
                    json.dumps(tool_calls),
                    1 if is_correction else 0,
                    salience,
                    status,
                    usd_cost,
                ),
            )
            row_id = cur.lastrowid

        saved = dict(entry)
        saved["id"] = row_id
        saved["timestamp"] = timestamp
        return saved

    async def list_chat_messages(self, limit: int = 100) -> list[dict[str, Any]]:
        """Return the latest `limit` persisted chat messages oldest-first."""
        async with self.conn.execute(
            """
            SELECT * FROM (
                SELECT id, timestamp, role, actor_id, source, trigger_actor, trigger_preview,
                       episode_id, content, thoughts, tool_calls, is_correction,
                       salience, status, usd_cost
                FROM chat_messages
                ORDER BY id DESC
                LIMIT ?
            ) ORDER BY id ASC
            """,
            (max(1, int(limit)),),
        ) as cur:
            rows = await cur.fetchall()

        result: list[dict[str, Any]] = []
        for r in rows:
            item: dict[str, Any] = {
                "id": int(r["id"]),
                "timestamp": str(r["timestamp"]),
                "role": str(r["role"]),
                "actor_id": str(r["actor_id"]),
                "source": str(r["source"]),
                "content": str(r["content"]),
                "status": str(r["status"]),
                "usd_cost": float(r["usd_cost"]),
            }
            if r["trigger_actor"] is not None:
                item["trigger_actor"] = str(r["trigger_actor"])
            if r["trigger_preview"] is not None:
                item["trigger_preview"] = str(r["trigger_preview"])
            if r["episode_id"] is not None:
                item["episode_id"] = str(r["episode_id"])
            if r["salience"] is not None:
                item["salience"] = float(r["salience"])
            if r["role"] == "user":
                item["is_correction"] = bool(r["is_correction"])
            else:
                item["thoughts"] = json.loads(r["thoughts"])
                item["tool_calls"] = json.loads(r["tool_calls"])
            result.append(item)
        return result

    async def clear_chat_messages(self) -> int:
        """Delete all rows from chat_messages without touching Core Memory or fast_buffer."""
        async with self.transaction() as db:
            cur = await db.execute("DELETE FROM chat_messages")
            return int(cur.rowcount or 0)

    # -------------------------------------------------------------------------
    # Internal row mappers
    # -------------------------------------------------------------------------

    @staticmethod
    def _row_to_memory_item(row: aiosqlite.Row) -> MemoryItem:
        return MemoryItem(
            id=str(row["id"]),
            category=row["category"],
            content=str(row["content"]),
            pinned=bool(row["pinned"]),
            confidence=float(row["confidence"]),
            source_episodes=json.loads(row["source_episodes"]),
            version=int(row["version"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

    @staticmethod
    def _row_to_episode(row: aiosqlite.Row) -> FastBufferEpisode:
        raw_tools = json.loads(row["tool_calls"])
        return FastBufferEpisode(
            id=str(row["id"]),
            created_at=str(row["created_at"]),
            source=row["source"],
            actor_id=str(row["actor_id"]),
            event_payload=json.loads(row["event_payload"]),
            salience_score=float(row["salience_score"]),
            summary=str(row["summary"]),
            thoughts=json.loads(row["thoughts"]),
            tool_calls=[ToolCallRecord(**tc) for tc in raw_tools],
            agent_response=str(row["agent_response"]),
            token_count=int(row["token_count"]),
            had_error=bool(row["had_error"]),
            had_correction=bool(row["had_correction"]),
        )


def parse_iso(value: str | None) -> datetime | None:
    """Parse an ISO-8601 timestamp into timezone-aware UTC datetime."""
    if not value:
        return None
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)
