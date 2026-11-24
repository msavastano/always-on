-- Always-On CLS AI Agent SQLite Schema (WAL Mode)

CREATE TABLE IF NOT EXISTS core_memory (
    id TEXT PRIMARY KEY,
    category TEXT NOT NULL CHECK(category IN ('fact', 'preference', 'project', 'lesson')),
    content TEXT NOT NULL,
    pinned INTEGER NOT NULL DEFAULT 0 CHECK(pinned IN (0, 1)),
    confidence REAL NOT NULL DEFAULT 0.8 CHECK(confidence >= 0.0 AND confidence <= 1.0),
    source_episodes TEXT NOT NULL DEFAULT '[]',
    version INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_core_memory_category ON core_memory(category);
CREATE INDEX IF NOT EXISTS idx_core_memory_pinned ON core_memory(pinned);

CREATE TABLE IF NOT EXISTS memory_archive (
    archive_id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    category TEXT NOT NULL CHECK(category IN ('fact', 'preference', 'project', 'lesson')),
    content TEXT NOT NULL,
    pinned INTEGER NOT NULL DEFAULT 0,
    confidence REAL NOT NULL,
    source_episodes TEXT NOT NULL DEFAULT '[]',
    action TEXT NOT NULL CHECK(action IN ('UPDATE_SUPERSEDED', 'ARCHIVE')),
    reason TEXT NOT NULL,
    sleep_cycle_id TEXT NOT NULL,
    archived_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_memory_archive_memory_id ON memory_archive(memory_id);
CREATE INDEX IF NOT EXISTS idx_memory_archive_archived_at ON memory_archive(archived_at DESC);

CREATE TABLE IF NOT EXISTS active_priorities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rank INTEGER NOT NULL,
    description TEXT NOT NULL,
    embedding BLOB NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS fast_buffer (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    source TEXT NOT NULL CHECK(source IN ('chat', 'file_watcher', 'webhook', 'heartbeat')),
    actor_id TEXT NOT NULL,
    event_payload TEXT NOT NULL DEFAULT '{}',
    salience_score REAL NOT NULL CHECK(salience_score >= 0.0 AND salience_score <= 1.0),
    summary TEXT NOT NULL,
    thoughts TEXT NOT NULL DEFAULT '[]',
    tool_calls TEXT NOT NULL DEFAULT '[]',
    agent_response TEXT NOT NULL DEFAULT '',
    token_count INTEGER NOT NULL DEFAULT 0,
    had_error INTEGER NOT NULL DEFAULT 0 CHECK(had_error IN (0, 1)),
    had_correction INTEGER NOT NULL DEFAULT 0 CHECK(had_correction IN (0, 1)),
    embedding BLOB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_fast_buffer_created_at ON fast_buffer(created_at ASC);

CREATE TABLE IF NOT EXISTS sleep_cycles (
    id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    status TEXT NOT NULL CHECK(status IN ('running', 'committed', 'aborted', 'failed')),
    trigger_reason TEXT NOT NULL,
    fatigue_before REAL NOT NULL,
    episodes_count INTEGER NOT NULL,
    ops_applied TEXT NOT NULL DEFAULT '{}',
    abort_reason TEXT
);

CREATE INDEX IF NOT EXISTS idx_sleep_cycles_started_at ON sleep_cycles(started_at DESC);

CREATE TABLE IF NOT EXISTS daily_spend (
    date_utc TEXT PRIMARY KEY,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    cache_read_tokens INTEGER NOT NULL DEFAULT 0,
    cache_creation_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    web_searches INTEGER NOT NULL DEFAULT 0,
    usd_spent REAL NOT NULL DEFAULT 0.0,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS web_search_cache (
    cache_key TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    query TEXT NOT NULL,
    results_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_web_search_cache_expires ON web_search_cache(expires_at ASC);

CREATE TABLE IF NOT EXISTS circuit_breaker_state (
    id INTEGER PRIMARY KEY CHECK(id = 1),
    consecutive_errors INTEGER NOT NULL DEFAULT 0,
    cooldown_until TEXT,
    last_sleep_at TEXT NOT NULL,
    last_wake_at TEXT,
    last_user_activity_at TEXT,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS wake_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    source TEXT NOT NULL,
    salience REAL NOT NULL,
    effort TEXT NOT NULL,
    usd_cost REAL NOT NULL DEFAULT 0.0,
    steps INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_wake_log_timestamp ON wake_log(timestamp DESC);

CREATE TABLE IF NOT EXISTS event_dedup (
    payload_hash TEXT PRIMARY KEY,
    seen_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_event_dedup_seen_at ON event_dedup(seen_at ASC);

CREATE TABLE IF NOT EXISTS chat_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
    actor_id TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'chat',
    trigger_actor TEXT,
    trigger_preview TEXT,
    episode_id TEXT,
    content TEXT NOT NULL,
    thoughts TEXT NOT NULL DEFAULT '[]',
    tool_calls TEXT NOT NULL DEFAULT '[]',
    is_correction INTEGER NOT NULL DEFAULT 0 CHECK(is_correction IN (0, 1)),
    salience REAL,
    status TEXT NOT NULL DEFAULT 'ok',
    usd_cost REAL NOT NULL DEFAULT 0.0
);

CREATE INDEX IF NOT EXISTS idx_chat_messages_id ON chat_messages(id DESC);
