"""Local inbox directory watcher input source."""

from __future__ import annotations

import hashlib
from pathlib import Path

from always_on.models import IncomingEvent
from always_on.sources.base import EventSource

_IGNORED_SUFFIXES = (".tmp", ".swp", ".part", ".crdownload")


class FileWatcherSource(EventSource):
    """Non-blocking watcher that scans a local inbox directory for new/changed files."""

    name = "file_watcher"

    def __init__(self, inbox_dir: Path, *, max_preview_chars: int = 2000) -> None:
        self.inbox_dir = inbox_dir
        self.max_preview_chars = max_preview_chars
        self._known_hashes: dict[str, str] = {}
        self._initialized = False

    async def start(self) -> None:
        self.inbox_dir.mkdir(parents=True, exist_ok=True)
        # Snapshot existing files on startup so old files don't storm the daemon at boot
        for path in self._iter_files():
            rel = path.relative_to(self.inbox_dir).as_posix()
            self._known_hashes[rel] = self._hash_file(path)
        self._initialized = True

    def _iter_files(self) -> list[Path]:
        if not self.inbox_dir.exists():
            return []
        result: list[Path] = []
        for p in sorted(self.inbox_dir.rglob("*")):
            if not p.is_file():
                continue
            if p.name.startswith(".") or p.name.endswith(_IGNORED_SUFFIXES):
                continue
            result.append(p)
        return result

    @staticmethod
    def _hash_file(path: Path) -> str:
        try:
            data = path.read_bytes()
        except OSError:
            return ""
        return hashlib.sha256(data).hexdigest()

    async def poll(self) -> list[IncomingEvent]:
        if not self._initialized:
            await self.start()
            return []

        events: list[IncomingEvent] = []
        current_files = self._iter_files()
        seen_rel: set[str] = set()

        for path in current_files:
            rel = path.relative_to(self.inbox_dir).as_posix()
            seen_rel.add(rel)
            file_hash = self._hash_file(path)
            if not file_hash:
                continue
            prev_hash = self._known_hashes.get(rel)
            if prev_hash == file_hash:
                continue

            change_kind = "created" if prev_hash is None else "modified"
            self._known_hashes[rel] = file_hash
            try:
                preview = path.read_text(encoding="utf-8", errors="replace")[
                    : self.max_preview_chars
                ]
            except OSError:
                preview = ""

            events.append(
                IncomingEvent(
                    source="file_watcher",
                    actor_id="filesystem",
                    content=f"Inbox file {change_kind}: {rel}\n{preview}".strip(),
                    payload={
                        "filename": rel,
                        "change": change_kind,
                        "sha256": file_hash,
                        "preview": preview,
                    },
                    priority="normal",
                    hard_override=False,
                )
            )

        # Forget deleted files so re-creating them later triggers a fresh event
        for old_rel in list(self._known_hashes.keys()):
            if old_rel not in seen_rel:
                del self._known_hashes[old_rel]

        return events
