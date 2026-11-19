"""User reviews of generated models: a score, reported problems, an optional comment and, only if the
user asks for a follow-up, contact details. Stored in SQLite together with the job's settings and a
snapshot of its result (stats, data sources, notes, errors, log tail) so a reported problem can be
reproduced later.

The database lives in ``config.DATA_DIR`` (not the cache, which may be wiped) and contains personal
data when users leave contact details: keep it out of version control and backups you share. No IP
addresses are stored. Anything rendered from it (the future admin page) must be HTML-escaped - every
text field is user input.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone

from . import config

DB_PATH = config.DATA_DIR / "reviews.sqlite"
SCHEMA_VERSION = 1

# problem categories (keys are stored; labels live in the UI)
PROBLEM_KEYS = ("terrain", "missing_data", "buildings", "water", "roads", "colors", "slicer", "slow_failed", "other")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS reviews (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT    NOT NULL,             -- ISO 8601 UTC
    job_id          TEXT    NOT NULL,
    job_state       TEXT,                         -- done | error | cancelled | unknown
    rating          INTEGER CHECK (rating IS NULL OR rating BETWEEN 1 AND 5),
    problems        TEXT    NOT NULL DEFAULT '[]', -- JSON list of PROBLEM_KEYS
    comment         TEXT    NOT NULL DEFAULT '',
    wants_followup  INTEGER NOT NULL DEFAULT 0,
    contact_name    TEXT,                         -- only when wants_followup
    contact_email   TEXT,                         -- only when wants_followup
    options_json    TEXT    NOT NULL,             -- the job's settings (Options)
    result_json     TEXT,                         -- stats, sources, notes, errors, log tail
    snapshot_source TEXT    NOT NULL DEFAULT 'server',  -- 'server' (job still in memory) or 'client'
    app_version     TEXT,
    user_agent      TEXT,
    status          TEXT    NOT NULL DEFAULT 'new',     -- for the admin page: new | read | resolved
    admin_note      TEXT    NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS reviews_created ON reviews (created_at);
CREATE INDEX IF NOT EXISTS reviews_job ON reviews (job_id);
CREATE INDEX IF NOT EXISTS reviews_status ON reviews (status);
"""


class ReviewStore:
    def __init__(self, path=DB_PATH):
        self.path = str(path)
        self._lock = threading.Lock()
        with self._connect() as con:
            con.executescript(_SCHEMA)
            if con.execute("PRAGMA user_version").fetchone()[0] < SCHEMA_VERSION:
                con.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path, timeout=30)
        con.execute("PRAGMA journal_mode=WAL")
        return con

    def add(self, *, job_id: str, job_state: str, rating: int | None, problems: list[str], comment: str,
            wants_followup: bool, contact_name: str | None, contact_email: str | None, options: dict,
            result: dict | None, snapshot_source: str, app_version: str, user_agent: str) -> int:
        row = (datetime.now(timezone.utc).isoformat(timespec="seconds"), job_id, job_state, rating,
               json.dumps(problems), comment, int(wants_followup),
               contact_name if wants_followup else None, contact_email if wants_followup else None,
               json.dumps(options, separators=(",", ":")),
               json.dumps(result, separators=(",", ":")) if result is not None else None,
               snapshot_source, app_version, user_agent)
        with self._lock, self._connect() as con:
            cur = con.execute(
                "INSERT INTO reviews (created_at, job_id, job_state, rating, problems, comment, wants_followup, "
                "contact_name, contact_email, options_json, result_json, snapshot_source, app_version, user_agent) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", row)
            return int(cur.lastrowid)

    def count_for_job(self, job_id: str) -> int:
        with self._connect() as con:
            return int(con.execute("SELECT COUNT(*) FROM reviews WHERE job_id = ?", (job_id,)).fetchone()[0])

    def recent(self, limit: int = 100, status: str | None = None) -> list[dict]:
        """Newest first, as dicts (for the admin report page)."""
        q = "SELECT * FROM reviews" + (" WHERE status = ?" if status else "") + " ORDER BY id DESC LIMIT ?"
        args = (status, limit) if status else (limit,)
        with self._connect() as con:
            con.row_factory = sqlite3.Row
            return [dict(r) for r in con.execute(q, args)]


_STORE: ReviewStore | None = None
_STORE_LOCK = threading.Lock()


def store() -> ReviewStore:
    global _STORE
    with _STORE_LOCK:
        if _STORE is None:
            _STORE = ReviewStore()
        return _STORE
