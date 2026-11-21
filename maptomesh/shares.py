"""Share links: a model's settings under a short id (``/?s=<id>``). Opening a link loads the
settings and builds the model again, on the visitor's device by default.

Only settings are stored, validated as ``Options``, never model files: a 3MF uploaded by one
user and opened by others in their slicer could carry a crafted file (slicers have had 3MF
parsing bugs). Settings are a few kB; identical settings share one id (a hash of them).

The database lives in ``config.DATA_DIR`` (links must outlive cache clean-ups). It holds places
people chose to share and the model names they typed: keep it out of version control.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import sqlite3
import threading
import time

from . import config, limits

DB_PATH = config.DATA_DIR / "shares.sqlite"
# Links kept: the newest-used KEEP, but a link used (created or opened) within STALE_DAYS is never
# pruned, so a burst of new links can't push out ones people still open; new links are capped per
# client and overall (PER_DAY) instead, which bounds how fast the table can grow.
KEEP = int(os.environ.get("MAPTOMESH_KEEP_SHARES", "200000"))
STALE_DAYS = float(os.environ.get("MAPTOMESH_SHARES_STALE_DAYS", "30"))
PER_HOUR = int(os.environ.get("MAPTOMESH_SHARES_PER_HOUR", "60"))   # new links per client address
PER_DAY = int(os.environ.get("MAPTOMESH_SHARES_PER_DAY", "5000"))   # new links from everyone
ID_CHARS = "abcdefghijklmnopqrstuvwxyz234567"


def valid_id(sid: str) -> bool:
    return 10 <= len(sid) <= 16 and all(c in ID_CHARS for c in sid)


def _canonical(options: dict) -> str:
    return json.dumps(options, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _ids(text: str):
    digest = base64.b32encode(hashlib.sha256(text.encode()).digest()).decode().lower()
    for n in (10, 12, 14, 16):        # longer only on a (practically impossible) collision
        yield digest[:n]


class ShareStore:
    def __init__(self, path=DB_PATH):
        self.path = str(path)
        self._lock = threading.Lock()
        self._added = 0
        with self._connect() as con:
            con.execute("""CREATE TABLE IF NOT EXISTS shares (
                id TEXT PRIMARY KEY, options_json TEXT NOT NULL, created REAL NOT NULL,
                used REAL NOT NULL, opens INTEGER NOT NULL DEFAULT 0)""")
            con.execute("CREATE INDEX IF NOT EXISTS shares_used ON shares (used)")

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        try:
            con.execute("PRAGMA journal_mode=WAL")
        except Exception:
            con.close()
            raise
        return _Closing(con)

    def add(self, options: dict) -> str:
        """The id for these settings (an existing one if they were shared before)."""
        text = _canonical(options)
        now = time.time()
        with self._lock, self._connect() as con:
            for sid in _ids(text):
                row = con.execute("SELECT options_json FROM shares WHERE id = ?", (sid,)).fetchone()
                if row is None:
                    con.execute("INSERT INTO shares (id, options_json, created, used) VALUES (?, ?, ?, ?)",
                                (sid, text, now, now))
                    self._added += 1
                    if self._added % 500 == 1:
                        self._prune(con)
                    return sid
                if row[0] == text:
                    con.execute("UPDATE shares SET used = ? WHERE id = ?", (now, sid))
                    return sid
        raise RuntimeError("share id collision")

    def get(self, sid: str, opened: bool = True) -> dict | None:
        if not valid_id(sid):
            return None
        with self._connect() as con:
            row = con.execute("SELECT options_json FROM shares WHERE id = ?", (sid,)).fetchone()
            if row is None:
                return None
            if opened:
                con.execute("UPDATE shares SET used = ?, opens = opens + 1 WHERE id = ?", (time.time(), sid))
        return json.loads(row[0])

    def _prune(self, con, now: float | None = None) -> None:
        """Forget the least recently used links beyond ``KEEP``, of those unused for ``STALE_DAYS``."""
        now = time.time() if now is None else now
        n = con.execute("SELECT COUNT(*) FROM shares").fetchone()[0]
        if n > KEEP:
            con.execute("DELETE FROM shares WHERE id IN (SELECT id FROM shares WHERE used < ? ORDER BY used LIMIT ?)",
                        (now - STALE_DAYS * 86400, n - KEEP))


class _Closing:
    """A connection that closes at the end of a ``with`` block (sqlite3's only commits)."""

    def __init__(self, con: sqlite3.Connection):
        self.con = con

    def __enter__(self) -> sqlite3.Connection:
        return self.con

    def __exit__(self, *exc) -> None:
        self.con.close()


_STORE: ShareStore | None = None
_STORE_LOCK = threading.Lock()
_LIMIT = limits.RateLimit(PER_HOUR)     # new links per client key (limits.client_key), per hour
_GLOBAL = limits.RateLimit(PER_DAY, window_s=86400, group_factor=0)


def store() -> ShareStore:
    global _STORE
    with _STORE_LOCK:
        if _STORE is None:
            _STORE = ShareStore()
        return _STORE


def rate_limited(addr: str) -> bool:
    """More than ``PER_HOUR`` new links from this client key in the last hour, or ``PER_DAY`` from
    everyone in the last day."""
    return _LIMIT.hit(addr) > 0 or (addr != limits.LOCAL and _GLOBAL.hit("all") > 0)
