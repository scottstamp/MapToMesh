"""Builds that failed on a user's device (the page reports them, then offers to build on the
server): kept for review, to find what breaks in the browser build.

Each report holds the kind of failure, the error and the end of its traceback, the last progress
message, the model's settings, and facts about the browser (memory estimate, cores, phone or not,
whether Python had started and with which features, the last log lines) plus its user-agent
string. No IP addresses. The database lives in ``config.DATA_DIR`` (not the cache); keep it out of
version control.

Review the latest reports::

    python -m maptomesh.device_failures [N]
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
import threading
import time
from datetime import datetime, timezone

from . import config, limits

DB_PATH = config.DATA_DIR / "device_failures.sqlite"
PER_HOUR = 30                 # reports per client address
GLOBAL_PER_HOUR = 300         # reports from everyone: turning the whole store over takes days
KEEP = 20_000                 # newest reports kept

_SCHEMA = """
CREATE TABLE IF NOT EXISTS failures (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at   TEXT NOT NULL,          -- ISO 8601 UTC
    kind         TEXT NOT NULL,          -- memory | too_large | startup | error
    message      TEXT NOT NULL,
    trace        TEXT NOT NULL DEFAULT '',
    stage        TEXT NOT NULL DEFAULT '',   -- last progress message before the failure
    options_json TEXT,                   -- the model's settings (Options)
    env_json     TEXT NOT NULL DEFAULT '{}',  -- browser facts, last log lines
    user_agent   TEXT NOT NULL DEFAULT '',
    app_version  TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS failures_created ON failures (created_at);
CREATE INDEX IF NOT EXISTS failures_kind ON failures (kind);
"""


class FailureStore:
    def __init__(self, path=DB_PATH):
        self.path = str(path)
        self._lock = threading.Lock()
        self._added = 0
        con = self._connect()
        try:
            con.executescript(_SCHEMA)
        finally:
            con.close()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        try:
            con.execute("PRAGMA journal_mode=WAL")
        except Exception:
            con.close()
            raise
        return con

    def add(self, *, kind: str, message: str, trace: str, stage: str, options: dict | None, env: dict,
            user_agent: str, app_version: str) -> int:
        row = (datetime.now(timezone.utc).isoformat(timespec="seconds"), kind, message, trace, stage,
               json.dumps(options) if options is not None else None, json.dumps(env), user_agent, app_version)
        with self._lock:
            con = self._connect()
            try:
                cur = con.execute("INSERT INTO failures (created_at, kind, message, trace, stage, options_json, "
                                  "env_json, user_agent, app_version) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", row)
                self._added += 1
                if self._added % 200 == 1:
                    con.execute("DELETE FROM failures WHERE id <= (SELECT MAX(id) FROM failures) - ?", (KEEP,))
                return int(cur.lastrowid)
            finally:
                con.close()

    def recent(self, n: int = 20) -> list[dict]:
        con = self._connect()
        con.row_factory = sqlite3.Row
        try:
            return [dict(r) for r in con.execute("SELECT * FROM failures ORDER BY id DESC LIMIT ?", (n,))]
        finally:
            con.close()


_STORE: FailureStore | None = None
_STORE_LOCK = threading.Lock()
_LIMIT = limits.RateLimit(PER_HOUR)     # reports per client key (limits.client_key), per hour
_GLOBAL = limits.RateLimit(GLOBAL_PER_HOUR, group_factor=0)


def store() -> FailureStore:
    global _STORE
    with _STORE_LOCK:
        if _STORE is None:
            _STORE = FailureStore()
        return _STORE


def rate_limited(addr: str) -> bool:
    """Over this client's hourly allowance, or over everyone's (``GLOBAL_PER_HOUR``)."""
    return _LIMIT.hit(addr) > 0 or (addr != limits.LOCAL and _GLOBAL.hit("all") > 0)


_CTRL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def _safe(s, keep_newlines: bool = False) -> str:
    """Printable form of client-supplied text: control bytes (ESC etc.) become U+FFFD."""
    s = "" if s is None else str(s)
    s = s.replace("\r", "") if keep_newlines else s.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    return _CTRL.sub("\ufffd", s)


def _dict(text) -> dict:
    """A stored JSON object (anything else - a report from an older version - as {})."""
    try:
        v = json.loads(text or "{}")
    except ValueError:
        return {}
    return v if isinstance(v, dict) else {}


def _main(argv: list[str]) -> None:
    n = int(argv[0]) if argv else 20
    rows = store().recent(n)
    if not rows:
        print("No device failures recorded.")
        return
    for r in reversed(rows):
        o = _dict(r["options_json"])
        env = _dict(r["env_json"])
        try:
            where = (f"{o.get('lat', 0):.4f},{o.get('lon', 0):.4f} {o.get('width_m', 0) / 1000:g}x"
                     f"{o.get('height_m', 0) / 1000:g} km {o.get('grid_cells')} cells {o.get('color_mode')} "
                     f"{o.get('source')}") if o else "-"
        except (TypeError, ValueError):
            where = "(unreadable settings)"
        py = env.get("python") if isinstance(env.get("python"), dict) else {}
        e = lambda k, d=env: _safe(str(d.get(k)))[:40]  # noqa: E731 - client-supplied values
        print(f"#{r['id']} {r['created_at']} [{_safe(r['kind'])}] {_safe(r['message'][:200])}")
        print(f"    model: {_safe(where)}")
        print(f"    device: {'phone' if env.get('phone') is True else 'desktop'}, {e('memory_mb')} MB allowed, "
              f"{e('device_memory_gb')} GB RAM, {e('cores')} cores, secure={e('secure')}, "
              f"python ready={e('ready', py)} jspi={e('jspi', py)} native={e('native', py)}")
        print(f"    stage: {_safe(r['stage'][:160])}")
        print(f"    browser: {_safe(r['user_agent'][:160])}")
        if r["trace"]:
            print("    trace: " + _safe(r["trace"][-600:], True).replace("\n", "\n           "))
        print()


if __name__ == "__main__":
    _main(sys.argv[1:])
