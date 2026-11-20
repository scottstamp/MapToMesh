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
from collections import deque
from datetime import datetime, timezone

from . import config

DB_PATH = config.DATA_DIR / "device_failures.sqlite"
PER_HOUR = 30                 # reports per client address
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
        con.execute("PRAGMA journal_mode=WAL")
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
_TIMES: dict[str, deque] = {}


def store() -> FailureStore:
    global _STORE
    with _STORE_LOCK:
        if _STORE is None:
            _STORE = FailureStore()
        return _STORE


def rate_limited(addr: str) -> bool:
    now = time.time()
    with _STORE_LOCK:
        q = _TIMES.setdefault(addr, deque())
        while q and now - q[0] > 3600:
            q.popleft()
        if len(q) >= PER_HOUR:
            return True
        q.append(now)
        if len(_TIMES) > 10_000:
            for k in [k for k, v in _TIMES.items() if not v]:
                _TIMES.pop(k, None)
        return False


_CTRL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def _safe(s, keep_newlines: bool = False) -> str:
    """Printable form of client-supplied text: control bytes (ESC etc.) become U+FFFD."""
    s = "" if s is None else str(s)
    s = s.replace("\r", "") if keep_newlines else s.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    return _CTRL.sub("\ufffd", s)


def _main(argv: list[str]) -> None:
    n = int(argv[0]) if argv else 20
    rows = store().recent(n)
    if not rows:
        print("No device failures recorded.")
        return
    for r in reversed(rows):
        o = json.loads(r["options_json"]) if r["options_json"] else {}
        env = json.loads(r["env_json"] or "{}")
        where = (f"{o.get('lat', 0):.4f},{o.get('lon', 0):.4f} {o.get('width_m', 0) / 1000:g}x{o.get('height_m', 0) / 1000:g} km "
                 f"{o.get('grid_cells')} cells {o.get('color_mode')} {o.get('source')}") if o else "-"
        py = env.get("python") or {}
        print(f"#{r['id']} {r['created_at']} [{r['kind']}] {_safe(r['message'][:200])}")
        print(f"    model: {where}")
        print(f"    device: {'phone' if env.get('phone') else 'desktop'}, {env.get('memory_mb')} MB allowed, "
              f"{env.get('device_memory_gb')} GB RAM, {env.get('cores')} cores, secure={env.get('secure')}, "
              f"python ready={py.get('ready')} jspi={py.get('jspi')} native={py.get('native')}")
        print(f"    stage: {_safe(r['stage'][:160])}")
        print(f"    browser: {_safe(r['user_agent'][:160])}")
        if r["trace"]:
            print("    trace: " + _safe(r["trace"][-600:], True).replace("\n", "\n           "))
        print()


if __name__ == "__main__":
    _main(sys.argv[1:])
