"""Persistent HTTP range cache for public LiDAR / DEM files (NRCan CanElevation, USGS 3DEP).

These are free public resources: every byte fetched from them is kept in
an on-disk SQLite block store (64 KiB blocks per URL) so that any later query -
another area overlapping the same tiles, a different level of detail, a
re-generation with other print settings - only downloads blocks it has never
seen. Blocks are evicted least-recently-used beyond a size limit, and each URL's
ETag is re-checked after 30 days (if the file changed, its blocks are dropped).

Used by
  * laspy's COPC reader (``HttpRangeStream.read`` is routed through ``RANGE_CACHE``)
  * rasterio / GDAL for HRDEM COGs via ``CachedHttpFile`` as a rasterio ``opener``

Hosts that need signed URLs (Planetary Computer's SAS tokens) register a signer: callers and
cache keys always use the plain URL, the token is only added to the outgoing request (and
renewed once when the host rejects an expired one), so cached blocks outlive the tokens.
"""

from __future__ import annotations

import concurrent.futures as cf
import io
import os
import sqlite3
import threading
import time

import requests

from . import config

BLOCK = 64 * 1024
REVALIDATE_S = 30 * 86400
TOUCH_S = 3600            # LRU timestamps are refreshed at most this often (no write per cache hit)
PREFETCH_PIECE = 4 << 20  # prefetch() splits ranges into requests of at most this size
PREFETCH_GAP = 4 * BLOCK  # ... and merges ranges closer than this

# (url prefix, signer(url, refresh) -> signed url)
_SIGNERS: list = []


def register_signer(prefix: str, signer) -> None:
    _SIGNERS.append((prefix, signer))


def _signer_for(url: str):
    for prefix, fn in _SIGNERS:
        if url.startswith(prefix):
            return fn
    return None


def http_request(session: requests.Session, method: str, url: str, **kw) -> requests.Response:
    """``session.request`` with the URL signed when its host needs it (retry once with a fresh
    token if the signature was rejected)."""
    fn = _signer_for(url)
    if fn is None:
        return session.request(method, url, **kw)
    r = session.request(method, fn(url, False), **kw)
    if r.status_code in (403, 409):  # expired / invalid token
        r = session.request(method, fn(url, True), **kw)
    return r


class RangeCache:
    def __init__(self, path, max_bytes: int):
        self.path = str(path)
        self.max_bytes = max_bytes
        self._local = threading.local()
        self._lock = threading.Lock()
        self.hit_bytes = 0
        self.miss_bytes = 0
        self.requests = 0
        self._writes_since_check = 0
        con = self._conn()
        con.executescript("""
            CREATE TABLE IF NOT EXISTS urls (id INTEGER PRIMARY KEY, url TEXT UNIQUE NOT NULL,
                                             size INTEGER, etag TEXT, checked REAL);
            CREATE TABLE IF NOT EXISTS blocks (url_id INTEGER NOT NULL, block INTEGER NOT NULL,
                                               data BLOB NOT NULL, used REAL NOT NULL,
                                               PRIMARY KEY (url_id, block));
            CREATE INDEX IF NOT EXISTS blocks_used ON blocks (used);
        """)
        con.commit()

    # ------------------------------------------------------------------ plumbing
    def _conn(self) -> sqlite3.Connection:
        con = getattr(self._local, "con", None)
        if con is None:
            con = sqlite3.connect(self.path, timeout=60, check_same_thread=False)
            con.execute("PRAGMA journal_mode=WAL")
            con.execute("PRAGMA synchronous=NORMAL")
            self._local.con = con
        return con

    def _session(self) -> requests.Session:
        s = getattr(self._local, "session", None)
        if s is None:
            s = requests.Session()
            s.headers["User-Agent"] = config.USER_AGENT
            adapter = requests.adapters.HTTPAdapter(max_retries=requests.adapters.Retry(
                total=4, backoff_factor=0.5, status_forcelist=(429, 500, 502, 503, 504)))
            s.mount("https://", adapter)
            s.mount("http://", adapter)
            self._local.session = s
        return s

    def _url_row(self, url: str, session: requests.Session | None) -> tuple[int, int | None]:
        con = self._conn()
        row = con.execute("SELECT id, size, etag, checked FROM urls WHERE url = ?", (url,)).fetchone()
        now = time.time()
        if row is None:
            con.execute("INSERT OR IGNORE INTO urls (url, checked) VALUES (?, ?)", (url, now))
            con.commit()
            row = con.execute("SELECT id, size, etag, checked FROM urls WHERE url = ?", (url,)).fetchone()
        uid, size, etag, checked = row
        if etag and checked and now - checked > REVALIDATE_S:
            try:
                r = http_request(session or self._session(), "HEAD", url, timeout=30)
                with self._lock:
                    self.requests += 1
                new = r.headers.get("ETag")
                if new and new != etag:  # file was replaced upstream: drop everything we had
                    con.execute("DELETE FROM blocks WHERE url_id = ?", (uid,))
                    size = int(r.headers.get("Content-Length", 0)) or None
                con.execute("UPDATE urls SET etag = ?, size = ?, checked = ? WHERE id = ?",
                            (new or etag, size, now, uid))
                con.commit()
            except requests.RequestException:
                pass  # offline: keep serving the cache
        return uid, size

    # ------------------------------------------------------------------ public
    def size(self, url: str, session: requests.Session | None = None) -> int:
        uid, size = self._url_row(url, session)
        if size is None:
            r = http_request(session or self._session(), "HEAD", url, timeout=30, allow_redirects=True)
            r.raise_for_status()
            with self._lock:
                self.requests += 1
            size = int(r.headers["Content-Length"])
            con = self._conn()
            con.execute("UPDATE urls SET size = ?, etag = COALESCE(etag, ?) WHERE id = ?",
                        (size, r.headers.get("ETag"), uid))
            con.commit()
        return size

    def read(self, url: str, start: int, n: int, session: requests.Session | None = None) -> bytes:
        if n <= 0:
            return b""
        uid, size = self._url_row(url, session)
        if size is not None:
            n = max(0, min(n, size - start))
            if n == 0:
                return b""
        b0, b1 = start // BLOCK, (start + n - 1) // BLOCK
        con = self._conn()
        rows = con.execute("SELECT block, data, used FROM blocks WHERE url_id = ? AND block BETWEEN ? AND ?",
                           (uid, b0, b1)).fetchall()
        have = {b: d for b, d, _ in rows}
        missing = [b for b in range(b0, b1 + 1) if b not in have]
        if missing:
            have.update(self._fetch(url, uid, missing, session))
        now = time.time()
        stale = [b for b, _, used in rows if used < now - TOUCH_S]   # LRU order only needs hours
        if stale:
            con.execute(f"UPDATE blocks SET used = ? WHERE url_id = ? AND block IN ({','.join('?' * len(stale))})",
                        (now, uid, *stale))
            con.commit()
        buf = b"".join(have[b] for b in range(b0, b1 + 1) if b in have)
        off = start - b0 * BLOCK
        out = buf[off:off + n]
        with self._lock:
            self.hit_bytes += sum(len(d) for _, d, _ in rows)
        return out

    def prefetch(self, url: str, ranges: list[tuple[int, int]], workers: int = 8) -> None:
        """Download the (start, length) byte ranges of ``url`` that aren't cached yet, in parallel.

        For readers that fetch one small range after another (GDAL reading a COG through
        ``CachedHttpFile``): asking for everything they're about to read first turns a long chain
        of round trips into a few concurrent requests; their own reads then hit the cache."""
        if not ranges:
            return
        merged: list[list[int]] = []
        for s, n in sorted((s, n) for s, n in ranges if n > 0):
            if merged and s <= merged[-1][1] + PREFETCH_GAP:
                merged[-1][1] = max(merged[-1][1], s + n)
            else:
                merged.append([s, s + n])
        pieces = [(p, min(PREFETCH_PIECE, e - p)) for s, e in merged for p in range(s, e, PREFETCH_PIECE)]
        if len(pieces) == 1:
            self.read(url, *pieces[0])
            return
        with cf.ThreadPoolExecutor(max_workers=min(workers, len(pieces))) as ex:
            for fut in [ex.submit(self.read, url, s, n) for s, n in pieces]:
                fut.result()

    def _fetch(self, url: str, uid: int, blocks: list[int], session) -> dict[int, bytes]:
        # contiguous runs -> one ranged GET each
        runs, cur = [], [blocks[0]]
        for b in blocks[1:]:
            if b == cur[-1] + 1:
                cur.append(b)
            else:
                runs.append(cur); cur = [b]
        runs.append(cur)
        out: dict[int, bytes] = {}
        con = self._conn()
        now = time.time()
        sess = session or self._session()
        for run in runs:
            lo, hi = run[0] * BLOCK, (run[-1] + 1) * BLOCK - 1
            r = http_request(sess, "GET", url, headers={"Range": f"bytes={lo}-{hi}"}, timeout=120)
            r.raise_for_status()
            data = r.content
            total = None
            cr = r.headers.get("Content-Range")  # "bytes lo-hi/total"
            if cr and "/" in cr and cr.rsplit("/", 1)[1].isdigit():
                total = int(cr.rsplit("/", 1)[1])
            if r.status_code == 200:  # server ignored the range: whole file
                data = data[lo:hi + 1]
            with self._lock:
                self.requests += 1
                self.miss_bytes += len(data)
            for k, b in enumerate(run):
                chunk = data[k * BLOCK:(k + 1) * BLOCK]
                if not chunk:
                    break
                out[b] = chunk
            con.executemany("INSERT OR REPLACE INTO blocks (url_id, block, data, used) VALUES (?, ?, ?, ?)",
                            [(uid, b, d, now) for b, d in out.items() if b in run])
            con.execute("UPDATE urls SET size = COALESCE(?, size), etag = COALESCE(etag, ?) WHERE id = ?",
                        (total, r.headers.get("ETag"), uid))
            con.commit()
        with self._lock:
            self._writes_since_check += len(blocks)
            check = self._writes_since_check > 2000
            if check:
                self._writes_since_check = 0
        if check:
            self.evict()
        return out

    def total_bytes(self) -> int:
        return int(self._conn().execute("SELECT COALESCE(SUM(LENGTH(data)), 0) FROM blocks").fetchone()[0])

    def evict(self) -> None:
        """Drop least-recently-used blocks beyond the size limit (down to 90% of it)."""
        total = self.total_bytes()
        if total <= self.max_bytes:
            return
        con = self._conn()
        target = total - int(self.max_bytes * 0.9)
        freed = 0
        for uid, b, ln in con.execute("SELECT url_id, block, LENGTH(data) FROM blocks ORDER BY used"):
            con.execute("DELETE FROM blocks WHERE url_id = ? AND block = ?", (uid, b))
            freed += ln
            if freed >= target:
                break
        con.commit()

    def counters(self) -> tuple[int, int, int]:
        with self._lock:
            return self.hit_bytes, self.miss_bytes, self.requests


RANGE_CACHE = RangeCache(config.CACHE_DIR / "nrcan_range_cache.sqlite",
                         int(float(os.environ.get("MAPTOMESH_DATA_CACHE_GB", "10")) * 1e9))


class CachedHttpFile(io.RawIOBase):
    """Seekable read-only file over HTTP, backed by ``RANGE_CACHE`` (rasterio opener)."""

    def __init__(self, url: str):
        self.url = url
        self.pos = 0
        self._size = RANGE_CACHE.size(url)

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def seek(self, pos: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self.pos = pos
        elif whence == io.SEEK_CUR:
            self.pos += pos
        else:
            self.pos = self._size + pos
        return self.pos

    def tell(self) -> int:
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self._size - self.pos
        data = RANGE_CACHE.read(self.url, self.pos, n)
        self.pos += len(data)
        return data

    def readinto(self, b) -> int:
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def install_laspy_cache() -> None:
    """Route laspy's COPC HTTP range reads through the persistent cache (and URL signers)."""
    from laspy import copc

    if getattr(copc.HttpRangeStream, "_maptomesh_cached", False):
        return

    def read(self, n):
        if n == 0:
            return b""
        data = RANGE_CACHE.read(self.url, self.range_start, n, self.session)
        self.range_start += len(data)
        return data

    copc.HttpRangeStream.read = read
    copc.HttpRangeStream._maptomesh_cached = True
