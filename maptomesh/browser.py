"""Run a generation job inside the browser (Pyodide: CPython on WebAssembly, in a Web Worker).

See docs/plans/browser-wasm.md. Point clouds (laspy + the lazrs wheel; the NRCan tile index is
queried on the server) and elevation models from every provider that works without GDAL
(``geotiff.py`` reads GeoTIFFs; hosts without CORS go through the server's ``/api/proxy``), plus
OpenStreetMap layers. The worker script (``web/browser-worker.js``) loads Pyodide, unpacks this package and
calls ``run_job``; it also provides the network functions used here:

    mtmFetch(url, headers, method, body)     -> Promise<{status, headers, body: Uint8Array}>
    mtmFetchMany([[url, headers], ...])      -> Promise<[...]>   (concurrent, bounded)
    mtmFetchSync(url, headers, method, body) -> {status, headers, body}  (synchronous XHR)
    mtmHint([[url, headers], ...])           start downloads now; a later mtmFetch picks them up

``install`` adapts the server code to the browser before any of it is imported:

* no threads: thread pools run their tasks inline;
* HTTP: ``requests`` and the range cache go through the worker's ``fetch`` (with the proxy as a
  fallback for hosts without CORS). Python waits on JavaScript promises with JavaScript Promise
  Integration (``run_sync``, Chrome / Edge); without it, synchronous XHR, one request at a time;
* no GDAL: ``dem_readers`` and the Mapterhorn provider use their pyproj / scipy / Pillow paths.
"""

from __future__ import annotations

import concurrent.futures as cf
import functools
import json
import os
import re
import time
from collections import OrderedDict

# providers that can't work here yet: LERC-compressed GeoTIFFs (NZ: Mapterhorn carries LINZ's
# 1 m anyway)
UNSUPPORTED_PROVIDERS = {"linz_1m", "linz_8m"}

NET = {"requests": 0, "bytes": 0, "seconds": 0.0}


# ---------------------------------------------------------------- threads

_GEOTIFF_URL = re.compile(r"^https?://\S+\.tiff?(\?\S*)?$", re.IGNORECASE)


def _urls_in(obj, depth: int = 0):
    """GeoTIFF URLs in a task's arguments (strings, tuples, lists, dicts, partials, closures)."""
    if isinstance(obj, str):
        if _GEOTIFF_URL.match(obj):
            yield obj
        return
    if depth > 3:
        return
    if isinstance(obj, (tuple, list)):
        for o in obj[:10_000]:
            yield from _urls_in(o, depth + 1)
    elif isinstance(obj, dict):
        for o in obj.values():
            yield from _urls_in(o, depth + 1)
    elif isinstance(obj, functools.partial):
        yield from _urls_in((obj.func, obj.args, obj.keywords), depth + 1)
    elif callable(obj) and getattr(obj, "__closure__", None):
        for c in obj.__closure__:
            try:
                yield from _urls_in(c.cell_contents, depth + 1)
            except ValueError:            # empty cell
                pass


class DeferredFuture(cf.Future):
    """A task of ``InlineExecutor``: runs when its result is first needed."""

    def __init__(self, ex: "InlineExecutor", fn, args, kwargs):
        super().__init__()
        self._ex, self._task = ex, (fn, args, kwargs)

    def run_now(self) -> None:
        if self._task is None:
            return
        fn, args, kwargs = self._task
        self._task = None
        if not self.set_running_or_notify_cancel():    # cancelled before it ran
            return
        try:
            self.set_result(fn(*args, **kwargs))
        except BaseException as e:  # noqa: BLE001 - delivered through the future, like a thread pool
            self.set_exception(e)

    def result(self, timeout=None):
        self._ex.hint()
        self.run_now()
        return super().result(0)

    def exception(self, timeout=None):
        self._ex.hint()
        self.run_now()
        return super().exception(0)


class InlineExecutor(cf.Executor):
    """``ThreadPoolExecutor`` stand-in: Pyodide has no threads. Tasks run one at a time, in order,
    when their results are asked for; before the first runs, the GeoTIFFs every queued task will
    open start downloading (their headers), which a thread pool would have fetched concurrently."""

    def __init__(self, *args, **kwargs):
        self.pending: list[DeferredFuture] = []
        self._hinted = 0

    def submit(self, fn, /, *args, **kwargs):
        f = DeferredFuture(self, fn, args, kwargs)
        self.pending.append(f)
        return f

    def hint(self) -> None:
        new = self.pending[self._hinted:]
        self._hinted = len(self.pending)
        urls = {u for f in new if f._task is not None for u in _urls_in(f._task)}
        if len(urls) > 1:
            from .httpcache import RANGE_CACHE
            for u in sorted(urls):
                RANGE_CACHE.hint(u, [(0, 1)])

    def map(self, fn, *iterables, timeout=None, chunksize=1):
        fs = [self.submit(fn, *args) for args in zip(*iterables)]
        return (f.result() for f in fs)

    def shutdown(self, wait=True, *, cancel_futures=False):
        if cancel_futures:
            for f in self.pending:
                f.cancel()
        if wait:
            for f in self.pending:
                f.run_now()


def _as_completed(fs, timeout=None):
    """``concurrent.futures.as_completed`` for deferred tasks: in submission order."""
    fs = list(fs)
    for f in fs:
        if isinstance(f, DeferredFuture):
            f._ex.hint()
    for f in fs:
        if isinstance(f, DeferredFuture):
            f.run_now()
        yield f


# ---------------------------------------------------------------- HTTP

def _js_headers(headers: dict | None):
    import js
    from pyodide.ffi import to_js
    h = {k: str(v) for k, v in (headers or {}).items() if k.lower() not in ("user-agent", "accept-encoding", "content-length", "connection", "host")}
    return to_js(h, dict_converter=js.Object.fromEntries)


def _response(url: str, r):
    import requests
    from requests.structures import CaseInsensitiveDict
    resp = requests.models.Response()
    resp.status_code = int(r.status)
    resp._content = r.body.to_bytes()
    resp.headers = CaseInsensitiveDict(r.headers.to_py())
    resp.url = url
    resp.reason = ""
    NET["requests"] += 1
    NET["bytes"] += len(resp._content)
    return resp


def http(url: str, headers: dict | None = None, method: str = "GET", body: bytes | None = None):
    """One request -> ``requests.Response``."""
    import js
    from pyodide.ffi import can_run_sync, run_sync, to_js
    t = time.perf_counter()
    b = to_js(memoryview(body)) if body else None
    if can_run_sync():
        r = run_sync(js.mtmFetch(url, _js_headers(headers), method, b))
    else:
        r = js.mtmFetchSync(url, _js_headers(headers), method, b)
    NET["seconds"] += time.perf_counter() - t
    return _response(url, r)


def http_many(reqs: list[tuple[str, dict | None]]) -> list:
    """Several GETs at once (concurrently with Promise Integration) -> ``requests.Response``s."""
    import js
    from pyodide.ffi import can_run_sync, run_sync, to_js
    if not reqs:
        return []
    t = time.perf_counter()
    if can_run_sync():
        arr = to_js([[u, _js_headers(h)] for u, h in reqs])
        rs = list(run_sync(js.mtmFetchMany(arr)))
    else:
        rs = [js.mtmFetchSync(u, _js_headers(h), "GET") for u, h in reqs]
    NET["seconds"] += time.perf_counter() - t
    return [_response(u, r) for (u, _), r in zip(reqs, rs)]


def _session_request(self, method, url, params=None, data=None, headers=None, cookies=None, files=None,
                     auth=None, timeout=None, allow_redirects=True, proxies=None, hooks=None, stream=None,
                     verify=None, cert=None, json=None):
    """``requests.Session.request`` over the worker's fetch (timeouts and redirects are the browser's)."""
    import requests
    method = method.upper()
    if method not in ("GET", "HEAD", "POST"):
        raise NotImplementedError(f"{method} requests are not supported in the browser")
    prep = requests.Request(method, url, params=params, data=data, json=json, headers=headers).prepare()
    body = prep.body.encode() if isinstance(prep.body, str) else prep.body
    return http(prep.url, dict(prep.headers), method, body)


class _Disk:
    """The worker's persistent cache (``web/opfs-cache.js``, in the browser's private file
    system): ``mtmCacheGet / Put / Has / DelPrefix``. Off while another tab holds it."""

    on = False

    @classmethod
    def refresh(cls) -> None:
        import js
        cls.on = bool(getattr(js, "mtmCacheOn", None) and js.mtmCacheOn())

    @classmethod
    def get(cls, key: str) -> bytes | None:
        if not cls.on:
            return None
        import js
        r = js.mtmCacheGet(key)
        if r is None or type(r).__name__ == "JsNull":
            return None
        return r.to_bytes()

    @classmethod
    def put(cls, key: str, data: bytes, when: float | None = None) -> None:
        if cls.on:
            import js
            from pyodide.ffi import to_js
            js.mtmCachePut(key, to_js(data), (when or time.time()) * 1000)

    @classmethod
    def has(cls, key: str) -> bool:
        if not cls.on:
            return False
        import js
        return bool(js.mtmCacheHas(key))

    @classmethod
    def drop(cls, prefix: str) -> None:
        if cls.on:
            import js
            js.mtmCacheDelPrefix(prefix)


class BrowserRangeCache:
    """``httpcache.RangeCache`` for the browser: same interface; blocks kept in memory (LRU) and
    in the persistent cache (``_Disk``), which outlives the page. Like the server's cache, each
    file's validator (ETag, else Last-Modified) is kept: a file replaced upstream drops its blocks,
    and files older than ``REVALIDATE_S`` are checked again."""

    def __init__(self, max_bytes: int = 256 << 20):
        from .httpcache import BLOCK
        self.block = BLOCK
        self.max_bytes = max_bytes
        self.blocks: OrderedDict[tuple[str, int], bytes] = OrderedDict()
        self.sizes: dict[str, int] = {}
        self.valid: dict[str, str] = {}       # url -> ETag / Last-Modified of the cached blocks
        self.checked: set[str] = set()        # urls whose saved state was looked at this session
        self.changed = 0                      # files found replaced upstream (read() retries)
        self.held = 0
        self.hit_bytes = self.miss_bytes = self.requests = 0
        self.disk_bytes = 0

    @staticmethod
    def _real(url: str) -> str:
        return url.split("#", 1)[0]          # "url#version" cache keys: the fragment isn't sent

    # ---------------------------------------------------------------- persistent part
    @staticmethod
    def _bkey(url: str, b: int) -> str:
        return f"b|{url}|{b}"

    def _meta_save(self, url: str, when: float | None = None) -> None:
        meta = {"v": self.valid.get(url, ""), "n": self.sizes.get(url), "t": when or time.time()}
        _Disk.put(f"v|{url}", json.dumps(meta).encode())

    def _check(self, url: str) -> None:
        """First use of ``url`` this session: its saved size and validator; blocks older than
        ``REVALIDATE_S`` are kept only if the file is unchanged (HEAD)."""
        if url in self.checked:
            return
        self.checked.add(url)
        raw = _Disk.get(f"v|{url}")
        if raw is None:
            return
        from .httpcache import REVALIDATE_S
        try:
            meta = json.loads(raw)
        except ValueError:
            meta = {}
        if time.time() - meta.get("t", 0) > REVALIDATE_S:
            try:
                r = http(self._real(url), method="HEAD")
                tag = r.headers.get("ETag") or r.headers.get("Last-Modified") or ""
                fresh = r.ok and tag and tag == meta.get("v")
            except Exception:  # noqa: BLE001 - offline: keep what we have
                fresh = True
            if not fresh:
                _Disk.drop(f"b|{url}|")
                _Disk.drop(f"v|{url}")
                return
            self._meta_save(url)
        if meta.get("v"):
            self.valid.setdefault(url, meta["v"])
        if meta.get("n") is not None:
            self.sizes.setdefault(url, int(meta["n"]))

    def _have(self, url: str, b: int) -> bool:
        return (url, b) in self.blocks or _Disk.has(self._bkey(url, b))

    def _block(self, url: str, b: int) -> bytes | None:
        d = self.blocks.get((url, b))
        if d is not None:
            self.blocks.move_to_end((url, b))
            return d
        d = _Disk.get(self._bkey(url, b))
        if d is not None:
            self.disk_bytes += len(d)
            self._keep(url, b, d)
        return d

    def _keep(self, url: str, b: int, d: bytes) -> None:
        old = self.blocks.pop((url, b), None)
        if old is not None:
            self.held -= len(old)
        self.blocks[(url, b)] = d
        self.held += len(d)
        while self.held > self.max_bytes and self.blocks:
            _, o = self.blocks.popitem(last=False)
            self.held -= len(o)

    def _validate(self, url: str, r) -> None:
        """A response's ETag / Last-Modified against the cached blocks'; a replaced file drops them."""
        tag = r.headers.get("ETag") or r.headers.get("Last-Modified") or ""
        old = self.valid.get(url)
        if tag and old and tag != old:
            for k in [k for k in self.blocks if k[0] == url]:
                self.held -= len(self.blocks.pop(k))
            _Disk.drop(f"b|{url}|")
            self.sizes.pop(url, None)
            self.valid[url] = tag
            self._meta_save(url)
            self.changed += 1
            if "#" in url:                    # versioned key: the caller looks the version up again
                from .httpcache import ChangedUpstream
                raise ChangedUpstream(f"{self._real(url)} changed upstream ({old} -> {tag})")
        elif tag and not old:
            self.valid[url] = tag
            self._meta_save(url)

    def size(self, url: str, session=None) -> int:
        self._check(url)
        if url not in self.sizes:
            r = http(self._real(url), method="HEAD")
            r.raise_for_status()
            self.sizes[url] = int(r.headers["Content-Length"])
            self._meta_save(url)
        return self.sizes[url]

    WHOLE = 64 << 20     # a read this large of a file of unknown size means "the whole file"

    def _runs(self, url: str, blocks: list[int], split: bool = True) -> list[tuple[int, int]]:
        """Missing blocks -> (first, last) runs of contiguous ones (at most PREFETCH_PIECE each)."""
        from .httpcache import PREFETCH_PIECE
        per = max(1, PREFETCH_PIECE // self.block) if split else 1 << 40
        runs: list[list[int]] = []
        for b in sorted(set(blocks)):
            if self._have(url, b):
                continue
            if runs and b == runs[-1][1] + 1 and b - runs[-1][0] < per:
                runs[-1][1] = b
            else:
                runs.append([b, b])
        return [(a, b) for a, b in runs]

    def _store(self, url: str, run: tuple[int, int], r) -> None:
        r.raise_for_status()
        self._validate(url, r)
        size0 = self.sizes.get(url)
        lo = run[0] * self.block
        data = r.content
        if r.status_code == 200:              # range ignored: the whole file came back
            self.sizes[url] = len(data)
            data = data[lo:(run[1] + 1) * self.block]
        cr = r.headers.get("Content-Range", "")
        if "/" in cr and cr.rsplit("/", 1)[1].isdigit():
            self.sizes[url] = int(cr.rsplit("/", 1)[1])
        if self.sizes.get(url) != size0:
            self._meta_save(url)
        self.requests += 1
        self.miss_bytes += len(data)
        for k, b in enumerate(range(run[0], run[1] + 1)):
            chunk = data[k * self.block:(k + 1) * self.block]
            if not chunk:
                break
            self._keep(url, b, chunk)
            _Disk.put(self._bkey(url, b), chunk)

    def _range(self, run: tuple[int, int]) -> dict:
        return {"Range": f"bytes={run[0] * self.block}-{(run[1] + 1) * self.block - 1}"}

    def _fetch(self, url: str, runs: list[tuple[int, int]]) -> None:
        reqs = [(self._real(url), self._range(run)) for run in runs]
        for run, r in zip(runs, http_many(reqs)):
            self._store(url, run, r)

    def _fetch_whole(self, url: str) -> None:
        r = http(self._real(url))
        r.raise_for_status()
        n = len(r.content)
        self.sizes[url] = n
        if n:
            self._store(url, (0, (n - 1) // self.block), r)

    def hint(self, url: str, ranges: list[tuple[int, int]] | None = None) -> None:
        """About to read these (start, length) ranges of ``url`` (None: the whole file): the worker
        starts downloading them in the background; the reads later pick them up."""
        import js
        from pyodide.ffi import to_js
        if not hasattr(js, "mtmHint"):
            return
        self._check(url)
        if ranges is None:
            if url in self.sizes or self._have(url, 0):
                return
            reqs = [[self._real(url), _js_headers(None)]]
        else:
            blocks = [b for st, n in ranges if n > 0 for b in range(st // self.block, (st + n - 1) // self.block + 1)]
            reqs = [[self._real(url), _js_headers(self._range(run))] for run in self._runs(url, blocks, split=False)]
        if reqs:
            js.mtmHint(to_js(reqs))

    def read(self, url: str, start: int, n: int, session=None) -> bytes:
        if n <= 0:
            return b""
        self._check(url)
        changed = self.changed
        out = self._read(url, start, n)
        if self.changed != changed:           # replaced upstream mid-read: read it all again
            out = self._read(url, start, n)
        return out

    def _read(self, url: str, start: int, n: int) -> bytes:
        size = self.sizes.get(url)
        if size is not None:
            n = max(0, min(n, size - start))
            if n == 0:
                return b""
        if size is None and n >= self.WHOLE and start == 0:
            self._fetch_whole(url)            # e.g. an EPT node: one plain GET
            size = self.sizes[url]
            n = min(n, size)
            if n == 0:
                return b""
        b0, b1 = start // self.block, (start + n - 1) // self.block
        runs = self._runs(url, list(range(b0, b1 + 1)), split=False)
        if runs:
            self._fetch(url, runs)
        parts = []
        for b in range(b0, b1 + 1):
            d = self._block(url, b)
            if d is None:
                break
            parts.append(d)
            self.hit_bytes += len(d)
        buf = b"".join(parts)
        off = start - b0 * self.block
        return buf[off:off + n]

    def prefetch(self, url: str, ranges: list[tuple[int, int]], workers: int = 8) -> None:
        self._check(url)
        blocks = [b for s, n in ranges if n > 0 for b in range(s // self.block, (s + n - 1) // self.block + 1)]
        runs = self._runs(url, blocks)
        if runs:
            self._fetch(url, runs)

    def counters(self) -> tuple[int, int, int]:
        return self.hit_bytes, self.miss_bytes, self.requests

    def total_bytes(self) -> int:
        return self.held

    def evict(self) -> None:
        pass


def _hint_sources(copc_urls: list[str], ept_urls: list[str]) -> None:
    """``pointcloud._hint_sources``: COPC headers and whole EPT nodes start downloading now, so
    they arrive while earlier tiles are being decoded (the tiles are processed one by one)."""
    from .httpcache import RANGE_CACHE
    for u in copc_urls:
        RANGE_CACHE.hint(u, [(0, 1)])
    for u in ept_urls:
        RANGE_CACHE.hint(u, None)


def _copc_chunks(source, byte_queries, out, num_threads) -> None:
    """laspy's COPC chunk download (a thread pool upstream): one concurrent batch instead."""
    from .httpcache import RANGE_CACHE
    RANGE_CACHE.prefetch(source.url, byte_queries)
    pos = 0
    for off, size in byte_queries:
        out[pos:pos + size] = RANGE_CACHE.read(source.url, off, size)
        pos += size


class RemoteTileIndex:
    """NRCan CanElevation tile lookups answered by the server (its index is 22 MB)."""

    def query(self, minx: float, miny: float, maxx: float, maxy: float) -> list:
        from .lidar_index import Tile
        r = http(f"/api/lidar/tiles?w={minx}&s={miny}&e={maxx}&n={maxy}")
        r.raise_for_status()
        return [Tile(**{**t, "bbox": tuple(t["bbox"])}) for t in r.json()["tiles"]]


class RemoteIndex:
    ready = True

    def get(self) -> RemoteTileIndex:
        return RemoteTileIndex()

    def load_now(self) -> bool:
        return True


def _osm_prefetch(url_tmpl: str, z: int, coords: list[tuple[int, int]]) -> None:
    """All of a job's OpenStreetMap tiles in one concurrent batch (``osm._get_tile`` then reads them
    from the in-memory file system)."""
    from . import config, osm
    todo = []
    for x, y in coords:
        path = osm.TILE_DIR / str(z) / str(x) / f"{y}.pbf"
        if not path.exists():
            todo.append((path, url_tmpl.replace("{z}", str(z)).replace("{x}", str(x)).replace("{y}", str(y))))
    for (path, _), r in zip(todo, http_many([(u, None) for _, u in todo])):
        if r.status_code in (204, 404):
            data = b""
        elif r.ok:
            data = r.content
        else:
            continue                          # osm._get_tile retries / reports it
        path.parent.mkdir(parents=True, exist_ok=True)
        config.atomic_write(path, data)


# ---------------------------------------------------------------- setup + jobs

_INSTALLED = False


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    os.environ.setdefault("MAPTOMESH_CACHE", "/cache")
    os.environ.setdefault("MAPTOMESH_MEMORY_MB", "3277")   # no psutil; the tab's limit is ~4 GB
    os.environ.setdefault("MAPTOMESH_PART_WORKERS", "1")
    cf.ThreadPoolExecutor = InlineExecutor   # before maptomesh modules bind it at import
    cf.as_completed = _as_completed
    import requests
    requests.sessions.Session.request = _session_request
    from . import httpcache, osm
    httpcache.RANGE_CACHE = BrowserRangeCache()
    osm._prefetch_tiles = _osm_prefetch
    import laspy.copc
    laspy.copc.http_thread_executor_strategy = _copc_chunks
    laspy.copc.http_queue_strategy = _copc_chunks
    from . import lidar_index, pointcloud
    lidar_index.INDEX = RemoteIndex()
    pointcloud._hint_sources = _hint_sources
    from . import dem, providers  # noqa: F401 - registers the providers
    dem.PROVIDERS[:] = [p for p in dem.PROVIDERS if p.key not in UNSUPPORTED_PROVIDERS]
    _INSTALLED = True


def run_job(opts_json: str, progress_js=None, memory_mb: float = 0) -> str:
    """Run one job; returns its result (JSON) with ``outdir``, the in-memory folder of its files,
    or ``{"error": {...}}`` when the job is refused up front (too large for ``memory_mb``)."""
    t0 = time.perf_counter()
    install()
    from . import config, limits, worker
    from .httpcache import RANGE_CACHE

    NET.update(requests=0, bytes=0, seconds=0.0)       # this job's traffic
    _, miss0, req0 = RANGE_CACHE.counters()
    disk0 = getattr(RANGE_CACHE, "disk_bytes", 0)
    _Disk.refresh()
    if hasattr(RANGE_CACHE, "checked"):
        RANGE_CACHE.checked.clear()           # the persistent cache may have changed (another tab)
    # /cache is mounted afresh by the worker (web/opfs-cache.js): the folders config made at import
    for d in (config.CACHE_DIR, config.OUTPUT_DIR, config.ELEV_CACHE_DIR, config.OSM_CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)
    from .options import Options

    opts = json.loads(opts_json)
    if memory_mb:
        limits.JOB_MEMORY_MB = int(memory_mb)   # what this tab may use (the page's estimate)
        RANGE_CACHE.max_bytes = int(min(256.0, max(48.0, memory_mb * 0.08)) * (1 << 20))
    try:
        limits.check(Options(**opts))
    except limits.LimitError as e:
        return json.dumps({"error": {"error_kind": e.kind, "message": e.message.replace("on this server", "on this device"),
                                     "advice": e.advice, "suggest": e.suggest}})
    jid = f"b{int(time.time() * 1000):x}"
    marks: list[tuple[float, float, str]] = []

    def progress(f: float, msg: str) -> None:
        now = time.perf_counter() - t0
        if not marks or marks[-1][2] != msg:
            marks.append((round(now, 2), round(f, 3), msg))
        if progress_js is not None:
            progress_js(float(f), str(msg))

    class Conn:   # worker.py's pipe to the server: the build may start right away here
        def send(self, msg):
            pass

        def recv(self):
            return ("build",)

    res = worker._pipeline(jid, opts, Conn(), progress)
    _, miss, nreq = RANGE_CACHE.counters()
    miss, nreq = miss - miss0, nreq - req0
    res["outdir"] = str(config.OUTPUT_DIR / jid)
    res["browser"] = {"seconds": round(time.perf_counter() - t0, 2), "marks": marks,
                      "net": {**NET, "seconds": round(NET["seconds"], 2), "range_bytes": miss,
                              "range_requests": nreq,
                              "cache_bytes": getattr(RANGE_CACHE, "disk_bytes", 0) - disk0,
                              "persistent_cache": _Disk.on}}
    return json.dumps(res)
