"""Resource limits for generation jobs.

Two lines of defence:

1. **Before a job starts** (``check``): the grid a job asks for must fit the memory budget, using
   a cost model measured on real jobs (peak memory grows with the number of grid vertices, and
   more for land-cover models - several coloured parts, roads, buildings - than for elevation
   bands or a single colour). Too large: the job is refused with the largest detail that fits.
2. **While it runs** (``server._run_job``): every job runs in its own process; one that runs past
   ``TIMEOUT_S`` or grows past ``JOB_MEMORY_MB`` is killed and reported with advice for retrying.

Point clouds over large areas don't need a refusal: beyond ``config.MAX_TILES`` tiles a job uses
elevation models instead (fast for any area).

Configuration: ``MAPTOMESH_MEMORY_MB`` (memory the jobs may use in total; default 80% of the
machine's RAM), ``MAPTOMESH_JOB_TIMEOUT_S`` (default 300), ``MAPTOMESH_MAX_QUEUE`` (default 8),
``MAPTOMESH_JOBS_PER_CLIENT`` (default 2).

Per-client request limits (``client_key``, ``RateLimit``, ``Slots``) are at the end.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field


def _total_ram_mb() -> int:
    try:
        import psutil
        return int(psutil.virtual_memory().total / 1e6)
    except Exception:  # noqa: BLE001
        return 4096


TIMEOUT_S = float(os.environ.get("MAPTOMESH_JOB_TIMEOUT_S", "300"))
TARGET_S = 120.0
MEMORY_MB = int(os.environ.get("MAPTOMESH_MEMORY_MB", "0")) or int(_total_ram_mb() * 0.8)
SERVER_MB = 450            # the server process itself (FastAPI, LiDAR tile index, caches)
# Jobs in flight: one building while the next downloads its data. Downloading point clouds
# needs memory too (~1 GB per million grid vertices), so small servers run one job at a time.
PIPELINE = int(os.environ.get("MAPTOMESH_PIPELINE", "0")) or (2 if MEMORY_MB - SERVER_MB >= 6000 else 1)
# what one job may use: everything but the server's share, less room for a second job in flight
JOB_MEMORY_MB = int(max(800, (MEMORY_MB - SERVER_MB) / (1.6 if PIPELINE > 1 else 1.0)))
MAX_QUEUE = int(os.environ.get("MAPTOMESH_MAX_QUEUE", "8"))
# Threads meshing a model's parts: 2 is ~20% faster but its peak is ~25% higher (land cover at 3 M
# vertices: 4.1 GB against 3.2 GB); small servers use 1. MAPTOMESH_PART_WORKERS overrides.
PART_WORKERS = int(os.environ.get("MAPTOMESH_PART_WORKERS", "0")) or (2 if JOB_MEMORY_MB >= 6000 else 1)

# Peak memory (MB) ~ BASE_MB + per-vertex cost * million grid vertices, the larger of
# - building the model: by colour mode (several coloured parts cost more) and part workers;
# - downloading point clouds (LiDAR jobs only): with classes (buildings, trees, land cover) or
#   ground only (bare terrain in bands / one colour).
# Measured on St. John's jobs at 600-2400 cells, 2026-11-19 (README "Server limits").
BASE_MB = 300.0
BUILD_MB = {1: {"landcover": 1000.0, "elevation": 650.0, "single": 650.0},
            2: {"landcover": 1240.0, "elevation": 900.0, "single": 670.0}}
FETCH_MB = {"classes": 1060.0, "ground": 330.0}
# build + files, seconds per million vertices on the development machine (downloads extra)
SEC_PER_MVERT = {"landcover": 10.0, "elevation": 6.5, "single": 4.0}


def kind_of(opts) -> str:
    return opts.color_mode if opts.color_mode in SEC_PER_MVERT else "landcover"


def _per_mvert(opts) -> float:
    build = BUILD_MB[min(PART_WORKERS, 2)][kind_of(opts)]
    if opts.source != "lidar":
        return build
    return max(build, FETCH_MB["ground" if opts.ground_only else "classes"])


def grid_shape(opts) -> tuple[int, int]:
    """(nx, ny) cells of the job's grid (as ``geo.Grid.for_aoi``)."""
    long_side = max(opts.width_m, opts.height_m)
    step = long_side / opts.grid_cells
    return max(4, round(opts.width_m / step)), max(4, round(opts.height_m / step))


def estimate(opts) -> dict:
    nx, ny = grid_shape(opts)
    mv = (nx + 1) * (ny + 1) / 1e6
    k = kind_of(opts)
    return {"mvertices": round(mv, 2), "memory_mb": round(BASE_MB + _per_mvert(opts) * mv),
            "build_s": round(SEC_PER_MVERT[k] * mv), "kind": k}


def max_cells(opts) -> int:
    """Largest ``grid_cells`` (long side) whose estimate fits the job memory budget (with 10% to
    spare for the estimate's error), capped at the option's own maximum."""
    aspect = min(opts.width_m, opts.height_m) / max(opts.width_m, opts.height_m)
    mv = max(0.05, (JOB_MEMORY_MB * 0.9 - BASE_MB) / _per_mvert(opts))
    return min(2400, int(math.sqrt(mv * 1e6 / max(aspect, 0.05))))


@dataclass
class LimitError(Exception):
    kind: str
    message: str
    advice: list[str] = field(default_factory=list)
    suggest: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return self.message


def check(opts) -> dict:
    """The estimate for ``opts``; raises ``LimitError`` when it doesn't fit."""
    est = estimate(opts)
    cap = max_cells(opts)
    if opts.grid_cells > cap:
        cap = max(150, cap // 50 * 50)      # a value the Detail slider (steps of 50) can take
        tips = [f"Lower Detail to {cap} cells or less (the most these settings allow here)."]
        if opts.color_mode == "landcover":
            tips.append("Elevation bands or a single colour need less memory than land-cover colours.")
        if opts.source == "lidar" and not opts.ground_only:
            tips.append("Bare terrain (no buildings or trees) reads lighter point clouds.")
        tips.append("A smaller area at the same Detail needs the same memory: memory follows the Detail.")
        raise LimitError(
            "too_large",
            f"{opts.grid_cells} cells of detail would need about {est['memory_mb'] / 1000:.1f} GB of memory; "
            f"a model may use {JOB_MEMORY_MB / 1000:.1f} GB on this server.",
            advice=tips, suggest={"grid_cells": cap})
    return est


# Point clouds, cold (nothing cached): ~0.2 s per CanElevation tile at 1000 cells (Toronto 10 km,
# 375 tiles: 86 s; Ottawa 20 km, 441 tiles: 74 s) and ~1 s per km2 of USGS Entwine (Denver 10 km:
# 96 s); finer detail reads deeper octree levels (data ~ cells^2). Over these, a job uses elevation
# models, which take seconds for any area. Keeps the download within ~1.5 minutes.
LIDAR_TILES_AT_1000 = int(os.environ.get("MAPTOMESH_MAX_LIDAR_TILES", "400"))
EPT_KM2_AT_1000 = 100.0


def lidar_caps(opts) -> tuple[int, float]:
    """(max CanElevation tiles, max USGS Entwine km2) for a job: fewer at finer detail."""
    f = (1000.0 / max(opts.grid_cells, 1)) ** 2
    tiles = int(min(LIDAR_TILES_AT_1000, max(100, LIDAR_TILES_AT_1000 * f)))
    km2 = float(min(150.0, max(25.0, EPT_KM2_AT_1000 * f)))
    return tiles, round(km2)


def public() -> dict:
    """The memory model for the UI, which caps its Detail slider the same way (``max_cells``)."""
    return {"memory_mb": JOB_MEMORY_MB, "timeout_s": TIMEOUT_S, "target_s": TARGET_S, "max_queue": MAX_QUEUE,
            "base_mb": BASE_MB, "build_mb": BUILD_MB[min(PART_WORKERS, 2)], "fetch_mb": FETCH_MB}


# ---------------------------------------------------------------- per-client request limits
# The public service sits behind Cloudflare, which puts the visitor's address in CF-Connecting-IP
# on every request (overwriting any the visitor sent). That header is believed only when the TCP
# peer can be Cloudflare: its published ranges (a proxied origin) or this machine (cloudflared /
# a local reverse proxy). Anyone reaching the server directly could otherwise choose the address
# their requests count against. X-Forwarded-For is not used: visitors can prepend to it.
#
# A request from this machine WITHOUT CF-Connecting-IP never went through Cloudflare: it is local
# (tests, maintenance) and no request limits apply to it (``LOCAL``), so test suites can run in
# parallel. MAPTOMESH_LOCAL_UNLIMITED=0 turns that off.
#
# IPv6 clients are keyed by their /64: one subscriber usually has a whole /64 and could otherwise
# rotate through addresses.

JOBS_PER_CLIENT = int(os.environ.get("MAPTOMESH_JOBS_PER_CLIENT", "2"))   # server jobs queued / running
LOCAL = "local"                 # the client key of local requests: exempt from every request limit
LOCAL_UNLIMITED = os.environ.get("MAPTOMESH_LOCAL_UNLIMITED", "1") != "0"

# https://www.cloudflare.com/ips-v4 and /ips-v6 (they change rarely; MAPTOMESH_TRUSTED_PROXIES
# adds ranges, comma-separated)
CLOUDFLARE_RANGES = (
    "173.245.48.0/20", "103.21.244.0/22", "103.22.200.0/22", "103.31.4.0/22", "141.101.64.0/18",
    "108.162.192.0/18", "190.93.240.0/20", "188.114.96.0/20", "197.234.240.0/22", "198.41.128.0/17",
    "162.158.0.0/15", "104.16.0.0/13", "104.24.0.0/14", "172.64.0.0/13", "131.0.72.0/22",
    "2400:cb00::/32", "2606:4700::/32", "2803:f800::/32", "2405:b500::/32", "2405:8100::/32",
    "2a06:98c0::/29", "2c0f:f248::/32",
)


def _networks() -> tuple:
    import ipaddress
    extra = [s.strip() for s in os.environ.get("MAPTOMESH_TRUSTED_PROXIES", "").split(",") if s.strip()]
    return tuple(ipaddress.ip_network(s, strict=False) for s in (*CLOUDFLARE_RANGES, *extra))


TRUSTED_PROXIES = _networks()


def _ip(text: str | None):
    import ipaddress
    try:
        ip = ipaddress.ip_address((text or "").strip().split("%", 1)[0])
    except ValueError:
        return None
    return ip.ipv4_mapped if ip.version == 6 and ip.ipv4_mapped is not None else ip


def _peer(request):
    """The TCP peer's address (uvicorn runs without proxy-header rewriting: see run.py)."""
    client = getattr(request, "client", None)
    return _ip(client.host) if client else None


def _cf_header(request) -> str | None:
    headers = getattr(request, "headers", None)
    return headers.get("cf-connecting-ip") if headers is not None else None


def client_ip(request) -> str:
    """The visitor's address: CF-Connecting-IP when the request came through a peer that can be
    Cloudflare (its ranges or this machine), else the peer itself."""
    peer = _peer(request)
    cf = _cf_header(request)
    if cf is not None and peer is not None and (peer.is_loopback or any(peer in n for n in TRUSTED_PROXIES)):
        ip = _ip(cf)
        if ip is not None:
            return str(ip)
    return str(peer) if peer is not None else ""


def is_local(request) -> bool:
    """From this machine and not through Cloudflare (tests, maintenance)."""
    peer = _peer(request)
    return LOCAL_UNLIMITED and peer is not None and peer.is_loopback and _cf_header(request) is None


def client_key(request) -> str:
    """The key per-client limits count against: an IPv4 address, an IPv6 /64, or ``LOCAL``."""
    import ipaddress
    if is_local(request):
        return LOCAL
    ip = _ip(client_ip(request))
    if ip is None:
        return "unknown"
    if ip.version == 6:
        return str(ipaddress.ip_network(f"{ip}/64", strict=False))
    return str(ip)


class RateLimit:
    """At most ``per_hour`` events per client key in any hour (sliding window)."""

    def __init__(self, per_hour: int, window_s: float = 3600.0):
        import threading
        self.per_hour, self.window = per_hour, window_s
        self._times: dict = {}
        self._lock = threading.Lock()

    def hit(self, key: str) -> float:
        """Count an event; 0 if allowed, else the seconds until the client may try again."""
        import time
        from collections import deque
        if key == LOCAL:
            return 0.0
        now = time.time()
        with self._lock:
            q = self._times.setdefault(key, deque())
            while q and now - q[0] > self.window:
                q.popleft()
            if len(q) >= self.per_hour:
                return max(1.0, self.window - (now - q[0]))
            q.append(now)
            if len(self._times) > 10_000:     # forget idle clients
                for k in [k for k, v in self._times.items() if not v]:
                    self._times.pop(k, None)
            return 0.0

    def clear(self) -> None:
        with self._lock:
            self._times.clear()


class Slots:
    """Work running at once: at most ``total`` overall and ``per_client`` per client. Never waits:
    a request that finds no free slot is refused (the caller answers 429 / 503 with Retry-After)
    instead of holding a server thread."""

    def __init__(self, total: int, per_client: int = 1):
        import threading
        self.total, self.per_client = max(1, total), max(1, per_client)
        self._active: dict[str, int] = {}
        self._n = 0
        self._lock = threading.Lock()

    def enter(self, key: str) -> str | None:
        """Take a slot: None if taken, else why not ("client": this client's share is in use,
        "busy": all slots are). Local requests (``LOCAL``) always get one and don't count."""
        if key == LOCAL:
            return None
        with self._lock:
            if self._active.get(key, 0) >= self.per_client:
                return "client"
            if self._n >= self.total:
                return "busy"
            self._active[key] = self._active.get(key, 0) + 1
            self._n += 1
            return None

    def leave(self, key: str) -> None:
        if key == LOCAL:
            return
        with self._lock:
            n = self._active.get(key, 0) - 1
            if n > 0:
                self._active[key] = n
            else:
                self._active.pop(key, None)
            self._n = max(0, self._n - 1)


# /api/coverage and /api/coverage/usgs: the page asks on every change of the area (debounced),
# so a higher hourly allowance than estimates; each may search a catalogue (network, CPU)
COVERAGE_RATE = RateLimit(int(os.environ.get("MAPTOMESH_COVERAGE_PER_HOUR", "1200")))
COVERAGE_SLOTS = Slots(int(os.environ.get("MAPTOMESH_COVERAGE_CONCURRENT", "6")), per_client=2)


def advice_for(kind: str, opts) -> tuple[list[str], dict]:
    """Retry advice (and a suggested setting) after a job was stopped for ``kind``."""
    lower = max(100, int(opts.grid_cells * 0.7) // 50 * 50)
    if kind == "timeout":
        tips = [f"Lower Detail (e.g. {lower} cells): time grows with the square of the detail."]
        if opts.source == "lidar":
            tips.append("For large areas choose 'Elevation models only' as the source: point clouds are "
                        "the slowest data to download.")
        if opts.color_mode == "landcover" or opts.surface != "terrain":
            tips.append("Bare terrain with elevation bands builds much faster than land cover with buildings.")
        tips.append("If the server is busy, retrying a little later can help (downloads are cached).")
        return tips, {"grid_cells": lower}
    if kind == "memory":
        tips = [f"Lower Detail (e.g. {lower} cells): memory grows with the square of the detail.",
                "Elevation bands or a single colour need less memory than land-cover colours."]
        return tips, {"grid_cells": lower}
    if kind == "busy":
        return ["Too many models are queued right now. Try again in a minute or two."], {}
    return [], {}
