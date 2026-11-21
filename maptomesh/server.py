"""FastAPI app: map UI + generation jobs."""

from __future__ import annotations

import contextlib
import copy
import hmac
import json
import logging
import math
import os
import re
import secrets
import threading
import time
import traceback
import uuid
from collections import Counter, OrderedDict
from contextlib import contextmanager
from pathlib import Path
from dataclasses import dataclass, field

from typing import Annotated

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, create_model, field_validator

from . import __version__, config
from .geo import AOI, Grid
from .lidar_index import INDEX, PROJECTS_GEOJSON
from .limits import RateLimit, client_key
from .options import Options
from .worker import scrub

log = logging.getLogger("maptomesh")

# ---------------------------------------------------------------- security headers (plain ASGI: streams pass through)

_SECURITY_HEADERS = ((b"x-content-type-options", b"nosniff"),
                     (b"referrer-policy", b"strict-origin-when-cross-origin"),
                     (b"x-frame-options", b"DENY"))

# Content-Security-Policy. The page (/ and /static/*.html): scripts only from this site, the pinned
# CDN files (Subresource Integrity in web/index.html) and its inline scripts by hash; styles from
# this site and Leaflet's (Leaflet and three.js style elements through the CSSOM, which CSP allows;
# web/app.js does the same); images from this site, the map tiles, blob: (models built on the
# device) and data:; data from any https host (builds on the device fetch the agencies' services
# directly); blob: workers only for app.js's module-worker feature test.
CDN_SCRIPTS = ("https://unpkg.com/leaflet@1.9.4/dist/leaflet.js", "https://unpkg.com/three@0.170.0/")
CDN_STYLES = ("https://unpkg.com/leaflet@1.9.4/dist/leaflet.css",)
MAP_IMAGES = ("https://tile.openstreetmap.org", "https://unpkg.com/leaflet@1.9.4/dist/images/")
# The build worker (web/browser-worker.js, a module worker: its policy comes with its own response,
# not the page's): Pyodide's JavaScript is imported from blob: URLs once its hashes are checked,
# and it compiles WebAssembly (Python and its packages).
WORKER_CSP = ("default-src 'none'; script-src 'self' blob: 'wasm-unsafe-eval'; connect-src 'self' https:; "
              "frame-ancestors 'none'")
# Everything else (JSON, models, images; the data proxy sets its own): nothing runs from it.
API_CSP = "default-src 'none'; frame-ancestors 'none'; sandbox"
_INLINE_SCRIPT = re.compile(r"<script\b(?![^>]*\bsrc\s*=)[^>]*>(.*?)</script\s*>", re.S | re.I)


def inline_script_hashes(page: str) -> list[str]:
    """CSP hash sources of a page's inline scripts (index.html: the colour theme, the import map)."""
    import base64
    import hashlib
    return [f"'sha256-{base64.b64encode(hashlib.sha256(s.encode('utf-8')).digest()).decode()}'"
            for s in _INLINE_SCRIPT.findall(page)]


def page_csp(page: str) -> str:
    return "; ".join((
        "default-src 'self'",
        "script-src " + " ".join(("'self'", *CDN_SCRIPTS, *inline_script_hashes(page))),
        "style-src " + " ".join(("'self'", *CDN_STYLES)),
        "img-src " + " ".join(("'self'", "data:", "blob:", *MAP_IMAGES)),
        "connect-src 'self' https: blob:",
        "worker-src 'self' blob:",
        "object-src 'none'",
        "base-uri 'self'",
        "form-action 'self'",
        "frame-ancestors 'none'",
    ))


_PAGE_CSP: dict[str, tuple[float, bytes]] = {}


def _html_csp(name: str) -> bytes:
    """The page CSP of web/<name> (hashes of its inline scripts; worked out again when it changes)."""
    try:
        path = (config.WEB_DIR / name).resolve()
        if path.parent != config.WEB_DIR.resolve():
            raise FileNotFoundError(name)
        mtime = path.stat().st_mtime
        hit = _PAGE_CSP.get(name)
        if hit is None or hit[0] != mtime:
            hit = _PAGE_CSP[name] = (mtime, page_csp(path.read_text(encoding="utf-8")).encode())
        return hit[1]
    except (OSError, ValueError):
        return page_csp("").encode()


def csp_for(path: str) -> bytes:
    if path == "/":
        return _html_csp("index.html")
    if path.startswith("/static/"):
        return _html_csp(path[len("/static/"):]) if path.endswith(".html") else WORKER_CSP.encode()
    return API_CSP.encode()


class _Headers:
    """Security headers and a Content-Security-Policy on every response (kept where a route set its
    own, e.g. the proxy's sandbox CSP), and ``Cache-Control: no-cache`` on the UI so browsers
    revalidate it after updates."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        path = scope.get("path", "")
        ui = path == "/" or path.startswith("/static/")

        async def send_(msg):
            if msg["type"] == "http.response.start":
                headers = [(k, v) for k, v in msg.get("headers", []) if not (ui and k.lower() == b"cache-control")]
                have = {k.lower() for k, _ in headers}
                headers += [(k, v) for k, v in _SECURITY_HEADERS if k not in have]
                if b"content-security-policy" not in have:
                    headers.append((b"content-security-policy", csp_for(path)))
                if ui:
                    headers.append((b"cache-control", b"no-cache"))
                msg = {**msg, "headers": headers}
            await send(msg)

        await self.app(scope, receive, send_)


class _App(FastAPI):
    def build_middleware_stack(self):
        # outermost, around Starlette's ServerErrorMiddleware too: its 500s get the headers as well
        return _Headers(super().build_middleware_stack())


# no interactive API docs or schema in public: the page is the only client
app = _App(title="MapToMesh", version=__version__, docs_url=None, redoc_url=None, openapi_url=None)


# Request bodies: the API takes small JSON documents (settings, reports) and the proxy forwards
# small POST queries; anything larger is refused with 413 before it is read into memory.
BODY_LIMIT = 64_000
BODY_LIMITS = {"/api/proxy": 64_000,                # POST queries of catalogue APIs (STAC searches)
               "/api/device-failures": 128_000}  # message + trace + browser facts
REVIEW_BODY_LIMIT = 300_000                         # a review carries the browser's copy of its job


def body_limit(path: str) -> int:
    if path.startswith("/api/jobs/") and path.endswith("/review"):
        return REVIEW_BODY_LIMIT
    return BODY_LIMITS.get(path, BODY_LIMIT)


class _BodyTooLarge(Exception):
    pass


class _BodyLimit:
    """413 for a request body over ``body_limit``: by its Content-Length before reading anything,
    and by counting the bytes of a body without one (chunked) as the route reads it."""

    def __init__(self, app):
        self.app = app

    @staticmethod
    async def _refuse(send) -> None:
        body = b'{"detail":"Request body too large"}'
        await send({"type": "http.response.start", "status": 413,
                    "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode()),
                                (b"connection", b"close")]})
        await send({"type": "http.response.body", "body": body})

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        limit = body_limit(scope.get("path", ""))
        for k, v in scope.get("headers", []):
            if k.lower() == b"content-length":
                if not v.strip().isdigit() or int(v) > limit:
                    return await self._refuse(send)
        seen = 0
        over = False
        started = False

        async def receive_():
            nonlocal seen, over
            if over:
                raise _BodyTooLarge()
            msg = await receive()
            if msg["type"] == "http.request":
                seen += len(msg.get("body", b""))
                if seen > limit:
                    over = True
                    raise _BodyTooLarge()
            return msg

        async def send_(msg):
            nonlocal started
            if over:
                return        # whatever the route made of the aborted read; 413 follows
            if msg["type"] == "http.response.start":
                started = True
            await send(msg)

        try:
            await self.app(scope, receive_, send_)
        except Exception:
            if not over:
                raise
        if over and not started:
            await self._refuse(send)


app.add_middleware(_BodyLimit)     # (_Headers wraps the whole stack: _App)

# MAPTOMESH_ALLOWED_HOSTS (comma-separated, e.g. "maptomesh.com,www.maptomesh.com"): requests for any
# other Host get 400, so an origin reachable from Cloudflare's shared ranges only answers for its
# own names (README "Deployment behind Cloudflare"). Empty: any Host (local use).
ALLOWED_HOSTS = [h.strip() for h in os.environ.get("MAPTOMESH_ALLOWED_HOSTS", "").split(",") if h.strip()]
if ALLOWED_HOSTS:
    from starlette.middleware.trustedhost import TrustedHostMiddleware
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)


# ---------------------------------------------------------------- jobs

@dataclass
class Job:
    id: str
    opts: Options
    state: str = "queued"  # queued | running | done | error | cancelled
    progress: float = 0.0
    message: str = "Queued"
    log: list[str] = field(default_factory=list)
    error: str | None = None
    # why it failed: "timeout" | "memory" | "too_large" | "busy" | "error"; with retry advice and
    # a suggested setting (e.g. {"grid_cells": 1000}) for the UI's failure dialog
    error_kind: str | None = None
    advice: list[str] = field(default_factory=list)
    suggest: dict = field(default_factory=dict)
    result: dict | None = None
    created: float = field(default_factory=time.time)
    seconds: float = 0.0          # active time (waiting for another model's build excluded)
    peak_mb: int = 0              # the job process's peak memory
    cancel: threading.Event = field(default_factory=threading.Event)
    # not public: who submitted it (per-client job cap) and the secret that lets them cancel it
    client: str = ""
    cancel_token: str = ""

    def set(self, f: float, msg: str) -> None:
        self.progress = max(0.0, min(1.0, f))
        msg = scrub(msg) if msg else msg       # progress text may quote errors: no local paths
        if msg and msg != self.message:
            self.message = msg
            # keep the log readable: drop repetitive per-tile updates
            if not self.log or self.log[-1].split(" ")[0:2] != msg.split(" ")[0:2]:
                self.log.append(msg)
            else:
                self.log[-1] = msg

    def public(self) -> dict:
        return {"id": self.id, "state": self.state, "progress": round(self.progress, 4),
                "message": self.message, "log": self.log[-30:], "error": self.error,
                "error_kind": self.error_kind, "advice": self.advice, "suggest": self.suggest,
                "seconds": round(self.seconds, 1), "peak_mb": self.peak_mb, "result": self.result,
                "options": self.opts.model_dump()}


JOBS: "OrderedDict[str, Job]" = OrderedDict()     # recent jobs (metadata only; results are files)
# Finished jobs' files: the newest KEEP_JOBS, and any finished within KEEP_JOB_HOURS (so links
# shared a moment ago keep working however many jobs others run), within OUTPUT_MAX_GB in total.
KEEP_JOBS = int(os.environ.get("MAPTOMESH_KEEP_JOBS", "10"))
KEEP_JOB_HOURS = float(os.environ.get("MAPTOMESH_KEEP_JOB_HOURS", "1"))
OUTPUT_MAX_GB = float(os.environ.get("MAPTOMESH_OUTPUT_MAX_GB", "5"))
# At most two jobs in flight: one builds (CPU and memory heavy, one at a time) while the next
# downloads its data (network-bound), so a queue moves up to twice as fast without holding more
# than two models' data in memory. Servers with less than ~7 GB run one job at a time
# (``limits.PIPELINE``; MAPTOMESH_PIPELINE overrides).
def _pipeline_size() -> int:
    from . import limits
    return limits.PIPELINE


_PIPELINE = threading.BoundedSemaphore(_pipeline_size())
_BUILD_LOCK = threading.Lock()
_CHILDREN: dict[str, object] = {}                 # job id -> psutil.Process of its worker
_CHILDREN_LOCK = threading.Lock()


class _Stop(Exception):
    """A job stopped by the server: ``kind`` is the reason (timeout / memory / cancelled / error).
    ``public``: ``message`` was written for users; else it is replaced by a generic message with an
    error id (``_internal_error``). ``trace`` only goes to the server log."""

    def __init__(self, kind: str, message: str, trace: str = "", public: bool = True):
        super().__init__(message)
        self.kind, self.message, self.trace, self.public = kind, message, trace, public


def _error_id() -> str:
    return uuid.uuid4().hex[:8]


def _internal_error(what: str, detail: str) -> str:
    """Log an unexpected error with a short id; the id is what users see (and can report)."""
    eid = _error_id()
    log.error("%s [error %s]\n%s", what, eid, detail)
    return eid


def _job_failed_message(eid: str) -> str:
    return (f"Something went wrong on the server while building this model (error id {eid}). "
            "Trying again may help; if it keeps happening, please report it with this id.")


def _wait_for(lock, job: "Job", msg: str) -> None:
    """Acquire ``lock`` (Lock / Semaphore), showing ``msg`` meanwhile; raise if the job is cancelled."""
    if lock.acquire(blocking=False):
        return
    job.set(job.progress, msg)
    while not lock.acquire(timeout=0.5):
        if job.cancel.is_set():
            raise RuntimeError("Cancelled")


def _slug(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", s).strip("_")
    return s[:60] or "model"


_JOB_ID = re.compile(r"[0-9a-f]{12}")      # uuid4().hex[:12]; always with fullmatch


def valid_job_id(jid) -> bool:
    return isinstance(jid, str) and _JOB_ID.fullmatch(jid) is not None


def job_path(jid: str, name: str = "") -> Path:
    """``OUTPUT_DIR/<jid>[/<name>]`` for a well-formed job id and a plain file name; anything else
    (separators, drive or UNC paths, ``..``) is refused, so a request can't reach other files."""
    if not valid_job_id(jid) or (name and (name != Path(name).name or name in (".", "..")
                                           or "\\" in name or ":" in name)):
        raise HTTPException(404, "Unknown job")
    root = config.OUTPUT_DIR.resolve()
    path = (root / jid / name) if name else (root / jid)
    if not path.resolve().is_relative_to(root):
        raise HTTPException(404, "Unknown job")
    return path


def _job_file(jid: str):
    return job_path(jid, "job.json")


def _persist(job: Job) -> None:
    """The finished job's metadata next to its files: links keep working after a restart."""
    d = job.public()
    d["created"] = job.created
    path = _job_file(job.id)
    path.parent.mkdir(parents=True, exist_ok=True)
    config.atomic_write(path, json.dumps(d, default=lambda v: v.item() if hasattr(v, "item") else str(v)))


def _dir_bytes(d: Path) -> int:
    total = 0
    for p in d.rglob("*"):
        try:
            total += p.stat().st_size if p.is_file() else 0
        except OSError:
            pass
    return total


def _prune_outputs(now: float | None = None) -> None:
    """Delete finished jobs' files beyond the newest ``KEEP_JOBS`` once they are older than
    ``KEEP_JOB_HOURS``, and beyond ``OUTPUT_MAX_GB`` (oldest first, the newest ``KEEP_JOBS``
    always kept); never touch running ones."""
    import shutil
    now = time.time() if now is None else now
    with _CHILDREN_LOCK:
        active = set(_CHILDREN) | {j.id for j in JOBS.values() if j.state in ("queued", "running")}
    try:
        dirs = [(d.stat().st_mtime, d) for d in config.OUTPUT_DIR.iterdir() if d.is_dir() and d.name not in active]
    except FileNotFoundError:
        return
    dirs.sort(key=lambda t: t[0], reverse=True)
    sizes = {d: _dir_bytes(d) for _, d in dirs}
    total = sum(sizes.values())
    doomed = [d for i, (mtime, d) in enumerate(dirs) if i >= KEEP_JOBS and now - mtime > KEEP_JOB_HOURS * 3600]
    total -= sum(sizes[d] for d in doomed)
    for _, d in reversed(dirs[KEEP_JOBS:]):         # over the size budget: oldest first
        if total <= OUTPUT_MAX_GB * 1e9:
            break
        if d not in doomed:
            doomed.append(d)
            total -= sizes[d]
    for d in doomed:
        shutil.rmtree(d, ignore_errors=True)
        j = JOBS.get(d.name)
        if j is not None and j.state != "running":
            JOBS.pop(d.name, None)


def _lookup(jid: str) -> Job | None:
    """A job from memory, or from its ``job.json`` after a restart; None if unknown or pruned."""
    if not valid_job_id(jid):
        return None
    j = JOBS.get(jid)
    if j is not None:
        return j
    path = _job_file(jid)
    if not path.exists():
        return None
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        j = Job(jid, Options(**d["options"]), state=d["state"], progress=d.get("progress", 1.0),
                message=d.get("message", ""), log=d.get("log", []), error=d.get("error"),
                error_kind=d.get("error_kind"), advice=d.get("advice", []), suggest=d.get("suggest", {}),
                result=d.get("result"), created=d.get("created", path.stat().st_mtime),
                seconds=d.get("seconds", 0.0), peak_mb=d.get("peak_mb", 0))
    except Exception:  # noqa: BLE001 - unreadable / from an older version
        return None
    return j


def _rss(proc) -> int:
    try:
        return int(proc.memory_info().rss)
    except Exception:  # noqa: BLE001 - exited meanwhile
        return 0


def _memory_check(job: Job, me) -> None:
    """Stop this job's worker if it, or all workers together, use more memory than allowed."""
    from . import limits
    mine = _rss(me)
    job.peak_mb = max(job.peak_mb, mine // 1_000_000)
    if mine > limits.JOB_MEMORY_MB * 1e6:
        raise _Stop("memory", f"The model needed more than {limits.JOB_MEMORY_MB / 1000:.1f} GB of memory.")
    with _CHILDREN_LOCK:
        others = [p for jid, p in _CHILDREN.items() if jid != job.id]
    total = mine + sum(_rss(p) for p in others)
    if total > (limits.MEMORY_MB - limits.SERVER_MB) * 1e6 and all(mine >= _rss(p) for p in others):
        raise _Stop("memory", "The server ran short of memory while building this model.")


def _run_job(job: Job) -> None:
    """Run a job in a worker process (``maptomesh.worker``) and watch it: progress comes over a
    pipe; the worker is killed when the job is cancelled, runs longer than ``limits.TIMEOUT_S``
    (time spent waiting for another model's build excluded) or uses too much memory."""
    import multiprocessing as mp

    import psutil

    from . import limits, worker

    try:
        _wait_for(_PIPELINE, job, "Queued behind other models")
    except RuntimeError:
        job.state, job.error, job.error_kind = "cancelled", "Cancelled", "cancelled"
        return
    proc = None
    has_build = False
    try:
        if job.cancel.is_set():
            raise _Stop("cancelled", "Cancelled")
        job.state = "running"
        o = job.opts
        from .providers.north_america import Hrdem
        in_canada = Hrdem().covers(AOI(o.lat, o.lon, o.width_m, o.height_m).lonlat_bbox())
        if o.source == "lidar" and not INDEX.ready and in_canada:
            job.set(0.0, "Waiting for LiDAR tile index")
            INDEX.start()
            while not INDEX.ready:
                if INDEX.state == "error":
                    raise _Stop("error", f"The LiDAR tile index is not available. {INDEX.error or ''}".strip())
                if job.cancel.is_set():
                    raise _Stop("cancelled", "Cancelled")
                job.set(0.0, f"Preparing LiDAR index: {INDEX.message}")
                time.sleep(0.5)
        ctx = mp.get_context("spawn")
        conn, child_conn = ctx.Pipe()
        proc = ctx.Process(target=worker.run, args=(job.id, o.model_dump(), child_conn), daemon=True,
                           name=f"maptomesh-job-{job.id}")
        proc.start()
        child_conn.close()
        me = psutil.Process(proc.pid)
        with _CHILDREN_LOCK:
            _CHILDREN[job.id] = me
        t0 = time.time()
        waited = 0.0
        wait_since = None
        result = None
        while result is None:
            msg = None
            try:
                if conn.poll(0.25):
                    msg = conn.recv()
            except (EOFError, OSError):
                proc.join(5)
                code = proc.exitcode
                if code in (-9, 137):
                    # killed from outside: on Linux the kernel's OOM killer (oom_score_adj = 1000)
                    raise _Stop("memory", "The server ran out of memory while building this model.")
                raise _Stop("error", f"The model's worker process stopped unexpectedly (exit code {code}).")
            if msg is not None:
                if msg[0] == "progress":
                    job.set(msg[1], msg[2])
                elif msg[0] == "ready_to_build":
                    wait_since = time.time()
                elif msg[0] == "done":
                    result = msg[1]
                    break
                elif msg[0] == "error":
                    raise _Stop(msg[1], msg[2], msg[3], bool(msg[4]) if len(msg) > 4 else False)
            if wait_since is not None and not has_build:
                if _BUILD_LOCK.acquire(blocking=False):
                    has_build = True
                    waited += time.time() - wait_since
                    wait_since = None
                    conn.send(("build",))
                elif job.message != "Data ready; waiting for the model ahead to finish building":
                    job.set(job.progress, "Data ready; waiting for the model ahead to finish building")
            if job.cancel.is_set():
                raise _Stop("cancelled", "Cancelled")
            job.seconds = time.time() - t0 - waited - (time.time() - wait_since if wait_since else 0.0)
            if job.seconds > limits.TIMEOUT_S:
                lim = limits.TIMEOUT_S
                span = f"{lim / 60:g} minutes" if lim >= 120 else f"{lim:.0f} seconds"
                raise _Stop("timeout", f"The model took longer than {span} and was stopped.")
            _memory_check(job, me)
        job.seconds = time.time() - t0 - waited
        for w in result.pop("warnings", []):
            job.log.append(scrub(w))
        job.result = result
        job.state = "done"
        job.set(1.0, "Done")
    except _Stop as s:
        job.error_kind = s.kind
        if s.kind == "cancelled" or job.cancel.is_set():
            job.state, job.error, job.error_kind = "cancelled", "Cancelled", "cancelled"
        else:
            # tracebacks and internal error text stay in the server log: users see the message
            # (written for them) or a generic one with an error id to report
            if s.public:
                job.error = scrub(s.message)
                if s.trace:
                    log.info("job %s stopped (%s): %s\n%s", job.id, s.kind, s.message, s.trace)
            else:
                job.error = _job_failed_message(_internal_error(f"job {job.id} failed: {s.message}", s.trace))
            job.state = "error"
            job.advice, job.suggest = limits.advice_for(s.kind, job.opts)
    except Exception:  # noqa: BLE001
        job.state, job.error_kind = "error", "error"
        job.error = _job_failed_message(_internal_error(f"job {job.id} failed", traceback.format_exc()))
    finally:
        if proc is not None:
            if proc.is_alive():
                proc.kill()
            proc.join(10)
        with _CHILDREN_LOCK:
            _CHILDREN.pop(job.id, None)
        if has_build:
            _BUILD_LOCK.release()
        _PIPELINE.release()
        try:
            if job.state == "done":
                _persist(job)
            elif (config.OUTPUT_DIR / job.id).exists():   # partial files of a stopped job
                import shutil
                shutil.rmtree(config.OUTPUT_DIR / job.id, ignore_errors=True)
            _prune_outputs()
        except OSError:
            pass


# ---------------------------------------------------------------- API

@app.on_event("startup")
def _startup() -> None:
    INDEX.start()
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _prune_outputs()   # finished jobs survive restarts (job.json); keep the most recent ones


_BROWSER_PKG: dict = {}
_SERVER_ONLY = {"server.py", "reviews.py", "shares.py", "proxy.py", "estimate.py", "device_failures.py"}


IMMUTABLE = {"Cache-Control": "public, max-age=31536000, immutable"}


def _browser_package() -> tuple[bytes, str]:
    """The ``maptomesh`` package's Python sources as a zip, and its version (content hash);
    rebuilt when a source file changes."""
    import hashlib
    import io
    import zipfile

    root = Path(__file__).resolve().parent
    files = sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts and p.name not in _SERVER_ONLY)
    stamp = (len(files), max(p.stat().st_mtime for p in files))
    if _BROWSER_PKG.get("stamp") != stamp:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for p in files:
                z.write(p, "maptomesh/" + p.relative_to(root).as_posix())
        data = buf.getvalue()
        _BROWSER_PKG.update(stamp=stamp, data=data, version=hashlib.sha256(data).hexdigest()[:20])
    return _BROWSER_PKG["data"], _BROWSER_PKG["version"]


@app.get("/api/browser/package.zip")
def browser_package(request: Request, v: str = ""):
    """The package for generating in the browser (Pyodide; see docs/plans/browser-wasm.md). The
    worker asks for ``?v=<version>`` from ``/api/browser/config`` and keeps it for good (the
    version changes with the code); without it, browsers revalidate (ETag)."""
    data, version = _browser_package()
    etag = f'"{version}"'
    headers = dict(IMMUTABLE) if v == version else {"Cache-Control": "no-cache"}
    headers["ETag"] = etag
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers=headers)
    return Response(data, media_type="application/zip", headers=headers)


_FILE_VERSIONS: dict[tuple, str] = {}


def _file_version(path: Path) -> str:
    import hashlib
    st = path.stat()
    key = (str(path), st.st_mtime_ns, st.st_size)
    if key not in _FILE_VERSIONS:
        _FILE_VERSIONS[key] = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    return _FILE_VERSIONS[key]


def _wheels() -> dict[str, Path]:
    """Wheels for Pyodide in ``web/pyodide`` (native/build_pyodide.sh, build_lazrs_pyodide.sh,
    laspy from PyPI): the newest of each package."""
    newest: dict[str, Path] = {}
    for w in sorted((config.WEB_DIR / "pyodide").glob("*.whl"), key=lambda p: p.stat().st_mtime):
        newest[w.name.split("-", 1)[0]] = w
    return newest


@app.get("/api/browser/files/{version}/{name}")
def browser_file(version: str, name: str):
    """A wheel under a content-versioned URL: browsers keep it until it changes."""
    w = _wheels().get(name.split("-", 1)[0])
    if w is None or w.name != name or _file_version(w) != version:
        raise HTTPException(404, "No such file (or a newer version exists)")
    return FileResponse(w, media_type="application/zip", headers=IMMUTABLE)


@app.get("/api/browser/config")
def browser_config():
    """What the browser worker needs: the proxy and the hosts that need it, and versioned URLs of
    the wheels and the package (cached by the worker until they change)."""
    from . import proxy
    _, version = _browser_package()
    return {"proxy": "/api/proxy?url=", "proxy_first": list(proxy.NO_CORS),
            "wheels": {k: f"/api/browser/files/{_file_version(w)}/{w.name}" for k, w in _wheels().items()},
            "package": f"/api/browser/package.zip?v={version}"}


# ---------------------------------------------------------------- per-client limits

def _rate(limit, request: Request, message: str) -> str:
    """Count a request against ``limit`` (``limits.RateLimit``); 429 with Retry-After when over.
    Returns the client key."""
    return _rate_key(limit, client_key(request), message)


def _rate_key(limit, key: str, message: str) -> str:
    wait = limit.hit(key)
    if wait:
        raise HTTPException(429, message, headers={"Retry-After": str(int(math.ceil(wait)))})
    return key


@contextmanager
def _slot(slots, key: str):
    """Hold one of ``slots`` (``limits.Slots``) or fail at once: 429 when this client already has
    its share in use, 503 when the server is busy; never waits on a server thread."""
    why = slots.enter(key)
    if why == "client":
        raise HTTPException(429, "Another request from here is still being worked on; try again in a moment",
                            headers={"Retry-After": "2"})
    if why:
        raise HTTPException(503, "Busy; please try again in a moment", headers={"Retry-After": "5"})
    try:
        yield
    finally:
        slots.leave(key)


def check_bbox(w: float, s: float, e: float, n: float, max_deg2: float | None = None) -> None:
    """400 unless (w, s, e, n) is a finite lon/lat box with w < e and s < n (and at most
    ``max_deg2`` square degrees)."""
    if not all(math.isfinite(v) for v in (w, s, e, n)) or not (-180 <= w < e <= 180 and -90 <= s < n <= 90):
        raise HTTPException(400, "Box out of range")
    if max_deg2 is not None and (e - w) * (n - s) > max_deg2:
        raise HTTPException(400, "Box too large")


@app.get("/api/lidar/tiles")
def lidar_tiles(request: Request, w: float, s: float, e: float, n: float):
    """NRCan CanElevation tiles in a lon/lat box, for jobs built in the browser (the index is 22 MB)."""
    import dataclasses

    from . import limits
    check_bbox(w, s, e, n, 25)
    _rate(limits.COVERAGE_RATE, request, "Too many requests from here; please try again later")
    if not INDEX.ready:
        raise HTTPException(503, "The LiDAR tile index is not available yet; try again in a minute.")
    tiles = INDEX.get().query(w, s, e, n, limit=5001)
    return {"tiles": [dataclasses.asdict(t) for t in tiles]}


# Reading an upstream blocks a thread (requests is synchronous) for as long as the upstream takes.
# The proxy's threads come from a limiter of its own, not the pool that serves every other route
# and static file, so slow upstreams can stall the proxy but never the site; one client (and one
# IPv6 /48) holds only a few of them at a time - its other requests wait without a thread.
PROXY_THREADS = int(os.environ.get("MAPTOMESH_PROXY_THREADS", "24"))
PROXY_CLIENT_THREADS = 4
_PROXY_LIMITER: list = []                   # [anyio.CapacityLimiter], made on first use
_PROXY_KEYS: dict[str, list] = {}           # key -> [CapacityLimiter, requests using it] (event loop only)
# a navigation (address bar, link, frame) never needs the proxy: only the build's own fetches do
_NAVIGATION = {"document", "iframe", "frame", "embed", "object"}


@contextlib.asynccontextmanager
async def _proxy_threads(key: str):
    """Hold this client's and its /48's share of the proxy threads (``PROXY_CLIENT_THREADS``)."""
    import anyio

    from . import limits
    if key == limits.LOCAL:
        yield
        return
    group = limits.group_of(key)
    shares = [(key, PROXY_CLIENT_THREADS)] + ([(group, int(PROXY_CLIENT_THREADS * limits.GROUP_FACTOR))] if group else [])
    entries = []
    for k, n in shares:
        e = _PROXY_KEYS.get(k)
        if e is None:
            e = _PROXY_KEYS[k] = [anyio.CapacityLimiter(n), 0]
        e[1] += 1
        entries.append((k, e))
    try:
        async with contextlib.AsyncExitStack() as stack:
            for _, e in entries:
                await stack.enter_async_context(e[0])
            yield
    finally:
        for k, e in entries:
            e[1] -= 1
            if e[1] <= 0 and _PROXY_KEYS.get(k) is e:
                del _PROXY_KEYS[k]


async def _proxy_blocking(key: str, fn, *args):
    """``fn(*args)`` on one of the proxy's threads, within the client's share."""
    import anyio
    if not _PROXY_LIMITER:
        _PROXY_LIMITER.append(anyio.CapacityLimiter(PROXY_THREADS))
    async with _proxy_threads(key):
        return await anyio.to_thread.run_sync(fn, *args, limiter=_PROXY_LIMITER[0])


def _next_chunk(it):
    try:
        return next(it)
    except StopIteration:
        return None


async def _proxy_body(it, key: str):
    try:
        while (chunk := await _proxy_blocking(key, _next_chunk, it)) is not None:
            yield chunk
    finally:
        it.close("dropped")       # no-op once finished; a visitor who left frees the slot at once


@app.api_route("/api/proxy", methods=["GET", "HEAD", "POST"])
async def data_proxy(request: Request, url: str = Query(..., max_length=4000)):
    """Stream a data file from an allowlisted host that has no CORS headers (maptomesh/proxy.py)."""
    from fastapi.responses import StreamingResponse

    from . import proxy
    # refuse before reading anything: only the page's own requests for allowlisted data URLs,
    # within the client's hourly request allowance; the body (POST queries of catalogue APIs) is
    # capped by _BodyLimit
    if request.headers.get("sec-fetch-dest", "").lower() in _NAVIGATION:
        raise HTTPException(403, "The data proxy only answers the page's own requests")
    if not proxy.allowed(url, request.method):
        raise HTTPException(403, "Not a data source this server forwards")
    client = client_key(request)
    wait = proxy.rate_wait(client, request.method)
    if wait:
        raise HTTPException(429, "Too many requests through the data proxy from here; please try again later",
                            headers={"Retry-After": str(int(math.ceil(wait)))})
    body = await request.body() if request.method == "POST" else None
    if body is not None and len(body) > BODY_LIMITS.get("/api/proxy", BODY_LIMIT):
        raise HTTPException(413, "Request body too large")
    try:
        status, headers, it = await _proxy_blocking(client, proxy.open_upstream, url, request.method,
                                                    dict(request.headers), body, client)
    except proxy.Refused as e:
        retry = {"Retry-After": str(int(math.ceil(e.retry_after)))} if e.retry_after else None
        raise HTTPException(e.status, str(e), headers=retry) from None
    except Exception as e:  # noqa: BLE001 - upstream unreachable
        raise HTTPException(502, f"Data source unreachable: {type(e).__name__}") from None
    if request.method == "HEAD":
        it.close("done")          # no body: release the client's slot
        return Response(status_code=status, headers=headers)
    return StreamingResponse(_proxy_body(it, client), status_code=status, headers=headers)


@app.get("/api/status")
def status() -> dict:
    return {"version": __version__, "index": {"state": INDEX.state, "ready": INDEX.ready,
                                              "progress": INDEX.progress, "message": INDEX.message,
                                              "error": INDEX.error},
            "projects_ready": PROJECTS_GEOJSON.exists(), "limits": _limits_public()}


def _limits_public() -> dict:
    from . import limits
    return limits.public()


@app.get("/api/projects")
def projects():
    if not PROJECTS_GEOJSON.exists():
        raise HTTPException(503, "Project coverage not downloaded yet")
    return FileResponse(PROJECTS_GEOJSON, media_type="application/geo+json")


# The area fields as a job's (``Options``: same types, ranges and defaults), so coverage can't be
# asked about areas no job could build (e.g. a 1500 km box, lat 95)
AREA_FIELDS = ("lat", "lon", "width_m", "height_m", "grid_cells", "us_source", "source", "dem_preference")
AreaIn = create_model(
    "AreaIn", __config__=ConfigDict(allow_inf_nan=False),
    ground_only=(bool, False),          # bare terrain not coloured by land cover
    **{k: (Options.model_fields[k].annotation, copy.deepcopy(Options.model_fields[k])) for k in AREA_FIELDS})


@app.post("/api/coverage")
def coverage(a: AreaIn, request: Request) -> dict:
    from . import limits
    key = _rate(limits.COVERAGE_RATE, request, "Too many coverage checks from here; please try again later")
    with _slot(limits.COVERAGE_SLOTS, key):
        return _coverage(a)


def _coverage(a) -> dict:
    aoi = AOI(a.lat, a.lon, a.width_m, a.height_m)
    grid = Grid.for_aoi(aoi, max(50, min(a.grid_cells, config.MAX_GRID_CELLS)))
    from . import providers  # noqa: F401 - registers elevation providers
    from .dem import describe
    from .pointcloud import find_sources
    from .providers.north_america import Hrdem

    out = {"grid": {"nx": grid.nx, "ny": grid.ny, "step_m": round(grid.step, 3)}}
    bbox = aoi.lonlat_bbox(margin=grid.step * 2)
    area_km2 = a.width_m * a.height_m / 1e6
    shared = a.dem_preference != "official"
    out["dem_sources"] = describe(aoi, grid, dsm=not a.ground_only, shared_first=shared)
    models_suffice = False
    if a.source == "lidar" and a.ground_only and shared:
        from .model import MAPTERHORN_FIRST
        from .providers.mapterhorn import models_suffice as _suffice
        models_suffice = MAPTERHORN_FIRST and _suffice(aoi, grid)
    out["models_suffice"] = models_suffice
    if not models_suffice and not INDEX.ready and Hrdem().covers(bbox):  # Canadian point clouds need the tile index
        out.update({"ready": False, "message": INDEX.message})
        return out
    if a.source != "lidar" or models_suffice:
        out.update({"ready": True, "tiles": 0, "ept_projects": 0, "max_tiles": config.MAX_TILES,
                    "area_km2": round(area_km2, 2), "errors": [], "projects": []})
        return out
    from .limits import lidar_caps

    class _G:   # what lidar_caps reads
        grid_cells = a.grid_cells
    max_tiles, max_ept = lidar_caps(_G)
    tiles, projects, errors = find_sources(bbox, INDEX.get() if INDEX.ready else None, a.us_source,
                                           max_tiles + 1)
    cnt = Counter((t.project, t.year) for t in tiles)
    plist = [{"name": p, "year": y, "tiles": n} for (p, y), n in cnt.items()]
    # AWS Entwine projects have no tiles: report the projects (their octree is read per job)
    plist += [{"name": p.label, "year": p.year, "tiles": None} for p in projects]
    out.update({
        "ready": True, "tiles": len(tiles), "ept_projects": len(projects), "max_tiles": max_tiles,
        "max_ept_km2": max_ept,
        "area_km2": round(a.width_m * a.height_m / 1e6, 2), "errors": [scrub(x, 300) for x in errors],
        "projects": sorted(plist, key=lambda x: -x["year"]),
    })
    return out


@app.get("/api/projects/usgs")
def projects_usgs(request: Request):
    """Footprints of every USGS 3DEP project in the AWS Entwine collection (simplified)."""
    from . import limits
    from .usgs_ept import INDEX as EPT_INDEX, Busy

    _rate(limits.COVERAGE_RATE, request, "Too many requests from here; please try again later")
    try:
        return Response(EPT_INDEX.overlay(wait=False), media_type="application/geo+json",
                        headers={"Content-Encoding": "gzip"})
    except Busy as e:
        raise HTTPException(503, str(e), headers={"Retry-After": "30"}) from None
    except Exception:  # noqa: BLE001
        eid = _internal_error("USGS 3DEP project list unavailable", traceback.format_exc())
        raise HTTPException(502, f"USGS 3DEP project list unavailable (error id {eid})") from None


@app.get("/api/coverage/usgs")
def coverage_usgs(request: Request, w: float = Query(...), s: float = Query(...), e: float = Query(...),
                  n: float = Query(...)):
    """USGS 3DEP tile footprints in a map view (only when zoomed in: one search per view)."""
    from . import limits
    from .usgs3dep import footprints, in_us

    if not all(math.isfinite(v) for v in (w, s, e, n)) or not (w < e and -90 <= s < n <= 90):
        raise HTTPException(400, "Box out of range")
    bbox = (w, s, e, n)
    if w < -180 or e > 180 or not in_us(bbox):     # (a map view across the antimeridian)
        return {"type": "FeatureCollection", "features": [], "in_us": False}
    if (e - w) * (n - s) > 0.35:
        return {"type": "FeatureCollection", "features": [], "in_us": True, "too_large": True}
    key = _rate(limits.COVERAGE_RATE, request, "Too many coverage checks from here; please try again later")
    with _slot(limits.COVERAGE_SLOTS, key):
        try:
            gj = footprints(bbox)
        except Exception:  # noqa: BLE001
            eid = _internal_error("USGS 3DEP search failed", traceback.format_exc())
            raise HTTPException(502, f"USGS 3DEP search failed (error id {eid})") from None
    gj["in_us"] = True
    return gj


def _refuse(kind: str, message: str, advice: list[str], suggest: dict | None = None, status: int = 422,
            headers: dict | None = None):
    """A job refused before it starts: same fields as a failed job, for the UI's failure dialog."""
    return JSONResponse({"detail": message, "error_kind": kind, "advice": advice, "suggest": suggest or {}},
                        status_code=status, headers=headers)


_JOBS_LOCK = threading.Lock()


@app.post("/api/jobs")
def create_job(o: Options, request: Request):
    """Queue a job. The reply's ``cancel_token`` is the only way to cancel it (``job_cancel``):
    job ids appear in shared ``?job=`` links."""
    from . import limits
    try:
        limits.check(o)
    except limits.LimitError as e:
        return _refuse(e.kind, e.message, e.advice, e.suggest)
    key = client_key(request)
    group = limits.group_of(key)
    with _JOBS_LOCK:
        active = [j for j in JOBS.values() if j.state in ("queued", "running")]
        mine = sum(1 for j in active if j.client == key)
        ours = sum(1 for j in active if group and limits.group_of(j.client) == group)
        if key != limits.LOCAL:            # this machine (tests run in parallel): no queue limits
            if mine >= limits.JOBS_PER_CLIENT or (group and ours >= limits.JOBS_PER_CLIENT * limits.GROUP_FACTOR):
                return _refuse("busy", f"You already have {max(mine, 1)} model{'s' if mine > 1 else ''} queued or "
                               "being built on the server; wait for one to finish (or cancel it).",
                               ["Wait for your other model to finish, or cancel it, then try again."], status=429,
                               headers={"Retry-After": "30"})
            # a client with a model in the queue gets a second one only while a quarter of the
            # queue is free: that quarter stays for people who have none yet
            if len(active) >= limits.MAX_QUEUE or (mine and len(active) >= limits.MAX_QUEUE - limits.MAX_QUEUE // 4):
                advice, _ = limits.advice_for("busy", o)
                return _refuse("busy", f"The server is busy: {len(active)} models are queued or being built.",
                               advice, status=429, headers={"Retry-After": "60"})
            wait = limits.JOB_RATE.hit(key)
            if wait:
                return _refuse("busy", "You have built many models on the server in the last hour; please try "
                               "again later (or build on this device).",
                               [f"Try again in about {max(1, round(wait / 60))} minutes."], status=429,
                               headers={"Retry-After": str(int(math.ceil(wait)))})
        job = Job(uuid.uuid4().hex[:12], o, client=key, cancel_token=secrets.token_urlsafe(18))
        JOBS[job.id] = job
        # keep the newest 50 in memory; never drop one still queued or running
        done = [jid for jid, j in JOBS.items() if j.state not in ("queued", "running")]
        for jid in done[:max(0, len(JOBS) - 50)]:
            del JOBS[jid]
    threading.Thread(target=_run_job, args=(job,), daemon=True).start()
    return {"id": job.id, "cancel_token": job.cancel_token}


def _job(jid: str) -> Job:
    j = _lookup(jid) if valid_job_id(jid) else None
    if not j:
        raise HTTPException(404, "Unknown job (finished models are kept for a limited time)")
    return j


def _done_file(jid: str, key: str):
    """Path of a finished job's file (``result[key]``); 409 while running, 404 / 410 if gone."""
    j = _job(jid)
    if j.state != "done":
        raise HTTPException(409, "Not ready")
    name = (j.result or {}).get(key)
    if not name or not isinstance(name, str):
        raise HTTPException(404, "No such file for this job")
    path = job_path(jid, name)
    if not path.exists():
        raise HTTPException(410, "This model's files were removed (finished models are kept for a limited time); generate it again")
    return path, name


@app.get("/api/jobs/{jid}")
def job_status(jid: str) -> dict:
    return _job(jid).public()


@app.post("/api/jobs/{jid}/cancel")
def job_cancel(jid: str, request: Request) -> dict:
    """Cancel a job: only with the ``cancel_token`` its submission returned (``X-Cancel-Token``)."""
    j = _job(jid)
    given = request.headers.get("x-cancel-token", "")
    if not j.cancel_token or not hmac.compare_digest(given.encode(), j.cancel_token.encode()):
        raise HTTPException(403, "Only the page that started this model can cancel it")
    j.cancel.set()
    return {"ok": True}


@app.get("/api/jobs/{jid}/model.3mf")
def job_3mf(jid: str):
    path, name = _done_file(jid, "file")
    return FileResponse(path, media_type="model/3mf", filename=name)


@app.get("/api/jobs/{jid}/base.3mf")
def job_base_3mf(jid: str):
    path, name = _done_file(jid, "base_file")
    return FileResponse(path, media_type="model/3mf", filename=name)


@app.get("/api/jobs/{jid}/thumbnail.png")
def job_thumb(jid: str):
    path = job_path(jid, "thumbnail.png")
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path, media_type="image/png")


@app.get("/api/jobs/{jid}/stl.zip")
def job_stl(jid: str):
    path, name = _done_file(jid, "stl_file")
    return FileResponse(path, media_type="application/zip", filename=name)


@app.get("/api/jobs/{jid}/preview.bin")
def job_preview(jid: str, lod: str = "high"):
    """The 3D-preview mesh (``export.preview_bytes``, gzip-encoded) at a level of detail, written
    by the job's worker; a job's preview never changes, so browsers may cache it for good."""
    from .export import PREVIEW_LODS
    if lod not in PREVIEW_LODS or not valid_job_id(jid):
        raise HTTPException(404, "Unknown preview")
    path = job_path(jid, f"preview_{lod}.bin.gz")
    if not path.exists():
        j = _lookup(jid)
        raise HTTPException(409 if j is not None and j.state != "done" else 404, "Preview not available")
    return FileResponse(path, media_type="application/octet-stream",
                        headers={"Content-Encoding": "gzip", "Cache-Control": "public, max-age=31536000, immutable"})


@app.post("/api/estimate")
def download_estimate(o: Options, request: Request) -> dict:
    """About how much data building this model downloads (``maptomesh.estimate``): the page warns
    before building on a phone."""
    from . import estimate as est
    key = _rate(est.RATE, request, "Too many estimates from here; please try again later")
    with _slot(est.SLOTS, key):
        return est.estimate(o)


# ---------------------------------------------------------------- device failures

class DeviceFailureIn(BaseModel):
    kind: str = Field("error", max_length=40)
    message: str = Field("", max_length=2000)
    trace: str = Field("", max_length=8000)
    stage: str = Field("", max_length=300)
    options: dict | None = None
    env: dict = Field(default_factory=dict)


class _PythonEnv(BaseModel):
    model_config = ConfigDict(extra="ignore")
    ready: bool | None = None
    jspi: bool | None = None
    native: bool | None = None
    runtime_cached: bool | None = None
    pyodide: str | None = Field(None, max_length=40)


class DeviceEnv(BaseModel):
    """The browser facts a device-failure report may carry (web/app.js reportDeviceFailure);
    anything else is dropped, so what's stored and printed has known types."""
    model_config = ConfigDict(extra="ignore", allow_inf_nan=False)
    memory_mb: float | None = None
    device_memory_gb: float | None = None
    cores: int | None = Field(None, ge=0, le=4096)
    phone: bool | None = None
    secure: bool | None = None
    seconds: float | None = None
    python: _PythonEnv = Field(default_factory=_PythonEnv)
    log: list[Annotated[str, Field(max_length=500)]] = Field(default_factory=list, max_length=30)

    @field_validator("log", mode="before")
    @classmethod
    def _trim_log(cls, v):
        """The last lines, each cut to length (long lines are kept, not the reason to drop the facts)."""
        if not isinstance(v, list):
            raise ValueError("log must be a list")
        return [x[:500] for x in v[-30:] if isinstance(x, str)]


@app.post("/api/device-failures")
def device_failure(f: DeviceFailureIn, request: Request) -> dict:
    """A build that failed in the user's browser (``maptomesh.device_failures``), for review."""
    from . import device_failures as df
    if df.rate_limited(client_key(request)):
        raise HTTPException(429, "Too many reports from here")
    try:
        env = DeviceEnv.model_validate(f.env).model_dump()
    except Exception:  # noqa: BLE001 - keep the report without unusable browser facts
        env = {"invalid": True}
    try:
        options = Options(**f.options).model_dump() if f.options else None
    except Exception:  # noqa: BLE001 - keep the report without unusable settings
        options = None
    kind = f.kind if f.kind in ("memory", "too_large", "startup", "error") else "error"
    rid = df.store().add(kind=kind, message=f.message, trace=f.trace, stage=f.stage, options=options, env=env,
                         user_agent=request.headers.get("user-agent", "")[:300], app_version=__version__)
    return {"ok": True, "id": rid}


# ---------------------------------------------------------------- share links

@app.post("/api/shares")
def create_share(o: Options, request: Request) -> dict:
    """Store a model's settings for a share link (``/?s=<id>``; see ``maptomesh.shares``)."""
    from . import shares
    if shares.rate_limited(client_key(request)):
        raise HTTPException(429, "Too many share links from here; please try again later")
    return {"id": shares.store().add(o.model_dump())}


@app.get("/api/shares/{sid}")
def get_share(sid: str) -> dict:
    from . import shares
    opts = shares.store().get(sid)
    if opts is None:
        raise HTTPException(404, "Unknown share link")
    return {"id": sid, "options": opts}


# ---------------------------------------------------------------- reviews

class ReviewIn(BaseModel):
    rating: int | None = Field(None, ge=1, le=5)
    problems: list[str] = Field(default_factory=list, max_length=12)
    comment: str = Field("", max_length=4000)
    followup: bool = False
    name: str = Field("", max_length=200)
    email: str = Field("", max_length=254)
    website: str = Field("", max_length=200)   # honeypot: hidden from people, filled by bots
    snapshot: dict | None = None                # the browser's copy of the job (server restarted)


_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
REVIEWS_PER_HOUR = 20      # per client (limits.client_key)
REVIEWS_GLOBAL_PER_HOUR = 200   # from everyone: filling the store (it refuses when full) takes days
REVIEWS_PER_JOB = 10
SNAPSHOT_MAX = 200_000     # bytes of the browser's copy of a job (JSON) accepted
# what is kept of the browser's copy of a job's result: the parts a reviewer reads (stats, data
# sources, notes), within CLIENT_RESULT_MAX bytes - the largest parts are dropped first
CLIENT_RESULT_KEYS = ("stats", "data", "notes", "filaments", "file_size", "browser")
CLIENT_RESULT_MAX = 8_000


REVIEW_RATE = RateLimit(REVIEWS_PER_HOUR)
REVIEW_GLOBAL = RateLimit(REVIEWS_GLOBAL_PER_HOUR, group_factor=0)


def _job_snapshot(job: Job) -> tuple[str, dict, dict]:
    result = {"result": job.result, "error": job.error, "log": job.log[-40:],
              "created": job.created, "message": job.message}
    return job.state, job.opts.model_dump(), result


def _client_result(res) -> dict | None:
    if res is None:
        return None
    if not isinstance(res, dict):
        raise HTTPException(400, "Invalid job result in snapshot")
    out = {k: res[k] for k in CLIENT_RESULT_KEYS if k in res}
    sizes = {k: len(json.dumps(v, default=str)) for k, v in out.items()}
    for k in sorted(sizes, key=sizes.get, reverse=True):
        if sum(sizes.values()) <= CLIENT_RESULT_MAX:
            break
        del out[k], sizes[k]
    return out


def _client_snapshot(snap: dict) -> tuple[str, dict, dict]:
    if len(json.dumps(snap)) > SNAPSHOT_MAX:
        raise HTTPException(413, "Job snapshot too large")
    try:
        opts = Options(**(snap.get("options") or {})).model_dump()
    except Exception:  # noqa: BLE001
        raise HTTPException(400, "Invalid job settings in snapshot") from None
    state = snap.get("state") if snap.get("state") in ("done", "error", "cancelled") else "unknown"
    log, error = snap.get("log") or [], snap.get("error")
    if not isinstance(log, list) or not all(isinstance(x, str) for x in log):
        raise HTTPException(400, "Invalid job log in snapshot")
    if error is not None and not isinstance(error, str):
        raise HTTPException(400, "Invalid job error in snapshot")
    result = {"result": _client_result(snap.get("result")), "error": error[:2000] if error else error,
              "log": [x[:300] for x in log[-40:]]}
    return state, opts, result


@app.post("/api/jobs/{jid}/review")
def job_review(jid: str, rv: ReviewIn, request: Request) -> dict:
    """Store a review of a job with the job's settings and result (see ``maptomesh.reviews``).

    A job this server ran is reviewed with the server's own record of it. Any other well-formed
    id needs the browser's copy of the job (``snapshot``; only its readable parts are kept): models
    built on the device (the default; the page gives them an id in the same format) and server
    jobs from before a restart. The store keeps a row and size budget (``reviews.KEEP``,
    ``reviews.MAX_MB``) and refuses new reviews when it is full: anonymous posts never delete
    stored ones."""
    from . import limits
    from .reviews import PROBLEM_KEYS, StoreFull, store

    if not valid_job_id(jid):
        raise HTTPException(404, "Unknown job")
    problems = list(dict.fromkeys(rv.problems))
    if any(p not in PROBLEM_KEYS for p in problems):
        raise HTTPException(400, "Unknown problem category")
    comment, name, email = rv.comment.strip(), rv.name.strip(), rv.email.strip()
    if rv.rating is None and not problems and not comment:
        raise HTTPException(400, "Give a score, pick a problem or write a comment")
    if rv.followup and not _EMAIL.match(email):
        raise HTTPException(400, "Enter an email address so we can follow up (or untick follow-up)")
    if rv.website:   # bot: pretend it worked
        return {"ok": True, "id": 0}
    key = _rate(REVIEW_RATE, request, "Too many reviews from here; please try again later")
    if key != limits.LOCAL:
        _rate_key(REVIEW_GLOBAL, "all", "Too many reviews are coming in right now; please try again later")
    st = store()
    if st.count_for_job(jid) >= REVIEWS_PER_JOB:
        raise HTTPException(429, "This model has already been reviewed several times")
    job = _lookup(jid)
    if job is not None:
        state, options, result = _job_snapshot(job)
        source = "server"
    elif rv.snapshot:
        state, options, result = _client_snapshot(rv.snapshot)
        source = "client"
    else:
        raise HTTPException(404, "This job is no longer available; generate it again to review it")
    try:
        rid = st.add(job_id=jid, job_state=state, rating=rv.rating, problems=problems, comment=comment,
                     wants_followup=rv.followup, contact_name=name or None, contact_email=email or None,
                     options=options, result=result, snapshot_source=source, app_version=__version__,
                     user_agent=request.headers.get("user-agent", "")[:300])
    except StoreFull:
        log.warning("review store full (MAPTOMESH_KEEP_REVIEWS / MAPTOMESH_REVIEWS_MAX_MB): new reviews refused")
        raise HTTPException(503, "Reviews can't be saved right now; please try again later") from None
    return {"ok": True, "id": rid}


@app.exception_handler(ValueError)
def _value_error(req, exc: ValueError):
    """An invalid value deep in a request's work: its text (paths, internals) stays in the log."""
    eid = _internal_error(f"{req.method} {req.url.path}: ValueError",
                          "".join(traceback.format_exception(exc)))
    return JSONResponse({"detail": f"Invalid request (error id {eid})"}, status_code=400)


@app.exception_handler(RequestValidationError)
def _invalid_request(_req, exc: RequestValidationError):
    """422 with FastAPI's error list (the page turns ``loc`` / ``msg`` / ``input`` into a message),
    minus the parts that can't or shouldn't be echoed: NaN / Infinity inputs (which would make
    the default handler fail with a 500), long inputs and exception context."""
    from fastapi.encoders import jsonable_encoder
    detail = []
    for e in exc.errors()[:50]:
        d = {k: e[k] for k in ("type", "loc", "msg") if k in e}
        if "input" in e:
            try:
                text = json.dumps(jsonable_encoder(e["input"]), allow_nan=False)
                if len(text) <= 200:
                    d["input"] = json.loads(text)
            except (TypeError, ValueError):
                d["input"] = str(e["input"])[:50]
        detail.append(jsonable_encoder(d))
    return JSONResponse({"detail": detail}, status_code=422)


@app.exception_handler(Exception)
def _unexpected(req, exc: Exception):
    """Any other failure: a generic 500 with an error id; the traceback goes to the log."""
    eid = _internal_error(f"{req.method} {req.url.path}: {type(exc).__name__}",
                          "".join(traceback.format_exception(exc)))
    return JSONResponse({"detail": f"Internal server error (error id {eid})"}, status_code=500)


SITE_TITLE = "MapToMesh: 3D-printable maps from LiDAR"
SITE_DESCRIPTION = ("Turn LiDAR and elevation data into 3D-printable, multi-colour terrain and city models (3MF) "
                    "of anywhere on Earth, with water, roads, buildings and trees.")
EXAMPLE_IMAGE = ("/static/og-image.jpg", 1200, 630, "image/jpeg",
                 "3D-printable model of San Francisco made with MapToMesh: streets, parks and the bay over the hills")


@app.get("/")
def index(request: Request, job: str | None = None, s: str | None = None):
    """The UI, with link-preview metadata (absolute URLs) filled in."""
    import html

    base = config.PUBLIC_URL or str(request.base_url).rstrip("/")
    title, desc, page = SITE_TITLE, SITE_DESCRIPTION, f"{base}/"
    path, w, h, mime, alt = EXAMPLE_IMAGE
    image = f"{base}{path}"
    j = _lookup(job) if job and valid_job_id(job) else None
    if j is not None and j.state == "done" and j.result:
        o = j.opts
        title = f"{o.name} - MapToMesh"
        desc = (f"3D-printable model of {o.width_m / 1000:g} x {o.height_m / 1000:g} km around "
                f"{o.lat:.4f}, {o.lon:.4f} ({j.result['stats'].get('scale', '')}), made with MapToMesh.")
        page = f"{base}/?job={j.id}"
        image, w, h, mime, alt = f"{base}/api/jobs/{j.id}/thumbnail.png", None, None, "image/png", f"Top view of the {o.name} model"
    elif s:
        from . import shares
        so = shares.store().get(s, opened=False) if shares.valid_id(s) else None
        if so is not None:
            title = f"{so.get('name', 'MapToMesh')} - MapToMesh"
            desc = (f"3D-printable model of {so['width_m'] / 1000:g} x {so['height_m'] / 1000:g} km around "
                    f"{so['lat']:.4f}, {so['lon']:.4f}, made with MapToMesh. Open the link to build it.")
            page = f"{base}/?s={s}"
    esc = lambda v: html.escape(str(v), quote=True)  # noqa: E731 - names are user input
    size_tags = [f'  <meta property="og:image:type" content="{mime}">']
    if w:
        size_tags += [f'  <meta property="og:image:width" content="{w}">',
                      f'  <meta property="og:image:height" content="{h}">']
    size = "\n".join(size_tags)
    page_html = (config.WEB_DIR / "index.html").read_text(encoding="utf-8")
    for key, val in (("__OG_IMAGE_META__", size), ("__OG_TITLE__", esc(title)), ("__OG_DESCRIPTION__", esc(desc)),
                     ("__PAGE_URL__", esc(page)), ("__OG_IMAGE_ALT__", esc(alt)), ("__OG_IMAGE__", esc(image))):
        page_html = page_html.replace(key, val)
    return HTMLResponse(page_html)


app.mount("/static", StaticFiles(directory=config.WEB_DIR), name="static")
