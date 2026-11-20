"""FastAPI app: map UI + generation jobs."""

from __future__ import annotations

import json
import os
import re
import threading
import time
import traceback
import uuid
from collections import Counter, OrderedDict, deque
from pathlib import Path
from dataclasses import dataclass, field

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__, config
from .geo import AOI, Grid
from .lidar_index import INDEX, PROJECTS_GEOJSON
from .options import Options

app = FastAPI(title="MapToMesh", version=__version__)


@app.middleware("http")
async def _no_cache_ui(request, call_next):
    resp = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        resp.headers["Cache-Control"] = "no-cache"  # always revalidate the UI after updates
    return resp


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

    def set(self, f: float, msg: str) -> None:
        self.progress = max(0.0, min(1.0, f))
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
KEEP_JOBS = int(os.environ.get("MAPTOMESH_KEEP_JOBS", "10"))   # finished jobs kept on disk
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
    """A job stopped by the server: ``kind`` is the reason (timeout / memory / cancelled / error)."""

    def __init__(self, kind: str, message: str, trace: str = ""):
        super().__init__(message)
        self.kind, self.message, self.trace = kind, message, trace


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


def _job_file(jid: str):
    return config.OUTPUT_DIR / jid / "job.json"


def _persist(job: Job) -> None:
    """The finished job's metadata next to its files: links keep working after a restart."""
    d = job.public()
    d["created"] = job.created
    path = _job_file(job.id)
    path.parent.mkdir(parents=True, exist_ok=True)
    config.atomic_write(path, json.dumps(d, default=lambda v: v.item() if hasattr(v, "item") else str(v)))


def _prune_outputs() -> None:
    """Keep the files of the ``KEEP_JOBS`` most recent finished jobs; never touch running ones."""
    import shutil
    with _CHILDREN_LOCK:
        active = set(_CHILDREN) | {j.id for j in JOBS.values() if j.state in ("queued", "running")}
    try:
        dirs = [d for d in config.OUTPUT_DIR.iterdir() if d.is_dir() and d.name not in active]
    except FileNotFoundError:
        return
    dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    for d in dirs[KEEP_JOBS:]:
        shutil.rmtree(d, ignore_errors=True)
        j = JOBS.get(d.name)
        if j is not None and j.state != "running":
            JOBS.pop(d.name, None)


def _lookup(jid: str) -> Job | None:
    """A job from memory, or from its ``job.json`` after a restart; None if unknown or pruned."""
    j = JOBS.get(jid)
    if j is not None or not _JOB_ID.match(jid):
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
                    raise RuntimeError(f"Tile index failed: {INDEX.error}")
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
                    raise _Stop(msg[1], msg[2], msg[3])
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
            job.log.append(w)
        job.result = result
        job.state = "done"
        job.set(1.0, "Done")
    except _Stop as s:
        job.error_kind = s.kind
        if s.kind == "cancelled" or job.cancel.is_set():
            job.state, job.error, job.error_kind = "cancelled", "Cancelled", "cancelled"
        else:
            job.state, job.error = "error", s.message
            job.advice, job.suggest = limits.advice_for(s.kind, job.opts)
            if s.trace:
                job.log.append(s.trace)
    except Exception as e:  # noqa: BLE001
        job.state, job.error, job.error_kind = "error", str(e), "error"
        job.log.append(traceback.format_exc(limit=3))
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


@app.get("/api/lidar/tiles")
def lidar_tiles(w: float, s: float, e: float, n: float):
    """NRCan CanElevation tiles in a lon/lat box, for jobs built in the browser (the index is 22 MB)."""
    import dataclasses
    if not (-180 <= w < e <= 180 and -90 <= s < n <= 90) or (e - w) * (n - s) > 25:
        raise HTTPException(400, "Box out of range or too large")
    if not INDEX.ready:
        raise HTTPException(503, "The LiDAR tile index is not available yet; try again in a minute.")
    tiles = INDEX.get().query(w, s, e, n)[:5001]
    return {"tiles": [dataclasses.asdict(t) for t in tiles]}


@app.api_route("/api/proxy", methods=["GET", "HEAD", "POST"])
async def data_proxy(request: Request, url: str = Query(..., max_length=4000)):
    """Stream a data file from an allowlisted host that has no CORS headers (maptomesh/proxy.py)."""
    from fastapi.responses import StreamingResponse
    from starlette.concurrency import run_in_threadpool

    from . import proxy
    body = await request.body() if request.method == "POST" else None
    if body is not None and len(body) > 64_000:
        raise HTTPException(413, "Request body too large")
    client = request.client.host if request.client else "unknown"
    try:
        status, headers, it = await run_in_threadpool(proxy.open_upstream, url, request.method,
                                                      dict(request.headers), body, client)
    except proxy.Refused as e:
        raise HTTPException(e.status, str(e)) from None
    except Exception as e:  # noqa: BLE001 - upstream unreachable
        raise HTTPException(502, f"Data source unreachable: {type(e).__name__}") from None
    if request.method == "HEAD":
        for _ in it:      # no body; running the iterator to its end releases the client's slot
            pass
        return Response(status_code=status, headers=headers)
    return StreamingResponse(it, status_code=status, headers=headers)


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


class AreaIn(BaseModel):
    lat: float
    lon: float
    width_m: float = Field(gt=0)
    height_m: float = Field(gt=0)
    grid_cells: int = 600
    us_source: str = "usgs"
    source: str = "lidar"
    ground_only: bool = False          # bare terrain not coloured by land cover
    dem_preference: str = "mapterhorn"


@app.post("/api/coverage")
def coverage(a: AreaIn) -> dict:
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
        "area_km2": round(a.width_m * a.height_m / 1e6, 2), "errors": errors,
        "projects": sorted(plist, key=lambda x: -x["year"]),
    })
    return out


@app.get("/api/projects/usgs")
def projects_usgs():
    """Footprints of every USGS 3DEP project in the AWS Entwine collection (simplified)."""
    from .usgs_ept import INDEX as EPT_INDEX

    try:
        return Response(EPT_INDEX.overlay(), media_type="application/geo+json",
                        headers={"Content-Encoding": "gzip"})
    except Exception as ex:  # noqa: BLE001
        raise HTTPException(502, f"USGS 3DEP project list unavailable: {ex}") from ex


@app.get("/api/coverage/usgs")
def coverage_usgs(w: float = Query(...), s: float = Query(...), e: float = Query(...), n: float = Query(...)):
    """USGS 3DEP tile footprints in a map view (only when zoomed in: one search per view)."""
    from .usgs3dep import footprints, in_us

    bbox = (w, s, e, n)
    if not in_us(bbox):
        return {"type": "FeatureCollection", "features": [], "in_us": False}
    if (e - w) * (n - s) > 0.35:
        return {"type": "FeatureCollection", "features": [], "in_us": True, "too_large": True}
    try:
        gj = footprints(bbox)
    except Exception as ex:  # noqa: BLE001
        raise HTTPException(502, f"USGS 3DEP search failed: {ex}") from ex
    gj["in_us"] = True
    return gj


def _refuse(kind: str, message: str, advice: list[str], suggest: dict | None = None, status: int = 422):
    """A job refused before it starts: same fields as a failed job, for the UI's failure dialog."""
    return JSONResponse({"detail": message, "error_kind": kind, "advice": advice, "suggest": suggest or {}},
                        status_code=status)


@app.post("/api/jobs")
def create_job(o: Options):
    from . import limits
    try:
        limits.check(o)
    except limits.LimitError as e:
        return _refuse(e.kind, e.message, e.advice, e.suggest)
    waiting = sum(1 for j in JOBS.values() if j.state in ("queued", "running"))
    if waiting >= limits.MAX_QUEUE:
        advice, _ = limits.advice_for("busy", o)
        return _refuse("busy", f"The server is busy: {waiting} models are queued or being built.", advice,
                       status=429)
    job = Job(uuid.uuid4().hex[:12], o)
    JOBS[job.id] = job
    while len(JOBS) > 50:
        JOBS.popitem(last=False)
    threading.Thread(target=_run_job, args=(job,), daemon=True).start()
    return {"id": job.id}


def _job(jid: str) -> Job:
    j = _lookup(jid)
    if not j:
        raise HTTPException(404, "Unknown job (finished models are kept for the last 10 jobs)")
    return j


def _done_file(jid: str, key: str):
    """Path of a finished job's file (``result[key]``); 409 while running, 404 / 410 if gone."""
    j = _job(jid)
    if j.state != "done":
        raise HTTPException(409, "Not ready")
    name = (j.result or {}).get(key)
    if not name:
        raise HTTPException(404, "No such file for this job")
    path = config.OUTPUT_DIR / jid / name
    if not path.exists():
        raise HTTPException(410, "This model's files were removed (only the last 10 jobs are kept); generate it again")
    return path, name


@app.get("/api/jobs/{jid}")
def job_status(jid: str) -> dict:
    return _job(jid).public()


@app.post("/api/jobs/{jid}/cancel")
def job_cancel(jid: str) -> dict:
    j = _job(jid)
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
    path = config.OUTPUT_DIR / jid / "thumbnail.png"
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
    if lod not in PREVIEW_LODS or not _JOB_ID.match(jid):
        raise HTTPException(404, "Unknown preview")
    path = config.OUTPUT_DIR / jid / f"preview_{lod}.bin.gz"
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
    if est.rate_limited(request.client.host if request.client else "unknown"):
        raise HTTPException(429, "Too many estimates from here; please try again later")
    if not est.SLOTS.acquire(timeout=20):
        raise HTTPException(503, "Busy; please try again in a moment")
    try:
        return est.estimate(o)
    finally:
        est.SLOTS.release()


# ---------------------------------------------------------------- device failures

class DeviceFailureIn(BaseModel):
    kind: str = Field("error", max_length=40)
    message: str = Field("", max_length=2000)
    trace: str = Field("", max_length=8000)
    stage: str = Field("", max_length=300)
    options: dict | None = None
    env: dict = Field(default_factory=dict)


@app.post("/api/device-failures")
def device_failure(f: DeviceFailureIn, request: Request) -> dict:
    """A build that failed in the user's browser (``maptomesh.device_failures``), for review."""
    from . import device_failures as df
    if df.rate_limited(request.client.host if request.client else "unknown"):
        raise HTTPException(429, "Too many reports from here")
    env = f.env if len(json.dumps(f.env, default=str)) <= 16_000 else {"dropped": "too large"}
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
    addr = request.client.host if request.client else "unknown"
    if shares.rate_limited(addr):
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


_JOB_ID = re.compile(r"^[0-9a-f]{12}$")
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_REVIEW_TIMES: dict[str, deque] = {}
_REVIEW_LOCK = threading.Lock()
REVIEWS_PER_HOUR = 20      # per client address
REVIEWS_PER_JOB = 10


def _rate_limited(addr: str) -> bool:
    now = time.time()
    with _REVIEW_LOCK:
        q = _REVIEW_TIMES.setdefault(addr, deque())
        while q and now - q[0] > 3600:
            q.popleft()
        if len(q) >= REVIEWS_PER_HOUR:
            return True
        q.append(now)
        if len(_REVIEW_TIMES) > 10_000:   # forget idle addresses
            for k in [k for k, v in _REVIEW_TIMES.items() if not v]:
                _REVIEW_TIMES.pop(k, None)
        return False


def _job_snapshot(job: Job) -> tuple[str, dict, dict]:
    result = {"result": job.result, "error": job.error, "log": job.log[-40:],
              "created": job.created, "message": job.message}
    return job.state, job.opts.model_dump(), result


def _client_snapshot(snap: dict) -> tuple[str, dict, dict]:
    if len(json.dumps(snap)) > 200_000:
        raise HTTPException(413, "Job snapshot too large")
    try:
        opts = Options(**(snap.get("options") or {})).model_dump()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"Invalid job settings in snapshot: {e}") from e
    state = snap.get("state") if snap.get("state") in ("done", "error", "cancelled") else "unknown"
    result = {"result": snap.get("result"), "error": snap.get("error"), "log": (snap.get("log") or [])[-40:]}
    return state, opts, result


@app.post("/api/jobs/{jid}/review")
def job_review(jid: str, rv: ReviewIn, request: Request) -> dict:
    """Store a review of a job with the job's settings and result (see ``maptomesh.reviews``)."""
    from .reviews import PROBLEM_KEYS, store

    if not _JOB_ID.match(jid):
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
    addr = request.client.host if request.client else "unknown"
    if _rate_limited(addr):
        raise HTTPException(429, "Too many reviews from here; please try again later")
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
    rid = st.add(job_id=jid, job_state=state, rating=rv.rating, problems=problems, comment=comment,
                 wants_followup=rv.followup, contact_name=name or None, contact_email=email or None,
                 options=options, result=result, snapshot_source=source, app_version=__version__,
                 user_agent=request.headers.get("user-agent", "")[:300])
    return {"ok": True, "id": rid}


@app.exception_handler(ValueError)
def _value_error(_req, exc: ValueError):
    return JSONResponse({"detail": str(exc)}, status_code=400)


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
    j = _lookup(job) if job and _JOB_ID.match(job) else None
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
                    f"{so['lat']:.4f}, {so['lon']:.4f}, made with MapToMesh. Opening the link builds it.")
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
