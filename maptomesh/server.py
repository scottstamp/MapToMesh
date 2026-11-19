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
    result: dict | None = None
    parts: list | None = None
    base_parts: list | None = None
    created: float = field(default_factory=time.time)
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
                "message": self.message, "log": self.log[-30:], "error": self.error, "result": self.result,
                "options": self.opts.model_dump()}


JOBS: "OrderedDict[str, Job]" = OrderedDict()
# At most two jobs in flight: one builds (CPU and memory heavy, one at a time) while the next
# downloads its data (network-bound), so a queue moves up to twice as fast without holding more
# than two models' data in memory. MAPTOMESH_PIPELINE=1 runs jobs strictly one after another.
_PIPELINE = threading.BoundedSemaphore(max(1, int(os.environ.get("MAPTOMESH_PIPELINE", "2"))))
_BUILD_LOCK = threading.Lock()


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


def _run_job(job: Job) -> None:
    from .export import render_topdown, write_3mf
    from .model import build_parts, compute_surface, get_elevation
    from .osm import fetch_osm, rasterize_osm

    try:
        _wait_for(_PIPELINE, job, "Queued behind other models")
    except RuntimeError:
        job.state, job.error = "cancelled", "Cancelled"
        return
    try:
        if job.cancel.is_set():
            raise RuntimeError("Cancelled")
        job.state = "running"
        o = job.opts
        stage = lambda a, b: (lambda f, m: job.set(a + (b - a) * f, m))  # noqa: E731
        from .providers.north_america import Hrdem
        in_canada = Hrdem().covers(AOI(o.lat, o.lon, o.width_m, o.height_m).lonlat_bbox())
        if o.source == "lidar" and not INDEX.ready and in_canada:
            job.set(0.0, "Waiting for LiDAR tile index")
            INDEX.start()
            while not INDEX.ready:
                if INDEX.state == "error":
                    raise RuntimeError(f"Tile index failed: {INDEX.error}")
                if job.cancel.is_set():
                    raise RuntimeError("Cancelled")
                job.set(0.0, f"Preparing LiDAR index: {INDEX.message}")
                time.sleep(0.5)
        elev = get_elevation(o, stage(0.0, 0.65), job.cancel)
        aoi = AOI(o.lat, o.lon, o.width_m, o.height_m)
        g = elev.grid
        mm_per_m = (o.size_mm / max(g.nx, g.ny)) / g.step
        osm = None
        want_water = o.osm_water
        want_roads = o.osm_roads != "none" and (o.color_mode == "landcover" or o.road_emboss_mm != 0)
        want_bld = o.surface != "terrain" or o.color_mode == "landcover"
        want_bridges = o.surface != "terrain"   # bridge decks over water (OSM tags them)
        if want_water or want_roads or want_bld or want_bridges:
            job.set(0.66, "Fetching OpenStreetMap layers")
            try:
                data = fetch_osm(aoi, g, o.osm_roads if want_roads else ("bridges" if want_bridges else "none"),
                                 want_water, want_bld, progress=stage(0.66, 0.72))
                osm = rasterize_osm(data, aoi, g, mm_per_m, o.road_min_width_mm, o.road_width_scale,
                                    o.osm_roads if want_roads else "none",
                                    o.osm_streams if o.osm_water else "none",
                                    o.river_min_width_mm)
                job.set(0.72, f"OpenStreetMap: {len(data['features']):,} features from {data['tiles']} tiles")
            except Exception as e:  # noqa: BLE001 - OSM is optional
                job.log.append(f"Warning: OpenStreetMap unavailable ({e}); continuing without it")
        if job.cancel.is_set():
            raise RuntimeError("Cancelled")
        # the build: one at a time; the next job's download overlaps it
        _wait_for(_BUILD_LOCK, job, "Data ready; waiting for the model ahead to finish building")
        try:
            job.set(0.74, "Building surface")
            surf = compute_surface(elev, o, osm)
            res = build_parts(elev, surf, o, stage(0.75, 0.88))
            if job.cancel.is_set():
                raise RuntimeError("Cancelled")
            outdir = config.OUTPUT_DIR / job.id
            outdir.mkdir(parents=True, exist_ok=True)
            thumb = render_topdown(res.cell_rgb, res.z_mm, res.stats["cell_mm"], 1024)
            (outdir / "thumbnail.png").write_bytes(thumb)
            fname = _slug(o.name) + ".3mf"
            job.set(0.9, f"Writing {fname} ({res.stats['triangles']:,} triangles)")
            credits = " ".join(elev.info.get("attribution", []))
            write_3mf(outdir / fname, res.parts, res.filaments, o.name, o.layout,
                      description=(f"{o.lat:.5f}, {o.lon:.5f} - {o.width_m:.0f} x {o.height_m:.0f} m - "
                                   f"{res.stats['scale']}. Elevation data: {credits} "
                                   "Map data (c) OpenStreetMap contributors (ODbL)."
                                   + "".join(f" LICENCE RESTRICTION: {w}" for w in elev.info.get("licence_warnings", []))),
                      thumbnail=thumb)
            size = (outdir / fname).stat().st_size
            base_file = base_size = None
            if res.base_parts:
                base_file = _slug(o.name) + "_base.3mf"
                base_color = res.base_parts[0].color
                write_3mf(outdir / base_file, res.base_parts,
                          [{"slot": 1, "color": base_color, "used_by": ["Base plate"]}], o.name + " - base plate",
                          o.layout, description="Base plate for " + o.name)
                base_size = (outdir / base_file).stat().st_size
            job.parts = res.parts
            job.base_parts = res.base_parts
            job.result = {
                "file": fname, "file_size": size, "base_file": base_file, "base_file_size": base_size,
                "stats": res.stats, "filaments": res.filaments,
                "color_changes": res.color_changes, "notes": res.notes, "tips": res.tips or [],
                "data": {k: v for k, v in elev.info.items() if k in ("tiles", "points", "projects", "coverage",
                                                                       "items", "failed_tiles", "errors",
                                                                       "ept_nodes", "us_source", "sources",
                                                                       "attribution", "skipped",
                                                                       "licence_warnings")},
            }
            job.state = "done"
            job.set(1.0, "Done")
        finally:
            _BUILD_LOCK.release()
    except Exception as e:  # noqa: BLE001
        if job.cancel.is_set():
            job.state, job.error = "cancelled", "Cancelled"
        else:
            job.state, job.error = "error", str(e)
            job.log.append(traceback.format_exc(limit=3))
    finally:
        _PIPELINE.release()
        # keep parts in memory only for recent jobs
        done = [j for j in list(JOBS.values()) if j.parts is not None]
        for j in done[:-3]:
            j.parts = None
            j.base_parts = None


# ---------------------------------------------------------------- API

@app.on_event("startup")
def _startup() -> None:
    INDEX.start()
    # outputs are only reachable through in-memory job ids; drop stale ones
    import shutil
    cutoff = time.time() - 3 * 86400
    for d in config.OUTPUT_DIR.iterdir():
        if d.is_dir() and d.stat().st_mtime < cutoff:
            shutil.rmtree(d, ignore_errors=True)


@app.get("/api/status")
def status() -> dict:
    return {"version": __version__, "index": {"state": INDEX.state, "ready": INDEX.ready,
                                              "progress": INDEX.progress, "message": INDEX.message,
                                              "error": INDEX.error},
            "projects_ready": PROJECTS_GEOJSON.exists()}


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
    out["dem_sources"] = describe(bbox, area_km2)
    if not INDEX.ready and Hrdem().covers(bbox):  # Canadian point clouds need the tile index
        out.update({"ready": False, "message": INDEX.message})
        return out
    if a.source != "lidar":
        out.update({"ready": True, "tiles": 0, "ept_projects": 0, "max_tiles": config.MAX_TILES,
                    "area_km2": round(area_km2, 2), "errors": [], "projects": []})
        return out
    tiles, projects, errors = find_sources(bbox, INDEX.get() if INDEX.ready else None, a.us_source,
                                           config.MAX_TILES + 1)
    cnt = Counter((t.project, t.year) for t in tiles)
    plist = [{"name": p, "year": y, "tiles": n} for (p, y), n in cnt.items()]
    # AWS Entwine projects have no tiles: report the projects (their octree is read per job)
    plist += [{"name": p.label, "year": p.year, "tiles": None} for p in projects]
    out.update({
        "ready": True, "tiles": len(tiles), "ept_projects": len(projects), "max_tiles": config.MAX_TILES,
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


@app.post("/api/jobs")
def create_job(o: Options) -> dict:
    job = Job(uuid.uuid4().hex[:12], o)
    JOBS[job.id] = job
    while len(JOBS) > 30:
        JOBS.popitem(last=False)
    threading.Thread(target=_run_job, args=(job,), daemon=True).start()
    return {"id": job.id}


def _job(jid: str) -> Job:
    j = JOBS.get(jid)
    if not j:
        raise HTTPException(404, "Unknown job")
    return j


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
    j = _job(jid)
    if j.state != "done":
        raise HTTPException(409, "Not ready")
    path = config.OUTPUT_DIR / jid / j.result["file"]
    return FileResponse(path, media_type="model/3mf", filename=j.result["file"])


@app.get("/api/jobs/{jid}/base.3mf")
def job_base_3mf(jid: str):
    j = _job(jid)
    if j.state != "done" or not j.result.get("base_file"):
        raise HTTPException(404, "No separate base plate for this job")
    return FileResponse(config.OUTPUT_DIR / jid / j.result["base_file"], media_type="model/3mf",
                        filename=j.result["base_file"])


@app.get("/api/jobs/{jid}/thumbnail.png")
def job_thumb(jid: str):
    path = config.OUTPUT_DIR / jid / "thumbnail.png"
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path, media_type="image/png")


@app.get("/api/jobs/{jid}/stl.zip")
def job_stl(jid: str):
    from .export import write_stl_zip
    j = _job(jid)
    if j.state != "done":
        raise HTTPException(409, "Not ready")
    name = j.result["file"].rsplit(".", 1)[0] + "_stl.zip"
    path = config.OUTPUT_DIR / jid / name
    if not path.exists():
        if j.parts is None:
            raise HTTPException(410, "Model evicted from memory; regenerate to export STL")
        write_stl_zip(path, j.parts + (j.base_parts or []))
    return FileResponse(path, media_type="application/zip", filename=name)


_PREVIEW_LOCK = threading.Lock()


@app.get("/api/jobs/{jid}/preview.bin")
def job_preview(jid: str, lod: str = "high"):
    """The 3D-preview mesh (``export.preview_bytes``, gzip-encoded) at a level of detail. Built on
    the first request and kept with the job's files (so it outlives the job in memory); a job's
    preview never changes, so browsers may cache it for good."""
    from .export import PREVIEW_LODS, preview_bytes
    if lod not in PREVIEW_LODS or not _JOB_ID.match(jid):
        raise HTTPException(404, "Unknown preview")
    path = config.OUTPUT_DIR / jid / f"preview_{lod}.bin.gz"
    if not path.exists():
        j = _job(jid)
        if j.state != "done" or j.parts is None:
            raise HTTPException(409, "Preview not available")
        with _PREVIEW_LOCK:
            if not path.exists():
                tmp = path.with_suffix(".tmp")
                tmp.write_bytes(preview_bytes(j.parts, PREVIEW_LODS[lod]))
                tmp.replace(path)
    return FileResponse(path, media_type="application/octet-stream",
                        headers={"Content-Encoding": "gzip", "Cache-Control": "public, max-age=31536000, immutable"})


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
    job = JOBS.get(jid)
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
def index(request: Request, job: str | None = None):
    """The UI, with link-preview metadata (absolute URLs) filled in."""
    import html

    base = config.PUBLIC_URL or str(request.base_url).rstrip("/")
    title, desc, page = SITE_TITLE, SITE_DESCRIPTION, f"{base}/"
    path, w, h, mime, alt = EXAMPLE_IMAGE
    image = f"{base}{path}"
    j = JOBS.get(job) if job and _JOB_ID.match(job) else None
    if j is not None and j.state == "done" and j.result:
        o = j.opts
        title = f"{o.name} - MapToMesh"
        desc = (f"3D-printable model of {o.width_m / 1000:g} x {o.height_m / 1000:g} km around "
                f"{o.lat:.4f}, {o.lon:.4f} ({j.result['stats'].get('scale', '')}), made with MapToMesh.")
        page = f"{base}/?job={j.id}"
        image, w, h, mime, alt = f"{base}/api/jobs/{j.id}/thumbnail.png", None, None, "image/png", f"Top view of the {o.name} model"
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
