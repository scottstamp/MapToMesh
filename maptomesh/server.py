"""FastAPI app: map UI + generation jobs."""

from __future__ import annotations

import re
import threading
import time
import traceback
import uuid
from collections import Counter, OrderedDict
from dataclasses import dataclass, field

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, Response
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
_RUN_LOCK = threading.Lock()  # one generation at a time (memory-heavy)


def _slug(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", s).strip("_")
    return s[:60] or "model"


def _run_job(job: Job) -> None:
    from .export import render_topdown, write_3mf
    from .model import build_parts, compute_surface, get_elevation
    from .osm import fetch_osm, rasterize_osm

    with _RUN_LOCK:
        if job.cancel.is_set():
            job.state = "cancelled"
            return
        job.state = "running"
        o = job.opts
        try:
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
            if want_water or want_roads or want_bld:
                job.set(0.66, "Fetching OpenStreetMap layers")
                try:
                    data = fetch_osm(aoi, g, o.osm_roads if want_roads else "none", want_water, want_bld,
                                     progress=stage(0.66, 0.72))
                    osm = rasterize_osm(data, aoi, g, mm_per_m, o.road_min_width_mm, o.road_width_scale,
                                        o.osm_roads, o.osm_streams if o.osm_water else "none",
                                        o.river_min_width_mm)
                    job.set(0.72, f"OpenStreetMap: {len(data['features']):,} features from {data['tiles']} tiles")
                except Exception as e:  # noqa: BLE001 - OSM is optional
                    job.log.append(f"Warning: OpenStreetMap unavailable ({e}); continuing without it")
            if job.cancel.is_set():
                raise RuntimeError("Cancelled")
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
        except Exception as e:  # noqa: BLE001
            if job.cancel.is_set():
                job.state, job.error = "cancelled", "Cancelled"
            else:
                job.state, job.error = "error", str(e)
                job.log.append(traceback.format_exc(limit=3))
        finally:
            # keep parts in memory only for recent jobs
            done = [j for j in JOBS.values() if j.parts is not None]
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


@app.get("/api/jobs/{jid}/preview.bin")
def job_preview(jid: str):
    from .export import preview_bytes
    j = _job(jid)
    if j.state != "done" or j.parts is None:
        raise HTTPException(409, "Preview not available")
    return Response(preview_bytes(j.parts), media_type="application/octet-stream")


@app.exception_handler(ValueError)
def _value_error(_req, exc: ValueError):
    return JSONResponse({"detail": str(exc)}, status_code=400)


@app.get("/")
def index():
    return FileResponse(config.WEB_DIR / "index.html")


app.mount("/static", StaticFiles(directory=config.WEB_DIR), name="static")
