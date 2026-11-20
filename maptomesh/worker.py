"""One generation job in a child process.

The server (``server._run_job``) starts one process per job and watches it: a job that runs past
the time limit or grows past the memory budget is killed, without taking the server (or other
jobs) down. The child does everything heavy - data, surface, meshing, the 3MF, previews and the
STL zip - and leaves the results as files in ``OUTPUT_DIR/<job id>``; nothing large crosses back.

Messages, child -> parent over a ``multiprocessing`` pipe:

    ("progress", fraction, message)
    ("ready_to_build",)          data is in; wait for ("build",) before the CPU / memory heavy part
    ("done", result)             result: the job's result dict (JSON-able), files are written
    ("error", kind, message, trace)

parent -> child: ("build",). Cancelling a job kills the child.
"""

from __future__ import annotations

import re
import sys
import threading
import time
import traceback


def _slug(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", s).strip("_")
    return s[:60] or "model"


def _oom_first() -> None:
    """Linux: if memory runs out, the kernel kills this process rather than the server."""
    try:
        with open("/proc/self/oom_score_adj", "w") as f:
            f.write("1000")
    except OSError:
        pass


def run(job_id: str, opts: dict, conn) -> None:
    """Child process entry point."""
    _oom_first()
    last = [0.0, ""]

    def progress(f: float, msg: str) -> None:
        now = time.time()
        if msg != last[1] or now - last[0] > 0.25:   # don't flood the pipe with per-tile updates
            last[0], last[1] = now, msg
            conn.send(("progress", float(f), str(msg)))

    try:
        result = _pipeline(job_id, opts, conn, progress)
        conn.send(("done", result))
    except MemoryError:
        conn.send(("error", "memory", "Out of memory", traceback.format_exc(limit=3)))
    except Exception as e:  # noqa: BLE001
        conn.send(("error", "error", str(e) or type(e).__name__, traceback.format_exc(limit=3)))
    finally:
        try:
            conn.close()
        except OSError:
            pass


def _pipeline(job_id: str, opts_d: dict, conn, progress) -> dict:
    from . import config
    from .export import PREVIEW_LODS, preview_bytes, render_topdown, write_3mf, write_stl_zip
    from .geo import AOI
    from .lidar_index import INDEX
    from .model import build_parts, compute_surface, get_elevation
    from .options import Options
    from .osm import fetch_osm, rasterize_osm

    o = Options(**opts_d)
    cancel = threading.Event()        # the parent cancels by killing the process
    stage = lambda a, b: (lambda f, m: progress(a + (b - a) * f, m))  # noqa: E731
    if o.source == "lidar" and not INDEX.load_now():   # built by the server before the job started
        from .providers.north_america import Hrdem
        if Hrdem().covers(AOI(o.lat, o.lon, o.width_m, o.height_m).lonlat_bbox()):
            # without it Canadian point clouds would silently be skipped (and cached that way)
            raise RuntimeError("The LiDAR tile index is not available yet; try again in a minute.")
    elev = get_elevation(o, stage(0.0, 0.65), cancel)
    aoi = AOI(o.lat, o.lon, o.width_m, o.height_m)
    g = elev.grid
    mm_per_m = (o.size_mm / max(g.nx, g.ny)) / g.step
    osm = None
    warnings = []
    want_water = o.osm_water
    want_roads = o.osm_roads != "none" and (o.color_mode == "landcover" or o.road_emboss_mm != 0)
    want_bld = o.show_buildings or o.color_mode == "landcover"
    want_bridges = o.show_bridges   # bridge decks over water (OSM tags them)
    if want_water or want_roads or want_bld or want_bridges:
        progress(0.66, "Fetching OpenStreetMap layers")
        try:
            data = fetch_osm(aoi, g, o.osm_roads if want_roads else ("bridges" if want_bridges else "none"),
                             want_water, want_bld, progress=stage(0.66, 0.72))
            osm = rasterize_osm(data, aoi, g, mm_per_m, o.road_min_width_mm, o.road_width_scale,
                                o.osm_roads if want_roads else "none",
                                o.osm_streams if o.osm_water else "none",
                                o.river_min_width_mm)
            progress(0.72, f"OpenStreetMap: {len(data['features']):,} features from {data['tiles']} tiles")
        except Exception as e:  # noqa: BLE001 - OSM is optional
            warnings.append(f"Warning: OpenStreetMap unavailable ({e}); continuing without it")
            progress(0.72, warnings[-1])
    # 3D buildings where the elevation is bare earth (PLATEAU in Japan, Overture heights elsewhere)
    from .buildings import fetch_buildings
    extra, bld_credits, bld_warnings = fetch_buildings(o, elev, aoi, g, stage(0.72, 0.735), cancel)
    if extra:
        osm = {**(osm or {}), **extra}
        del extra
    warnings += bld_warnings
    attribution = list(elev.info.get("attribution", [])) + bld_credits

    # the build: one at a time across jobs; the parent says when
    conn.send(("ready_to_build",))
    msg = conn.recv()
    if msg[0] != "build":
        raise RuntimeError("Cancelled")
    progress(0.74, "Building surface")
    surf = compute_surface(elev, o, osm)
    res = build_parts(elev, surf, o, stage(0.75, 0.86))
    del surf, osm
    outdir = config.OUTPUT_DIR / job_id
    outdir.mkdir(parents=True, exist_ok=True)
    thumb = render_topdown(res.cell_rgb, res.z_mm, res.stats["cell_mm"], 1024)
    (outdir / "thumbnail.png").write_bytes(thumb)
    fname = _slug(o.name) + ".3mf"
    progress(0.87, f"Writing {fname} ({res.stats['triangles']:,} triangles)")
    credits = " ".join(attribution)
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
        write_3mf(outdir / base_file, res.base_parts,
                  [{"slot": 1, "color": res.base_parts[0].color, "used_by": ["Base plate"]}],
                  o.name + " - base plate", o.layout, description="Base plate for " + o.name)
        base_size = (outdir / base_file).stat().st_size
    # everything the server may serve later, so the meshes needn't stay in memory anywhere
    progress(0.92, "Preparing the 3D preview")
    for lod, budget in PREVIEW_LODS.items():
        tmp = outdir / f"preview_{lod}.bin.tmp"
        tmp.write_bytes(preview_bytes(res.parts, budget))
        tmp.replace(outdir / f"preview_{lod}.bin.gz")
    progress(0.96, "Writing the STL zip")
    stl = fname.rsplit(".", 1)[0] + "_stl.zip"
    write_stl_zip(outdir / stl, res.parts + (res.base_parts or []))
    return {
        "file": fname, "file_size": size, "base_file": base_file, "base_file_size": base_size,
        "stl_file": stl,
        "stats": res.stats, "filaments": res.filaments,
        "color_changes": res.color_changes, "notes": res.notes, "tips": res.tips or [],
        "data": {k: v for k, v in elev.info.items() if k in ("tiles", "points", "projects", "coverage",
                                                               "items", "failed_tiles", "errors",
                                                               "ept_nodes", "us_source", "sources",
                                                               "attribution", "skipped",
                                                               "licence_warnings")}
                 | {"attribution": attribution},
        "warnings": warnings,
    }


if __name__ == "__main__":   # pragma: no cover - started through multiprocessing
    sys.exit("run through maptomesh.server")
