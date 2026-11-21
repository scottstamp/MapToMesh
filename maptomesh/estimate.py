"""How much data a job will download: shown before building on a phone (mobile data), so the
user can decide. Follows the job's own choices (``model._fetch_elevation``, ``worker._pipeline``):

* LiDAR point clouds: COPC tiles are measured exactly from their octree hierarchy (the nodes the
  job's query reads, at its level of detail); USGS Entwine from the points in the planned nodes;
* elevation models: the first provider in the job's order that has data there. Exact for those
  that list their cloud-optimised GeoTIFFs (``DemProvider.cog_urls``: the compressed tiles of the
  window, from the files' headers), else from how they read (``download_estimate``); the
  surface model too when buildings and trees are wanted;
* OpenStreetMap: the vector tiles of the job's zoom level.

The bytes-per-point / per-pixel / per-tile figures come from measured browser builds (README /
docs/plans/browser-wasm.md); expect about +-50% for elevation models and OpenStreetMap.
"""

from __future__ import annotations

import concurrent.futures as cf
import os

from .geo import AOI, Grid
from .limits import RateLimit, Slots
from .options import Options

EPT_BYTES_PER_POINT = 12.0       # LAZ in USGS Entwine nodes
COPC_OVERHEAD = 128 * 1024       # header + hierarchy reads per COPC file
OSM_BYTES_PER_TILE = 45_000      # OpenMapTiles vector tile, built-up areas
COPC_SAMPLE = 40                 # files inspected; more are extrapolated


def estimate(o: Options) -> dict:
    """``{"mb": total, "parts": [{"label", "mb", "exact"}], "notes": [...]}``."""
    from . import config, providers  # noqa: F401 - registers the elevation providers
    from .dem import candidates
    from .limits import lidar_caps
    from .lidar_index import INDEX
    from .model import _models_suffice
    from .osm import _tile_range
    from .pointcloud import GROUND_RESOLUTION_FACTOR, RESOLUTION_FACTOR, copc_download_bytes, find_sources

    aoi = AOI(o.lat, o.lon, o.width_m, o.height_m)
    grid = Grid.for_aoi(aoi, max(50, min(o.grid_cells, config.MAX_GRID_CELLS)))
    area = o.width_m * o.height_m
    parts: list[dict] = []
    notes: list[str] = []

    lidar_done = False
    if o.source == "lidar" and not _models_suffice(o, aoi, grid):
        max_tiles, max_ept_km2 = lidar_caps(o)
        bbox = aoi.lonlat_bbox(margin=grid.step * 2)
        tiles, projects, _ = find_sources(bbox, INDEX.get() if INDEX.ready else None, o.us_source, max_tiles + 1)
        ok = (tiles or projects) and len(tiles) <= max_tiles and not (projects and area > max_ept_km2 * 1e6)
        if ok:
            resolution = max(grid.step * (GROUND_RESOLUTION_FACTOR if o.ground_only else RESOLUTION_FACTOR), 0.25)
            total = 0.0
            exact = True
            if tiles:
                sample = tiles[:COPC_SAMPLE]
                with cf.ThreadPoolExecutor(max_workers=8) as ex:
                    sizes = list(ex.map(lambda t: _safe(copc_download_bytes, t, aoi, grid, resolution), sample))
                known = [s for s in sizes if s is not None]
                if known:
                    total += (sum(known) + COPC_OVERHEAD * len(known)) * len(tiles) / len(known)
                exact = len(known) == len(tiles)
            if projects:
                from . import usgs_ept
                for p in projects:
                    plan = _safe(usgs_ept.plan, p, aoi, grid, resolution)
                    if plan is not None:
                        total += plan.points * EPT_BYTES_PER_POINT
                exact = False
            parts.append({"label": "LiDAR point clouds", "mb": round(total / 1e6, 1), "exact": exact})
            notes.append("Elevation models fill any gaps between the point clouds.")
            lidar_done = True

    if not lidar_done:
        dsm = not o.ground_only
        cands = [p for p in candidates(aoi, grid, dsm=dsm, shared_first=o.dem_preference != "official")
                 if area <= p.max_area_km2 * 1e6]
        eff = lambda p: max(p.res_m, grid.step / 2)  # noqa: E731 - detail the job ranks by
        chosen = None
        for prov in cands:
            b, exact = _dem_bytes(prov, aoi, grid, dsm)
            if b is None:
                continue                  # no data here: the job moves on as well
            if chosen is None:
                chosen = (prov, b, exact)
                if exact:
                    break
                continue
            # the first choice is only known by its region (a box can reach over a border): a
            # provider of the same detail whose files cover the area may be the one with the data;
            # take the larger, so the warning doesn't fall short
            if eff(prov) > eff(chosen[0]):
                break
            if exact:
                if b > chosen[1]:
                    chosen = (prov, b, exact)
                break
        if chosen is not None:
            prov, b, exact = chosen
            layers = " (terrain and surface)" if dsm and prov.has_dsm else ""
            parts.append({"label": f"Elevation: {prov.label}{layers}", "mb": round(b / 1e6, 1), "exact": exact})
            if prov.key == "fabdem":      # land only: Copernicus fills the sea and gaps
                cop = next((p for p in cands if p.key == "copernicus_glo30"), None)
                cb = _dem_bytes(cop, aoi, grid, dsm)[0] if cop is not None else None
                if cb:
                    parts.append({"label": f"Elevation: {cop.label} (water and gaps, if any)",
                                  "mb": round(cb / 1e6, 1), "exact": True})

    want_roads = o.osm_roads != "none" and (o.color_mode == "landcover" or o.road_emboss_mm != 0)
    if o.osm_water or want_roads or o.surface != "terrain" or o.color_mode == "landcover":
        bbox = aoi.lonlat_bbox(margin=grid.step * 4)
        z = 14
        while z > 9:
            xs, ys = _tile_range(bbox, z)
            if len(xs) * len(ys) <= 64:
                break
            z -= 1
        xs, ys = _tile_range(bbox, z)
        parts.append({"label": "OpenStreetMap", "mb": round(len(xs) * len(ys) * OSM_BYTES_PER_TILE / 1e6, 1),
                      "exact": False})
    if o.show_buildings:
        # Japan's elevation is bare earth: buildings come from PLATEAU's 3D city models
        from .buildings import plateau
        km2 = o.width_m * o.height_m / 1e6
        if km2 <= plateau.MAX_KM2 and _safe(plateau.cities_for, aoi.lonlat_bbox()):
            parts.append({"label": "3D buildings: Project PLATEAU (Japan)",
                          "mb": round(min(km2 * plateau.MB_PER_KM2, plateau.MAX_MB), 1), "exact": False})
    return {"mb": round(sum(p["mb"] for p in parts), 1), "parts": parts, "notes": notes}


# An estimate makes the server read file headers: at most ``PER_HOUR`` per client
# (``limits.client_key``) and ``MAPTOMESH_ESTIMATES_CONCURRENT`` at once, one per client; a request
# that finds no free slot is refused at once (``limits.Slots``), not queued on a server thread.
PER_HOUR = int(os.environ.get("MAPTOMESH_ESTIMATES_PER_HOUR", "120"))
RATE = RateLimit(PER_HOUR)
SLOTS = Slots(int(os.environ.get("MAPTOMESH_ESTIMATES_CONCURRENT", "4")), per_client=1)


def rate_limited(addr: str) -> bool:
    """Count an estimate for ``addr``; True if it is over ``PER_HOUR``."""
    RATE.per_hour = PER_HOUR
    return RATE.hit(addr) > 0


def _dem_bytes(prov, aoi, grid, dsm) -> tuple[float | None, bool]:
    """(bytes, exact) ``prov`` would download; (None, True) if its files miss the area."""
    from .dem_readers import cog_download_bytes
    urls = _safe(prov.cog_urls, aoi, grid, dsm)
    if urls is None:
        return (_safe(prov.download_estimate, aoi, grid, dsm) or 0.0), False
    if not urls:
        return None, True
    sample = urls[:200]
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        b = sum(x or 0 for x in ex.map(lambda u: _safe(cog_download_bytes, u, aoi, grid), sample))
    b *= len(urls) / len(sample)
    return (b if b > len(sample) * (1 << 16) else None), True


def _safe(fn, *args):
    try:
        return fn(*args)
    except Exception:  # noqa: BLE001 - an unreadable file / project counts as unknown
        return None
