"""Overture Maps buildings (https://docs.overturemaps.org/guides/buildings/): footprints and heights
from the release's PMTiles archive, for areas where the elevation data is bare earth.

Overture conflates OpenStreetMap with other footprint datasets (Microsoft / Google / Esri ML
footprints) and merges heights into OSM footprints (Microsoft ML heights in the US and Canada,
USGS 3DEP lidar). Where OSM has no height tag, the OpenMapTiles ``render_height`` is a 5 m
default; Overture's height (or floors) fills those, and its non-OSM footprints add buildings OSM
doesn't have.

Properties are only kept at zoom 14 (lower zooms carry a sample of the footprints and no
attributes), so the tiles are read there. Releases are deleted after ~60 days: the latest one is
looked up in Overture's STAC catalogue. Tiles answer browsers directly (CORS) for a plain Range
request.
"""

from __future__ import annotations

import json
import logging
import math
import struct
import threading
import time

import numpy as np

from .. import config, mvt

log = logging.getLogger(__name__)

STAC = "https://stac.overturemaps.org/catalog.json"
TILES = "https://tiles.overturemaps.org/{release}/buildings.pmtiles"
ZOOM = 14
MAX_TILES = 100                  # ~15 x 15 km at mid latitudes (central Tokyo: ~0.9 MB per tile)
WINDOW = 16                      # tiles downloaded (in parallel) and held at once
LEVEL_M = 3.66                   # metres per floor (as OpenMapTiles' render_height)
MAX_HEIGHT_M = 1000.0            # anything taller is a data error (the tallest building: 828 m)
WORLD = 2 * math.pi * 6378137.0
LICENCE = "ODbL 1.0"
CREDIT = ("Buildings: © OpenStreetMap contributors, Overture Maps Foundation (ODbL); "
          "Microsoft Global ML Building Footprints (ODbL), Esri Community Maps contributors and "
          "Google Open Buildings (CC BY 4.0) where used")

_lock = threading.Lock()
_release: dict = {"t": 0.0, "name": ""}
_archives: dict = {}


def latest_release() -> str:
    """The newest release name (e.g. ``2026-11-18.0``), memoised for 6 hours and kept on disk so
    a brief outage of the catalogue doesn't lose the buildings (until that release is deleted)."""
    import requests
    with _lock:
        if _release["name"] and time.time() - _release["t"] < 6 * 3600:
            return _release["name"]
    path = config.CACHE_DIR / "overture" / "release.json"
    try:
        r = requests.get(STAC, headers={"User-Agent": config.USER_AGENT}, timeout=20)
        r.raise_for_status()
        name = str(r.json()["latest"])
        config.atomic_write(path, json.dumps({"latest": name}).encode())
    except Exception:  # noqa: BLE001 - fall back to the last known release
        if not path.exists():
            raise
        name = json.loads(path.read_text())["latest"]
    with _lock:
        _release.update(t=time.time(), name=name)
    return name


def _archive():
    from ..pmtiles import Archive
    rel = latest_release()
    with _lock:
        a = _archives.get(rel)
    if a is None:
        a = Archive(TILES.format(release=rel))
        with _lock:
            _archives.clear()            # older releases' directories: deleted upstream anyway
            _archives[rel] = a
    return a, rel


def _tiles(a, tiles: list[tuple[int, int]]):
    """(x, y, tile bytes) of the archive's zoom-14 tiles among ``tiles``, ``WINDOW`` downloaded at
    a time (up to 100 tiles of ~1-3 MB decompressed in a city: not all held at once)."""
    for k in range(0, len(tiles), WINDOW):
        got = a.get_many([(ZOOM, x, y) for x, y in tiles[k:k + WINDOW]])
        for key in list(got):            # in order; each dropped once handed over
            yield key[1], key[2], got.pop(key)


def needed_tiles(aoi, grid, need: np.ndarray) -> set[tuple[int, int]]:
    """Zoom-14 tiles under the ``need`` vertices."""
    from ..providers.mapterhorn import _needed_tiles
    return _needed_tiles(aoi, grid, need, ZOOM)


def _num(v) -> float:
    """A plausible positive number from a tile's properties (NaN otherwise: missing, a string, a
    flag, infinite, or beyond the tallest building)."""
    if isinstance(v, (int, float)) and not isinstance(v, bool) and 0 < v < MAX_HEIGHT_M:
        return float(v)
    return math.nan


def _height(p: dict) -> float:
    h = _num(p.get("height"))
    if h == h:
        return h
    f = _num(p.get("num_floors"))
    return f * LEVEL_M if f * LEVEL_M < MAX_HEIGHT_M else math.nan


def fetch(aoi, grid, need: np.ndarray, cancel=None, progress=lambda f, m: None) -> dict | None:
    """Footprints and heights on the vertex grid, for the zoom-14 tiles under ``need``:
    ``{"h": tallest known height (m, NaN: none), "mask": every footprint, "other": footprints
    from datasets other than OpenStreetMap, "tiles": n, "release": name}``; None when ``need`` is
    empty; ``TooLarge`` when the area needs more than ``MAX_TILES`` tiles."""
    from pyproj import Transformer

    from ..osm import _fill_box

    tiles = sorted(needed_tiles(aoi, grid, need))
    if not tiles:
        return None
    if len(tiles) > MAX_TILES:
        from . import TooLarge
        raise TooLarge(f"the area needs {len(tiles)} of its tiles, over {MAX_TILES}")
    from . import Cancelled
    a, rel = _archive()
    progress(0.1, f"Overture buildings: {len(tiles)} tiles")
    shape = grid.vshape
    h_out = np.full(shape, np.nan, np.float32)
    mask = np.zeros(shape, bool)
    other = np.zeros(shape, bool)
    to_aoi = Transformer.from_crs(3857, aoi.crs, always_xy=True)
    tile_m = WORLD / 2 ** ZOOM
    for i, (tx, ty, data) in enumerate(_tiles(a, tiles)):
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        try:
            layer = mvt.decode(data).get("building") if data else None
        except (ValueError, IndexError, KeyError, TypeError, struct.error) as e:   # a damaged tile: skip it
            log.warning("Overture tile %d/%d/%d unreadable: %r", ZOOM, tx, ty, e)
            continue
        del data
        if not layer:
            continue
        ext = float(layer["extent"])
        feats = []
        for f in layer["features"]:
            p = f["props"]
            if p.get("is_underground") is True or (isinstance(p.get("min_height"), (int, float))
                                                     and p["min_height"] > 0):
                continue                     # raised parts (canopies, skybridges) would wall off streets
            for poly in mvt.polygons(f["rings"]):
                feats.append((poly, _height(p), str(p.get("@geometry_source", "")) != "OpenStreetMap"))
        if not feats:
            continue
        # one transform per tile: all ring vertices at once
        lens = [len(r) for poly, _, _ in feats for r in poly]
        pts = np.asarray([pt for poly, _, _ in feats for r in poly for pt in r], dtype=np.float64)
        mx = (tx + pts[:, 0] / ext) * tile_m - WORLD / 2
        my = WORLD / 2 - (ty + pts[:, 1] / ext) * tile_m
        px, py = to_aoi.transform(mx, my)
        gc = np.column_stack([(np.asarray(px) - grid.x0) / grid.step, (np.asarray(py) - grid.y0) / grid.step])
        rings = np.split(gc, np.cumsum(lens)[:-1])
        k = 0
        for poly, h, oth in feats:
            parts = rings[k:k + len(poly)]
            k += len(poly)
            box = _fill_box(parts, shape)
            if box is None:
                continue
            sl, m = box
            mask[sl] |= m
            if oth:
                other[sl] |= m
            if h == h:
                view = h_out[sl]
                view[m] = np.fmax(view[m], h)
        progress(0.1 + 0.9 * (i + 1) / len(tiles), f"Overture buildings: tile {i + 1}/{len(tiles)}")
    return {"h": h_out, "mask": mask, "other": other, "tiles": len(tiles), "release": rel}
