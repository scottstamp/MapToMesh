"""Mapterhorn terrain tiles (https://mapterhorn.com): ~150 open national and regional terrain
models (DTMs, 0.25-20 m) merged into one global Web Mercator tile pyramid.

Data: PMTiles archives on download.mapterhorn.com (Cloudflare R2), read with HTTP range requests:
``planet.pmtiles`` holds zooms 0-12, ``6-{x}-{y}.pmtiles`` the zooms 13-17 under each zoom-6 tile.
Tiles are 512 px lossless WebP with Terrarium-encoded heights (h = R*256 + G + B/256 - 32768),
rounded to 1/256 m at zoom 19 and twice that per zoom level below (6 cm at z15, 1.6 cm at z17).
Heights stay in each source's national vertical datum; where two sources meet, Mapterhorn blends
them over ~150 m (Web Mercator metres). Areas without a regional model are filled with Copernicus
GLO-30, a surface model.

Coverage: a vector tile layer with each source's footprint. The provider offers one candidate per
resolution tier of the sources covering the area (``variants``), each masked to its sources'
footprints minus the blend band, so the global Copernicus filler and coarse tiers never pre-empt
a finer official model. Copernicus itself comes from our own provider (``global_``).

Licences: code BSD-3; the data keeps each source's licence (CC BY, OGL, Etalab, DL-DE, CC0, public
domain...), credited per source with every model. Japan's DEM is GSI's raw 基盤地図情報, used by
Mapterhorn under its own Survey Act approval: a warning goes with models that use it.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests

from .. import config
from ..dem import DemArrays, DemProvider, register, shared_first
from ..dem_readers import _dst_transform, _lonlat_to_tile, merge_fill, reproject_array

BASE = os.environ.get("MAPTOMESH_MAPTERHORN_URL", "https://download.mapterhorn.com/").rstrip("/") + "/"
COVERAGE_URL = "https://single-archive-tiles.mapterhorn.com/coverage/{z}/{x}/{y}.mvt"
CACHE = config.CACHE_DIR / "mapterhorn"
MANIFEST_TTL = 6 * 3600          # download_urls.json: which archives exist, their md5 (= cache version)
ATTRIBUTION_TTL = 86400
TILE_PX = 512
BLEND_M = 150.0                  # Web Mercator metres blended on each side of a seam between sources
# Mapterhorn's seam blend is a Gaussian (sigma ~ BLEND_M / 4) faded in near the seam: measured
# against the agencies' models, its error is gone ~2 sigma from the edge of a footprint (coasts at
# St. John's and Dover). Cells closer than that form an "edge band" candidate, ranked like
# EDGE_RES data: after any real model of the area, before Copernicus.
EDGE_FRACTION = 0.5
EDGE_RES = 25.0
SEA_M = 2.0                      # |filler height| below this far outside the footprints: sea
MAX_MOSAIC_PX = 64_000_000       # per zoom level: ~256 MB of float32
FILLER = {"glo30"}               # global filler (Copernicus, a surface model): our own provider has it
WORLD = 2 * math.pi * 6378137.0

ATTRIBUTION_BASE = "Terrain tiles: Mapterhorn (https://mapterhorn.com)"
LICENCE = ("Mapterhorn code BSD-3; the data keeps each source's open licence, credited per source "
           "(https://mapterhorn.com/attribution)")
JAPAN_NOTE = ("Japan (GSI 基盤地図情報 DEM via Mapterhorn): GSI's content terms apply, credit "
              "「出典：国土地理院」. Publishing or selling maps made from GSI survey results may need your own "
              "approval under Japan's Survey Act (測量法); the approval on Mapterhorn's copy covers only "
              "Mapterhorn. Check https://www.gsi.go.jp/LAW/2930-index.html before selling prints.")

_lock = threading.Lock()


# ---------------------------------------------------------------- manifests

def _cached_get(url: str, path, ttl: float) -> bytes:
    """GET with an on-disk copy refreshed after ``ttl`` seconds (a stale copy is used offline)."""
    if path.exists() and time.time() - path.stat().st_mtime < ttl:
        return path.read_bytes()
    try:
        r = requests.get(url, headers={"User-Agent": config.USER_AGENT}, timeout=(15, 60))
        r.raise_for_status()
        body = r.content
    except requests.RequestException:
        if path.exists():
            return path.read_bytes()
        raise
    path.parent.mkdir(parents=True, exist_ok=True)
    config.atomic_write(path, body)
    return body


_MANIFEST: dict = {"t": 0.0, "items": {}, "version": ""}
_SOURCES: dict = {"t": 0.0, "by_key": {}}


def manifest() -> tuple[dict, str]:
    """({archive name: item}, release version) from download_urls.json."""
    with _lock:
        if time.time() - _MANIFEST["t"] > 600 or not _MANIFEST["items"]:
            data = json.loads(_cached_get(BASE + "download_urls.json", CACHE / "download_urls.json", MANIFEST_TTL))
            _MANIFEST.update(t=time.time(), items={i["name"]: i for i in data["items"]},
                             version=str(data.get("version", "")))
        return _MANIFEST["items"], _MANIFEST["version"]


def sources() -> dict[str, dict]:
    """{source key: attribution.json entry} (name, producer, licence, resolution, website)."""
    with _lock:
        if time.time() - _SOURCES["t"] > 600 or not _SOURCES["by_key"]:
            data = json.loads(_cached_get(BASE + "attribution.json", CACHE / "attribution.json", ATTRIBUTION_TTL))
            _SOURCES.update(t=time.time(), by_key={d["source"]: d for d in data})
        return _SOURCES["by_key"]


# ---------------------------------------------------------------- coverage

def _coverage_zoom(w: float, s: float, e: float, n: float) -> int:
    z = 12
    while z > 7:
        x0, y0 = _lonlat_to_tile(w, n, z)
        x1, y1 = _lonlat_to_tile(e, s, z)
        if (int(x1) - int(x0) + 1) * (int(y1) - int(y0) + 1) <= 16:
            break
        z -= 1
    return z


_COVER_MEMO: dict = {}


def coverage_masks(aoi, grid) -> dict[str, np.ndarray]:
    """{source key: vertex mask of its footprint} for the sources covering the AOI (filler excluded)."""
    _, version = manifest()
    key = (aoi.key(), grid.key(), version)
    with _lock:
        hit = _COVER_MEMO.get(key)
    if hit is None:
        hit = _coverage_masks(aoi, grid, version)
        with _lock:
            _COVER_MEMO[key] = hit
            while len(_COVER_MEMO) > 4:
                _COVER_MEMO.pop(next(iter(_COVER_MEMO)))
    return hit


def _coverage_masks(aoi, grid, version: str) -> dict[str, np.ndarray]:
    from pyproj import Transformer

    from .. import mvt

    w, s, e, n = aoi.lonlat_bbox(margin=grid.step * 3)
    z = _coverage_zoom(w, s, e, n)
    x0, y0 = (int(v) for v in _lonlat_to_tile(w, n, z))
    x1, y1 = (int(v) for v in _lonlat_to_tile(e, s, z))
    tile_m = WORLD / 2 ** z
    to_aoi = Transformer.from_crs(3857, aoi.crs, always_xy=True)
    shapes: dict[str, list] = {}
    for ty in range(y0, y1 + 1):
        for tx in range(x0, x1 + 1):
            url = COVERAGE_URL.format(z=z, x=tx, y=ty)
            key = hashlib.sha1(f"{url}|{version}".encode()).hexdigest()[:20]
            try:
                data = _cached_get(url, CACHE / "coverage" / f"{key}.mvt", 30 * 86400)
            except requests.HTTPError as ex:
                if ex.response is not None and ex.response.status_code == 404:
                    continue
                raise
            layer = mvt.decode(data).get("coverage")
            if not layer:
                continue
            ext = float(layer["extent"])
            for f in layer["features"]:
                src = str(f["props"].get("source", ""))
                if not src or src in FILLER or src.startswith("debug"):
                    continue
                for poly in mvt.polygons(f["rings"]):
                    rings = []
                    for r in poly:
                        a = np.asarray(r, dtype=np.float64)
                        mx = (tx + a[:, 0] / ext) * tile_m - WORLD / 2
                        my = WORLD / 2 - (ty + a[:, 1] / ext) * tile_m
                        px, py = to_aoi.transform(mx, my)
                        rings.append(list(zip(px, py)) + [(px[0], py[0])])
                    shapes.setdefault(src, []).append({"type": "Polygon", "coordinates": rings})
    H, W = grid.vshape
    out = {}
    for src, geoms in shapes.items():
        m = _rasterize(geoms, grid)
        if m.any():
            out[src] = m
    return out


def _rasterize(geoms: list[dict], grid) -> np.ndarray:
    """Vertex mask of polygons (GeoJSON-like, in the AOI's CRS)."""
    H, W = grid.vshape
    try:
        from rasterio import features
    except ImportError:   # in the browser (Pyodide): no GDAL; even-odd fill on vertex positions
        from ..raster import fill_polygon
        m = np.zeros((H, W), bool)
        for g in geoms:
            rings = [np.column_stack([(np.asarray([p[0] for p in r]) - grid.x0) / grid.step,
                                      (np.asarray([p[1] for p in r]) - grid.y0) / grid.step])
                     for r in g["coordinates"]]
            m |= fill_polygon(rings, (H, W))
        return m
    return features.rasterize(geoms, out_shape=(H, W), transform=_dst_transform(grid), fill=0,
                              default_value=1, dtype="uint8")[::-1].astype(bool)


def _erode(mask: np.ndarray, metres: float, step: float) -> np.ndarray:
    """Cells of ``mask`` at least ``metres`` from its outside (the grid edge doesn't count)."""
    from scipy import ndimage
    if not mask.any() or mask.all():
        return mask
    pad = np.pad(mask, 1, constant_values=True)   # grid edges are not seams
    d = ndimage.distance_transform_edt(pad)[1:-1, 1:-1] * step
    return mask & (d > metres)


# ---------------------------------------------------------------- tiles

def _archive(name: str):
    from ..pmtiles import Archive
    items, _ = manifest()
    it = items.get(name)
    if it is None or it.get("size", 0) < 1024:     # not published / empty archive
        return None, None
    key = (name, it.get("md5sum", ""))
    with _lock:
        a = _ARCHIVES.get(key)
    if a is None:
        a = Archive(BASE + name, it.get("md5sum", ""))
        with _lock:
            _ARCHIVES[key] = a
    return a, it


_ARCHIVES: dict = {}


def _invalidate() -> None:
    """An archive changed upstream (new release): reload the manifest (new md5 = new cache key)."""
    with _lock:
        _MANIFEST.update(t=0.0, items={}, version="")
        _ARCHIVES.clear()
        _COVER_MEMO.clear()
    (CACHE / "download_urls.json").unlink(missing_ok=True)


def _decode(data: bytes) -> np.ndarray:
    import warnings

    try:
        from rasterio.errors import NotGeoreferencedWarning
        from rasterio.io import MemoryFile
    except ImportError:   # in the browser (Pyodide): Pillow decodes lossless WebP
        import io

        from PIL import Image
        p = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"), dtype=np.float32).transpose(2, 0, 1)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NotGeoreferencedWarning)
            with MemoryFile(data) as mf, mf.open() as src:
                p = src.read().astype(np.float32)
    return p[0] * 256.0 + p[1] + p[2] / 256.0 - 32768.0


def _tiles(w, s, e, n, z):
    x0, y0 = (int(math.floor(v)) for v in _lonlat_to_tile(w, n, z))
    x1, y1 = (int(math.floor(v)) for v in _lonlat_to_tile(e, s, z))
    lim = 2 ** z - 1
    return range(max(0, x0), min(lim, x1) + 1), range(max(0, y0), min(lim, y1) + 1)


def _zoom_px(z: int, lat: float) -> float:
    """Ground size (m) of a pixel at zoom ``z`` and latitude ``lat``."""
    return WORLD * math.cos(math.radians(lat)) / (2 ** z * TILE_PX)


def _needed_tiles(aoi, grid, need: np.ndarray, z: int) -> set[tuple[int, int]]:
    """Tiles at zoom ``z`` under the ``need`` vertices: ``need`` is pooled into blocks of at most
    a quarter tile (plus one block of margin for the resampling), whose corners and centres are
    looked up - a block never spans more than two tiles per direction, so no tile is missed."""
    from scipy import ndimage
    _, s0, _, n0 = aoi.lonlat_bbox()
    lat = max(abs(s0), abs(n0))       # smallest tiles on the ground: size the blocks for those
    k = max(1, int(_zoom_px(z, lat) * TILE_PX / 4 / grid.step))
    H, W = need.shape
    nr, nc = -(-H // k), -(-W // k)
    pad = np.zeros((nr * k, nc * k), bool)
    pad[:H, :W] = need
    blk = ndimage.binary_dilation(pad.reshape(nr, k, nc, k).any(axis=(1, 3)), structure=np.ones((3, 3), bool))
    r, c = np.nonzero(blk)
    if r.size == 0:
        return set()
    rr = np.concatenate([r, r, r + 1, r + 1, r + 0.5]) * k
    cc = np.concatenate([c, c + 1, c, c + 1, c + 0.5]) * k
    x = grid.x0 + cc * grid.step
    y = grid.y0 + rr * grid.step
    lon, la = aoi.to_lonlat.transform(x, y)
    n = 2 ** z
    tx = np.floor((np.asarray(lon) + 180.0) / 360.0 * n).astype(np.int64)
    lr = np.radians(np.clip(np.asarray(la), -85.0511, 85.0511))
    ty = np.floor((1.0 - np.arcsinh(np.tan(lr)) / np.pi) / 2.0 * n).astype(np.int64)
    ok = (tx >= 0) & (tx < n) & (ty >= 0) & (ty < n)
    return set(zip(tx[ok].tolist(), ty[ok].tolist()))


def _mosaic_tiles(aoi, grid, target_m: float, need: np.ndarray) -> tuple[int, set]:
    """The zoom ``read_mosaic`` starts at (the coarsest with pixels at most ``target_m``, coarser
    if the mosaic would be too large) and the tiles it needs there."""
    w, s, e, n = aoi.lonlat_bbox(margin=grid.step * 3)
    lat = (s + n) / 2
    z = 17
    while z > 0 and _zoom_px(z - 1, lat) <= target_m:
        z -= 1
    while True:   # cap the mosaic size
        wanted = _needed_tiles(aoi, grid, need, z)
        if not wanted:
            return z, set()
        xs, ys = [x for x, _ in wanted], [y for _, y in wanted]
        if z == 0 or (max(xs) - min(xs) + 1) * (max(ys) - min(ys) + 1) * TILE_PX ** 2 <= MAX_MOSAIC_PX:
            return z, wanted
        z -= 1


def read_mosaic(aoi, grid, target_m: float, need: np.ndarray | None = None, cancel=None,
                progress=lambda f, m: None) -> tuple[np.ndarray, str]:
    """Mapterhorn heights on the vertex grid (where ``need``, default everywhere), read at the
    coarsest zoom whose pixels are at most ``target_m``; tiles missing there (the archive's local
    max zoom is lower) come from lower zooms."""
    need = np.ones(grid.vshape, bool) if need is None else need
    z, wanted = _mosaic_tiles(aoi, grid, target_m, need)
    if not wanted:
        return np.full(grid.vshape, np.nan), ""
    out = np.full(grid.vshape, np.nan)
    zooms, ntiles = [], 0
    while wanted and z >= 0:
        if cancel is not None and cancel.is_set():
            raise RuntimeError("Cancelled")
        by_archive: dict[str, list] = {}
        for x, y in wanted:
            name = "planet.pmtiles" if z <= 12 else f"6-{x >> (z - 6)}-{y >> (z - 6)}.pmtiles"
            by_archive.setdefault(name, []).append((z, x, y))
        got: dict = {}
        for name, tl in by_archive.items():
            a, it = _archive(name)
            if a is None or z > it.get("max_zoom", 99) or z < it.get("min_zoom", 0):
                continue
            progress(0.1, f"Mapterhorn: {len(tl)} tiles at zoom {z}")
            got.update(a.get_many(tl))
        if got:
            xs = [x for _, x, _ in got]
            ys = [y for _, _, y in got]
            gx0, gy0 = min(xs), min(ys)
            mosaic = np.full(((max(ys) - gy0 + 1) * TILE_PX, (max(xs) - gx0 + 1) * TILE_PX), np.nan, np.float32)
            keys = list(got)
            with ThreadPoolExecutor(max_workers=min(8, len(keys))) as ex:
                for (_, x, y), arr in zip(keys, ex.map(lambda k: _decode(got[k]), keys)):
                    r, c = (y - gy0) * TILE_PX, (x - gx0) * TILE_PX
                    mosaic[r:r + TILE_PX, c:c + TILE_PX] = arr
            from affine import Affine
            px = WORLD / (2 ** z * TILE_PX)
            tf = Affine(px, 0, -WORLD / 2 + gx0 * TILE_PX * px, 0, -px, WORLD / 2 - gy0 * TILE_PX * px)
            merge_fill(out, reproject_array(mosaic, tf, 3857, aoi, grid))
            del mosaic
            zooms.append(f"z{z}x{len(got)}")
            ntiles += len(got)
        missing = wanted - {(x, y) for _, x, y in got}
        wanted = {(x >> 1, y >> 1) for x, y in missing}
        z -= 1
        if np.isfinite(out[need]).all():
            break
    return out, " ".join(zooms)


# ---------------------------------------------------------------- provider

def _edge_m(aoi) -> float:
    _, s0, _, n0 = aoi.lonlat_bbox()
    return BLEND_M * EDGE_FRACTION * math.cos(math.radians((s0 + n0) / 2))


def models_suffice(aoi, grid, max_res: float | None = None) -> bool:
    """True when Mapterhorn's models of ``max_res`` (default: the grid step) or finer cover all
    the land of the area, so point clouds would add nothing visible: everything outside their
    footprints (beyond the coastal blend) must be sea, i.e. Mapterhorn's global filler there is
    at ~0 m. Sources under a licence warning don't count. False on any error."""
    from scipy import ndimage
    max_res = grid.step if max_res is None else max_res
    try:
        masks = coverage_masks(aoi, grid)
        meta = sources()
        fine = np.zeros(grid.vshape, bool)
        for src, m in masks.items():
            if float(meta.get(src, {}).get("resolution", 99.0)) <= max_res and not src.startswith("jp"):
                fine |= m
        if not fine.any():
            return False
        far = ~fine
        if far.mean() <= 0.005:
            return True
        # beyond the coastal blend (Mapterhorn's filler is pulled towards the land there)
        d = ndimage.distance_transform_edt(far) * grid.step
        far &= d > 2 * BLEND_M
        if not far.any():
            return True
        h, _ = read_mosaic(aoi, grid, max(grid.step * 2, 20.0), far)
        hv = h[far]
        return bool(np.isfinite(hv).all() and np.mean(np.abs(hv) < SEA_M) >= 0.99)
    except Exception:  # noqa: BLE001 - offline, index changed...: use the point clouds
        return False


def _tier_res(res: float, step: float) -> float:
    """Sources whose detail a grid can't tell apart (finer than half a cell) share one tier."""
    return max(res, step / 2)


class MapterhornTier(DemProvider):
    """Mapterhorn data from the sources of one resolution tier, inside their footprints (or the
    footprints' edge band, ``band``)."""
    has_dsm = False
    prefer = 0
    max_area_km2 = 50_000.0

    def __init__(self, srcs: list[str], res: float, finest: float, mask: np.ndarray, meta: dict,
                 band: bool = False):
        self.srcs, self.res_m, self.finest, self.mask, self.band = srcs, res, finest, mask, band
        names = []
        for k in srcs:
            nm = meta.get(k, {}).get("name", k)
            if nm not in names:
                names.append(nm)
        self.key = "mapterhorn"
        self.label = (f"Mapterhorn {finest:g} m, coast / seam band: " if band else f"Mapterhorn {res:g} m: ") \
            + "; ".join(names)[:120]
        credits = [f"{meta[k].get('producer', k)}: {meta[k].get('name', k)} ({meta[k].get('license', '')})"
                   for k in srcs if k in meta]
        self.attribution = ATTRIBUTION_BASE + ", from " + "; ".join(dict.fromkeys(credits))
        self.licence = LICENCE
        self.licence_note = JAPAN_NOTE if any(k.startswith("jp") for k in srcs) else ""
        # data under a licence warning, or a job asking for the agencies' services first: the
        # agency's own provider (if any) goes first, Mapterhorn fills what it doesn't cover
        self.prefer = 2 if (self.licence_note or not shared_first()) else 0

    def fetch(self, aoi, grid, progress, cancel):
        need = self.mask
        if grid.read_blocks:   # gap fill: only the blocks still missing
            blk = np.zeros(grid.vshape, bool)
            for r0, r1, c0, c1 in grid.read_blocks:
                blk[r0:r1, c0:c1] = True
            need = need & blk
        target = max(self.finest, grid.step / 2)
        from ..httpcache import ChangedUpstream
        try:
            dtm, detail = read_mosaic(aoi, grid, target, need, cancel, progress)
        except ChangedUpstream:
            _invalidate()   # this job falls back to the next source; the next one reads the new release
            raise
        dtm = np.where(need, dtm, np.nan)
        return DemArrays(dtm, None, f"{', '.join(self.srcs)} ({detail})")

    def download_estimate(self, aoi, grid, dsm):
        need = self.mask if self.mask is not None else np.ones(grid.vshape, bool)
        _, wanted = _mosaic_tiles(aoi, grid, max(self.finest, grid.step / 2), need)
        return len(wanted) * TILE_PX ** 2 * 0.8 + 64_000   # WebP ~0.8 bytes/px (measured); archive index


class Mapterhorn(DemProvider):
    key = "mapterhorn"
    label = "Mapterhorn terrain tiles (global, 0.25-20 m)"
    res_m = 1.0
    prefer = 0
    attribution = ATTRIBUTION_BASE
    licence = LICENCE
    max_area_km2 = 50_000.0
    sample = (46.0207, 7.7491, 4000)   # Zermatt (swissALTI3D 0.5 m)

    def variants(self, aoi, grid):
        if os.environ.get("MAPTOMESH_MAPTERHORN", "1") == "0":
            return []
        masks = coverage_masks(aoi, grid)
        if not masks:
            return []
        meta = sources()
        res_of = lambda k: float(meta.get(k, {}).get("resolution", 30.0))  # noqa: E731
        union = np.zeros(grid.vshape, bool)
        for m in masks.values():
            union |= m
        core = _erode(union, _edge_m(aoi), grid.step)
        tiers: dict[float, list[str]] = {}
        for src, m in masks.items():
            if (m & core).any():
                tiers.setdefault(_tier_res(res_of(src), grid.step), []).append(src)
        out = []
        for _, srcs in sorted(tiers.items()):
            srcs.sort(key=res_of)
            sel = np.zeros_like(core)
            for k in srcs:
                sel |= masks[k]
            out.append(MapterhornTier(srcs, max(map(res_of, srcs)), min(map(res_of, srcs)), core & sel, meta))
        edge = union & ~core
        if edge.any():
            srcs = sorted((k for k, m in masks.items() if (m & edge).any()), key=res_of)
            out.append(MapterhornTier(srcs, EDGE_RES, min(map(res_of, srcs)), edge, meta, band=True))
        return out

    def fetch(self, aoi, grid, progress, cancel):   # all tiers at once (tests/probe_sources.py)
        tiers = self.variants(aoi, grid)
        dtm = np.full(grid.vshape, np.nan)
        detail, notes = [], []
        for t in tiers:
            arr = t.fetch(aoi, grid, progress, cancel)
            merge_fill(dtm, arr.dtm)
            detail.append(arr.detail)
            if t.licence_note:
                notes.append(t.licence_note)
        return DemArrays(dtm, None, "; ".join(detail), licence_notes=notes)


PROVIDER = register(Mapterhorn())
