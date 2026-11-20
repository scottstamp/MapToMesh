"""Fetch LiDAR COPC tiles for an AOI and bin them onto the vertex grid.

Sources: NRCan CanElevation (Canada, own tile index, COPC) and USGS 3DEP in the US - either the
AWS Entwine collection (octree nodes, ``usgs_ept``) or Microsoft Planetary Computer (COPC,
``usgs3dep``). Every tile / node is an independent job; an area on the border simply gets data
from both countries and the projects merge like any overlapping projects.

Each COPC file is queried remotely (HTTP range requests) for only the octree
nodes intersecting the AOI and only down to the level of detail needed for the
grid spacing, so a 1 km tile at 2 m spacing is ~1-3 MB of transfer.

Per vertex we accumulate (ASPRS classes):
  ground   local plane fit (cell + weighted neighbours) through ground points of classes
           2 ground, 8 key-point, 9 water, 11 road surface, 20 ignored ground (3DEP: ground
           next to breaklines), evaluated at the vertex - unlike a
           plain mean this does not depend on where inside a cell the points fall, which
           matters on cliffs where one 5 m cell can span tens of metres of height
  water    count of 9
  building max z of 6
  bridge   max z of 17
  veg      max z of 3/4/5
  unclass  max z of 0/1 - depending on the project this holds trees, buildings or both
  lowest   min z of anything (fallback ground for unclassified projects)
Classification practice differs per project (e.g. Ottawa 2020 leaves buildings
in class 1, Hamilton 2021 leaves trees in class 1), so per project we record
whether buildings / vegetation are classified at all and let the surface
builder route class 1 accordingly. Projects are merged oldest -> newest.
"""

from __future__ import annotations

import concurrent.futures as cf
import io
import json
import logging
import math
import uuid
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from typing import Callable

import numpy as np
import laspy
from laspy import CopcReader, DecompressionSelection
from laspy.copc import Bounds
from pyproj import CRS, Transformer
from scipy import ndimage

from . import config
from .geo import AOI, Grid
from .httpcache import RANGE_CACHE, install_laspy_cache
from . import usgs3dep, usgs_ept
from .lidar_index import Tile, TileIndex
from .raster import fill_polygon

log = logging.getLogger(__name__)


def _failure(what: str) -> str:
    """Log the exception being handled (its text can hold paths, URLs with tokens, internals)
    under a short id and return ``what`` with that id: the text users see."""
    eid = uuid.uuid4().hex[:8]
    log.warning("%s [error %s]", what, eid, exc_info=True)
    return f"{what} (error id {eid})"

ProgressFn = Callable[[float, str], None]

GROUND = (2, 8, 9, 11, 20)
NOISE = (7, 18)


class Cancelled(Exception):
    pass


@dataclass
class _Acc:
    """Per-project accumulators on the vertex grid."""
    shape: tuple[int, int]
    gsum: np.ndarray = field(init=False)
    gcnt: np.ndarray = field(init=False)
    gmom: np.ndarray = field(init=False)  # ground moments: dx, dy, dx2, dxdy, dy2, dxz, dyz
    wcnt: np.ndarray = field(init=False)
    bmax: np.ndarray = field(init=False)
    brmax: np.ndarray = field(init=False)
    vmax: np.ndarray = field(init=False)
    umax: np.ndarray = field(init=False)
    n_bld: int = 0
    n_veg: int = 0
    n_pts: int = 0
    lowest: np.ndarray = field(init=False)
    total: np.ndarray = field(init=False)
    covered: np.ndarray = field(init=False)

    def __post_init__(self):
        n = self.shape[0] * self.shape[1]
        self.gsum = np.zeros(n, np.float64)
        self.gcnt = np.zeros(n, np.int32)
        self.gmom = np.zeros((7, n), np.float64)
        self.wcnt = np.zeros(n, np.int32)
        self.bmax = np.full(n, -np.inf, np.float32)
        self.brmax = np.full(n, -np.inf, np.float32)
        self.vmax = np.full(n, -np.inf, np.float32)
        self.umax = np.full(n, -np.inf, np.float32)
        self.lowest = np.full(n, np.inf, np.float32)
        self.total = np.zeros(n, np.int32)
        self.covered = np.zeros(self.shape, bool)

    def add(self, idx: np.ndarray, z: np.ndarray, cls: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> None:
        n = self.gsum.size
        self.total += np.bincount(idx, minlength=n).astype(np.int32)
        np.minimum.at(self.lowest, idx, z)
        g = np.isin(cls, GROUND)
        gi, gz = idx[g], z[g].astype(np.float64)
        gx, gy = dx[g].astype(np.float64), dy[g].astype(np.float64)
        self.gsum += np.bincount(gi, weights=gz, minlength=n)
        self.gcnt += np.bincount(gi, minlength=n).astype(np.int32)
        for k, wgt in enumerate((gx, gy, gx * gx, gx * gy, gy * gy, gx * gz, gy * gz)):
            self.gmom[k] += np.bincount(gi, weights=wgt, minlength=n)
        w = cls == 9
        self.wcnt += np.bincount(idx[w], minlength=n).astype(np.int32)
        vsel = np.isin(cls, (3, 4, 5))
        bsel = cls == 6
        self.n_pts += len(cls)
        self.n_bld += int(bsel.sum())
        self.n_veg += int(vsel.sum())
        for sel, arr in ((bsel, self.bmax), (cls == 17, self.brmax), (vsel, self.vmax), (cls <= 1, self.umax)):
            if sel.any():
                np.maximum.at(arr, idx[sel], z[sel])


@dataclass
class LidarResult:
    ground: np.ndarray       # NaN where no ground points
    water_frac: np.ndarray   # fraction of ground-ish returns classified water
    building: np.ndarray     # NaN where none
    bridge: np.ndarray
    veg: np.ndarray
    unclass: np.ndarray         # class 0/1 max
    bld_classified: np.ndarray  # vertex owner project classifies buildings (6)
    veg_classified: np.ndarray  # ... and vegetation (3/4/5)
    density: np.ndarray      # points per vertex
    covered: np.ndarray      # vertex inside a processed tile footprint
    info: dict


def _file_crs(reader: CopcReader, fallback: str | None = None) -> tuple[CRS | None, float]:
    """Horizontal CRS of a file and the factor taking its Z values to metres (many 3DEP
    projects use US survey feet; a horizontal-only CRS in feet implies heights in feet).
    ``fallback`` (e.g. "EPSG:3794") is used when the file has no CRS record."""
    try:
        crs = reader.header.parse_crs()
    except Exception:  # noqa: BLE001
        crs = None
    if crs is None and fallback:
        crs = CRS.from_user_input(fallback)
    if crs is None:
        return None, 1.0
    zf = 1.0
    try:
        if crs.is_compound and len(crs.sub_crs_list) > 1:
            zf = crs.sub_crs_list[1].axis_info[0].unit_conversion_factor
        elif len(crs.axis_info) >= 3:
            zf = crs.axis_info[2].unit_conversion_factor
        elif crs.is_projected and crs.axis_info:
            zf = crs.axis_info[0].unit_conversion_factor
    except Exception:  # noqa: BLE001
        zf = 1.0
    zf = zf if 0.2 < zf < 2.0 else 1.0
    try:
        return crs.to_2d(), zf
    except Exception:  # noqa: BLE001
        return crs, zf


install_laspy_cache()  # every COPC byte read (NRCan, USGS) is kept in the on-disk range cache


def _keep(cls: np.ndarray, pts) -> np.ndarray:
    keep = ~np.isin(cls, NOISE)
    try:
        keep &= ~np.asarray(pts.withheld, dtype=bool)
    except Exception:  # noqa: BLE001
        pass
    return keep


def _bin(lx: np.ndarray, ly: np.ndarray, z: np.ndarray, cls: np.ndarray, grid: Grid):
    """Points in local metres -> (vertex index, z, class, dx, dy) for points on the grid."""
    fx = (lx - grid.x0) / grid.step
    fy = (ly - grid.y0) / grid.step
    j = np.rint(fx).astype(np.int64)
    i = np.rint(fy).astype(np.int64)
    ok = (j >= 0) & (j <= grid.nx) & (i >= 0) & (i <= grid.ny)
    idx = (i[ok] * (grid.nx + 1) + j[ok]).astype(np.int64)
    dx = ((fx - j) * grid.step)[ok].astype(np.float32)  # offset from the vertex, metres
    dy = ((fy - i) * grid.step)[ok].astype(np.float32)
    return idx, z[ok], cls[ok], dx, dy


_MERC_LOCK = threading.Lock()
_MERC: dict = {}


def _merc_to_local(aoi: AOI) -> Transformer:
    """Web Mercator -> local transformer (pyproj transformers are not thread-safe: one per thread)."""
    key = (threading.get_ident(), aoi.lat, aoi.lon)
    with _MERC_LOCK:
        t = _MERC.get(key)
        if t is None:
            if len(_MERC) > 64:
                _MERC.clear()
            t = _MERC[key] = Transformer.from_crs("EPSG:3857", aoi.crs, always_xy=True)
    return t


def _process_ept_node(project, key: str, aoi: AOI, grid: Grid, cancel: threading.Event):
    """One AWS Entwine octree node (a small LAZ file in Web Mercator) -> binned points."""
    if cancel.is_set():
        raise Cancelled()
    las = laspy.read(io.BytesIO(usgs_ept.node_bytes(project, key)))
    if cancel.is_set():
        raise Cancelled()
    cls = np.asarray(las.classification, dtype=np.uint8)
    keep = _keep(cls, las)
    lx, ly = _merc_to_local(aoi).transform(np.asarray(las.x)[keep], np.asarray(las.y)[keep])
    z = np.asarray(las.z, dtype=np.float32)[keep]
    data = _bin(np.asarray(lx), np.asarray(ly), z, cls[keep], grid)
    return (None, data if len(data[0]) else None), "ok"


# octree level of detail: point spacing of at most this many grid cells. Ground-only jobs read one
# level coarser (3x fewer points, ~40% faster; St. John's harbour at 2000 cells: 0.3% of cells change
# elevation band, model heights p99 0.03 mm); land-cover models keep the finer level, which their
# tree outlines need.
RESOLUTION_FACTOR = 0.7
GROUND_RESOLUTION_FACTOR = 1.4


def _query_box(r, tile: Tile, aoi: AOI, grid: Grid):
    """The AOI (plus margin) in the file's coordinates, clipped to the file: (crs, zf, qmin,
    qmax), or a status string ("no CRS" / "outside")."""
    crs, zf = _file_crs(r, tile.crs)
    if crs is None:
        return "no CRS"
    to_file = Transformer.from_crs(aoi.crs, crs, always_xy=True)
    ring = aoi.outline_local(margin=grid.step * 1.5)
    fx, fy = to_file.transform(ring[:, 0], ring[:, 1])
    hmin, hmax = r.header.mins, r.header.maxs
    qmin = np.array([max(fx.min(), hmin[0]), max(fy.min(), hmin[1])])
    qmax = np.array([min(fx.max(), hmax[0]), min(fy.max(), hmax[1])])
    if (qmax <= qmin).any():
        return "outside"
    return crs, zf, qmin, qmax


def copc_download_bytes(tile: Tile, aoi: AOI, grid: Grid, resolution: float) -> int:
    """Compressed bytes a job reads from ``tile``: the octree nodes ``CopcReader.query`` would
    fetch for this area and ``resolution`` (from the file's hierarchy; nothing else downloaded)."""
    from laspy.copc import load_octree_for_query
    with CopcReader.open(tile.url, http_num_threads=8) as r:
        q = _query_box(r, tile, aoi, grid)
        if isinstance(q, str):
            return 0
        _, _, qmin, qmax = q
        level_max = max(1, math.ceil(math.log2(r.copc_info.spacing / resolution)) + 1)
        bounds = Bounds(mins=qmin, maxs=qmax).ensure_3d(r.header.mins, r.header.maxs)
        nodes = load_octree_for_query(r.source, r.copc_info, r.root_page, query_bounds=bounds,
                                      level_range=range(0, level_max))
        return int(sum(n.byte_size for n in nodes))


def _process_tile(tile: Tile, aoi: AOI, grid: Grid, resolution: float, cancel: threading.Event):
    if cancel.is_set():
        raise Cancelled()
    sel = (DecompressionSelection.base() | DecompressionSelection.Z
           | DecompressionSelection.CLASSIFICATION | DecompressionSelection.FLAGS)
    with CopcReader.open(tile.url, http_num_threads=8, decompression_selection=sel) as r:
        q = _query_box(r, tile, aoi, grid)
        if isinstance(q, str):
            return None, q
        crs, zf, qmin, qmax = q
        to_local = Transformer.from_crs(crs, aoi.crs, always_xy=True)
        hmin, hmax = r.header.mins, r.header.maxs
        # tile footprint -> local grid coverage polygon
        cx = np.array([hmin[0], hmax[0], hmax[0], hmin[0]])
        cy = np.array([hmin[1], hmin[1], hmax[1], hmax[1]])
        lx, ly = to_local.transform(cx, cy)
        footprint = np.column_stack([(lx - grid.x0) / grid.step, (ly - grid.y0) / grid.step])
        pts = r.query(bounds=Bounds(mins=qmin, maxs=qmax), resolution=resolution)
        if cancel.is_set():
            raise Cancelled()
        if len(pts) == 0:
            return (footprint, None), "empty"
        cls = np.asarray(pts.classification, dtype=np.uint8)
        if tile.class_map:
            lut = np.arange(256, dtype=np.uint8)
            for src_c, dst_c in tile.class_map.items():
                lut[int(src_c)] = int(dst_c)
            cls = lut[cls]
        keep = _keep(cls, pts)
        x = np.asarray(pts.x)[keep]
        y = np.asarray(pts.y)[keep]
        z = (np.asarray(pts.z, dtype=np.float64)[keep] * zf).astype(np.float32)
        cls = cls[keep]
    lx, ly = to_local.transform(x, y)
    return (footprint, _bin(np.asarray(lx), np.asarray(ly), z, cls, grid)), "ok"


def _data_coverage(total: np.ndarray, step: float) -> np.ndarray:
    """Where a project actually has data: at least half the vertices within ~40 m hold points,
    plus fully enclosed holes (lakes without returns stay "covered", so they become water).

    The published footprints of the AWS Entwine projects are coarse and can include large areas
    the survey never captured (the inner Grand Canyon in AZ_GrandCanyonNP_1_2019 has only a few
    sparse points); treating those as covered turned them into "voids" and printed them as water.
    COPC tiles (NRCan, Planetary Computer) keep their exact tile extents instead."""
    k = max(3, int(round(40.0 / step)) | 1)
    hit = ndimage.uniform_filter((total > 0).astype(np.float32), size=k, mode="nearest")
    return ndimage.binary_fill_holes(hit >= 0.5)


_FLAGS_PATH = config.CACHE_DIR / "project_classes.json"
_FLAGS_LOCK = threading.Lock()


def _project_flags(name: str, n_bld: int, n_veg: int) -> tuple[bool, bool]:
    """Does a project classify buildings (6) / vegetation (3-5)?

    A handful of points is enough evidence (rural AOIs may contain only a few
    buildings). Positive evidence is remembered per project across runs.
    """
    with _FLAGS_LOCK:
        try:
            known = json.loads(_FLAGS_PATH.read_text())
        except (OSError, ValueError):
            known = {}
        k = known.get(name, {})
        has_b = bool(k.get("buildings")) or n_bld >= 25
        has_v = bool(k.get("vegetation")) or n_veg >= 100
        if has_b != bool(k.get("buildings")) or has_v != bool(k.get("vegetation")):
            known[name] = {"buildings": has_b, "vegetation": has_v}
            config.atomic_write(_FLAGS_PATH, json.dumps(known, indent=1, sort_keys=True))
    return has_b, has_v


def _plane_fit_ground(a: _Acc, shape: tuple[int, int], step: float) -> np.ndarray:
    """Weighted least-squares plane z = h + b*x + c*y through the ground points of each
    vertex bin and its 8 neighbours (weights 1 / 0.3 / 0.1), evaluated at the vertex.
    Falls back to the bin mean where the fit is ill-conditioned; NaN where the bin itself
    has no ground points. The native extension does the same on all cores."""
    from . import native
    H, W = shape
    if native.enabled():
        return native.ext.plane_fit_ground(a.gcnt, a.gsum, a.gmom, H, W, float(step))
    n = a.gcnt.astype(np.float64).reshape(H, W)
    sz = a.gsum.reshape(H, W)
    mx, my, mxx, mxy, myy, mxz, myz = (m.reshape(H, W) for m in a.gmom)
    S = {k: np.zeros((H, W)) for k in ("n", "x", "y", "z", "xx", "xy", "yy", "xz", "yz")}

    def sh(arr, di, dj):  # neighbour (i+di, j+dj) aligned onto (i, j)
        p = np.pad(arr, 1)
        return p[1 + di:1 + di + H, 1 + dj:1 + dj + W]

    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            w = (1.0, 0.3, 0.1)[abs(di) + abs(dj)]
            ox, oy = dj * step, di * step
            nn, sx, sy, s_z = sh(n, di, dj), sh(mx, di, dj), sh(my, di, dj), sh(sz, di, dj)
            S["n"] += w * nn
            S["x"] += w * (sx + nn * ox)
            S["y"] += w * (sy + nn * oy)
            S["z"] += w * s_z
            S["xx"] += w * (sh(mxx, di, dj) + 2 * ox * sx + nn * ox * ox)
            S["yy"] += w * (sh(myy, di, dj) + 2 * oy * sy + nn * oy * oy)
            S["xy"] += w * (sh(mxy, di, dj) + ox * sy + oy * sx + nn * ox * oy)
            S["xz"] += w * (sh(mxz, di, dj) + ox * s_z)
            S["yz"] += w * (sh(myz, di, dj) + oy * s_z)
    mean = np.where(n > 0, sz / np.maximum(n, 1), np.nan)
    out = mean.copy()
    ok = (n > 0) & (S["n"] >= 3)
    if ok.any():
        N = S["n"][ok]
        zc = S["z"][ok] / N                     # centre z for conditioning
        ridge = 1e-3 * N * step * step          # keeps near-collinear sets solvable
        A = np.empty((ok.sum(), 3, 3))
        A[:, 0, 0] = N; A[:, 0, 1] = A[:, 1, 0] = S["x"][ok]; A[:, 0, 2] = A[:, 2, 0] = S["y"][ok]
        A[:, 1, 1] = S["xx"][ok] + ridge; A[:, 1, 2] = A[:, 2, 1] = S["xy"][ok]; A[:, 2, 2] = S["yy"][ok] + ridge
        rhs = np.column_stack([np.zeros_like(N), S["xz"][ok] - zc * S["x"][ok], S["yz"][ok] - zc * S["y"][ok]])
        cond_ok = np.abs(np.linalg.det(A)) > 1e-9 * N ** 3 * step ** 4
        sol = np.zeros((len(N), 3))
        if cond_ok.any():
            sol[cond_ok] = np.linalg.solve(A[cond_ok], rhs[cond_ok][..., None])[..., 0]
        h = np.where(cond_ok, zc + sol[:, 0], mean[ok])
        out[ok] = h
    return out.ravel()


class PointCloudProvider:
    """Classified COPC point clouds outside the built-in NRCan / USGS sources. Subclasses set the
    attributes and return ``Tile`` objects (COPC URLs; ``Tile.crs`` when the files have no CRS)."""
    key: str = ""
    label: str = ""
    attribution: str = ""
    licence: str = ""
    regions: tuple = ()      # lon/lat boxes (w, s, e, n)
    sample: tuple = ()       # (lat, lon, size_m) for tests/probe_sources.py
    licence_note: str = ""   # restriction shown to the user whenever this provider's data is used

    def covers(self, bbox) -> bool:
        w, s, e, n = bbox
        return any(w <= rx1 and e >= rx0 and s <= ry1 and n >= ry0 for rx0, ry0, rx1, ry1 in self.regions)

    def tiles(self, bbox, max_items: int) -> list[Tile]:
        raise NotImplementedError


POINTCLOUD_PROVIDERS: list[PointCloudProvider] = []

# credits for the built-in sources (registered providers carry their own)
BUILTIN_CREDITS = {
    "nrcan": ("NRCan CanElevation LiDAR (Canada)",
              "Contains information licensed under the Open Government Licence - Canada (NRCan CanElevation)."),
    "usgs": ("USGS 3DEP LiDAR (US)", "USGS 3D Elevation Program (public domain)."),
}


def provider_licence_note(key: str) -> str:
    for p in POINTCLOUD_PROVIDERS:
        if p.key == key:
            return p.licence_note
    return ""


def provider_credit(key: str) -> tuple[str, str]:
    """(label, attribution) of a point-cloud provider key."""
    if key in BUILTIN_CREDITS:
        return BUILTIN_CREDITS[key]
    for p in POINTCLOUD_PROVIDERS:
        if p.key == key:
            return p.label, p.attribution
    return key, ""


def register_pointcloud(p: PointCloudProvider) -> PointCloudProvider:
    POINTCLOUD_PROVIDERS.append(p)
    return p


def find_sources(bbox: tuple[float, float, float, float], index: TileIndex | None, us_source: str = "usgs",
                 max_items: int = 5000) -> tuple[list[Tile], list, list[str]]:
    """LiDAR covering ``bbox`` (lon/lat): COPC tiles (NRCan, Planetary Computer), AWS Entwine
    projects (USGS) and any search errors."""
    tiles = index.query(*bbox, limit=max_items) if index is not None else []
    projects, errors = [], []
    if usgs3dep.in_us(bbox):
        try:
            if us_source == "planetary":
                tiles += usgs3dep.tiles(bbox, max_items)
            else:
                projects = usgs_ept.INDEX.query(bbox)
        except Exception as e:  # noqa: BLE001 - keep whatever the other provider has
            errors.append(_failure("USGS 3DEP search failed"))
    for p in POINTCLOUD_PROVIDERS:
        if p.covers(bbox):
            try:
                found = p.tiles(bbox, max_items)
                for t in found:
                    t.provider = p.key
                tiles += found
            except Exception as e:  # noqa: BLE001
                errors.append(_failure(f"{p.label} search failed"))
    return tiles, projects, errors


def empty_result(grid: Grid, info: dict | None = None) -> LidarResult:
    """No point clouds at all (elevation models fill everything)."""
    shape = grid.vshape
    nan = lambda: np.full(shape, np.nan)  # noqa: E731
    return LidarResult(nan(), np.zeros(shape), nan(), nan(), nan(), nan(), np.zeros(shape, bool),
                       np.zeros(shape, bool), np.zeros(shape, np.float32), np.zeros(shape, bool),
                       {"tiles": 0, "points": 0, "projects": [], "errors": [], "failed_tiles": 0, "coverage": 0.0,
                        "providers": [], **(info or {})})


def _hint_sources(copc_urls: list[str], ept_urls: list[str]) -> None:
    """Hook: the files a job is about to read. The server's thread pools need no hint; in the
    browser (no threads) ``maptomesh.browser`` replaces it to start the downloads in the background."""


def fetch_lidar(aoi: AOI, grid: Grid, index: TileIndex | None, progress: ProgressFn,
                cancel: threading.Event | None = None, max_tiles: int = 2500,
                workers: int = 6, us_source: str = "usgs", allow_empty: bool = False,
                ground_only: bool = False, max_ept_km2: float | None = None) -> LidarResult:
    """Classified point clouds on the grid. With ``allow_empty`` an area without point clouds,
    or one over the tile limit, returns an empty result (and a note) instead of raising, so that
    elevation models can take over."""
    cancel = cancel or threading.Event()
    bbox = aoi.lonlat_bbox(margin=grid.step * 2)
    tiles, projects, search_errors = find_sources(bbox, index, us_source, max_tiles + 1)

    def give_up(msg: str, note: str | None = None) -> LidarResult:
        if not allow_empty:
            raise RuntimeError(msg)
        return empty_result(grid, {"errors": search_errors[:5], "skipped": [note] if note else []})

    if not tiles and not projects:
        return give_up("No LiDAR point clouds (NRCan CanElevation or USGS 3DEP) cover this area. "
                       "Enable the coverage overlay, or use elevation models only."
                       + (f" {search_errors[0]}" if search_errors else ""))
    if len(tiles) > max_tiles:
        return give_up(f"{len(tiles)} LiDAR tiles intersect this area (limit {max_tiles}). "
                       "Pick a smaller area or use elevation models only.",
                       f"{len(tiles)} LiDAR tiles is over the {max_tiles} limit: used elevation models instead.")
    ept_km2 = max_tiles if max_ept_km2 is None else max_ept_km2
    if projects and aoi.width_m * aoi.height_m > ept_km2 * 1e6:
        return give_up(f"Area over {ept_km2:g} km2: pick a smaller area for USGS 3DEP point clouds.",
                       f"Area over {ept_km2:g} km2 for USGS point clouds at this detail: used elevation "
                       "models instead.")
    provider_of = {t.project: t.provider for t in tiles}
    provider_of.update({p.label: "usgs" for p in projects})
    # octree LOD giving ~1 point per (step/2)^2 at most
    resolution = max(grid.step * (GROUND_RESOLUTION_FACTOR if ground_only else RESOLUTION_FACTOR), 0.25)
    hit0, miss0, req0 = RANGE_CACHE.counters()
    shape = grid.vshape
    accs: dict[str, _Acc] = defaultdict(lambda: _Acc(shape))
    years: dict[str, int] = {}
    errors: list[str] = list(search_errors)
    ept_labels: set[str] = set()
    # jobs: (project, year, callable -> ((footprint | None, data | None) | None, status))
    jobs = [(t.project, t.year, partial(_process_tile, t, aoi, grid, resolution, cancel)) for t in tiles]
    ept_urls: list[str] = []
    if projects:
        progress(0.0, f"Reading the octree of {len(projects)} USGS 3DEP project(s)")
        with cf.ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(usgs_ept.plan, p, aoi, grid, resolution): p for p in projects}
            for fut in cf.as_completed(futs):
                p = futs[fut]
                try:
                    pl = fut.result()
                except Exception as e:  # noqa: BLE001
                    errors.append(_failure(f"{p.name}: could not read the point cloud index"))
                    continue
                if not pl.keys:
                    continue
                acc = accs[p.label]
                years[p.label] = p.year
                ept_labels.add(p.label)
                acc.covered |= fill_polygon(pl.footprint, shape)
                jobs += [(p.label, p.year, partial(_process_ept_node, p, k, aoi, grid, cancel)) for k in pl.keys]
                ept_urls += [f"{p.url}ept-data/{k}.laz" for k in pl.keys]
        if cancel.is_set():
            raise Cancelled()
    if not jobs:
        if allow_empty:
            return empty_result(grid, {"errors": errors[:5]})
        raise RuntimeError("No LiDAR point clouds cover this area." + (f" {errors[0]}" if errors else ""))
    npts = 0
    done = 0
    n_proj = len({j[0] for j in jobs})
    unit = "tiles" if len(jobs) == len(tiles) else "octree nodes" if not tiles else "tiles / octree nodes"
    progress(0.0, f"Fetching {len(jobs)} LiDAR {unit} from {n_proj} project(s)")
    _hint_sources([t.url for t in tiles], ept_urls)
    failed = 0
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fn): (proj, year) for proj, year, fn in jobs}
        try:
            for fut in cf.as_completed(futs):
                if cancel.is_set():
                    raise Cancelled()
                proj, year = futs[fut]
                try:
                    payload, status = fut.result()
                except Cancelled:
                    raise
                except Exception as e:  # noqa: BLE001
                    errors.append(_failure("A LiDAR tile could not be read"))
                    failed += 1
                    payload = None
                done += 1
                if payload is not None:
                    acc = accs[proj]
                    years[proj] = year
                    footprint, data = payload
                    if footprint is not None:
                        acc.covered |= fill_polygon([footprint], shape)
                    if data is not None:
                        idx, z, cls, dx, dy = data
                        acc.add(idx, z, cls, dx, dy)
                        npts += len(idx)
                progress(done / len(jobs), f"LiDAR {unit} {done}/{len(jobs)} - {npts / 1e6:.1f} M points")
        except BaseException:
            cancel.set()
            for f in futs:
                f.cancel()
            raise
    if failed == len(jobs):
        if allow_empty:
            return empty_result(grid, {"errors": errors[-5:],
                                       "skipped": ["Every LiDAR request failed: used elevation models instead."]})
        raise RuntimeError("All LiDAR requests failed: " + errors[-1])

    for label in ept_labels:
        accs[label].covered &= _data_coverage(accs[label].total.reshape(shape), grid.step)

    # merge projects, oldest first
    n = shape[0] * shape[1]
    ground = np.full(n, np.nan); water = np.zeros(n); bld = np.full(n, np.nan)
    brd = np.full(n, np.nan); veg = np.full(n, np.nan); unc = np.full(n, np.nan); dens = np.zeros(n)
    bcls = np.zeros(n, bool); vcls = np.zeros(n, bool); cov = np.zeros(shape, bool)
    proj_info = []
    used_providers: list[str] = []
    order = sorted(accs, key=lambda p: years.get(p, 0))
    for p in order:
        a = accs[p]
        has = a.total > 0
        has_ground_class = a.gcnt.sum() > 0.02 * max(a.total.sum(), 1)
        if has_ground_class:
            g = _plane_fit_ground(a, shape, grid.step)
        else:  # project without ground classification: lowest return
            g = np.where(np.isfinite(a.lowest), a.lowest, np.nan)
        ground[has] = g[has]
        water[has] = np.where(a.gcnt[has] > 0, a.wcnt[has] / np.maximum(a.gcnt[has], 1), 0)
        for dst, src in ((bld, a.bmax), (brd, a.brmax), (veg, a.vmax), (unc, a.umax)):
            s = np.where(np.isfinite(src), src, np.nan)
            dst[has] = s[has]
        has_b, has_v = _project_flags(p, a.n_bld, a.n_veg)
        bcls[has] = has_b
        vcls[has] = has_v
        proj_info.append({"name": p, "year": years.get(p, 0), "buildings_classified": bool(has_b),
                          "vegetation_classified": bool(has_v), "ground_classified": bool(has_ground_class)})
        dens[has] = a.total[has]
        cov |= a.covered
        prov = provider_of.get(p, "nrcan")
        if has.any() and prov not in used_providers:
            used_providers.append(prov)
    rs = lambda a: a.reshape(shape)  # noqa: E731
    info = {
        "tiles": len(tiles),
        "ept_nodes": len(jobs) - len(tiles),
        "us_source": us_source,
        "points": int(npts),
        "projects": proj_info,
        "errors": errors[:5],
        "failed_tiles": failed,
        "coverage": float(cov.mean()),
        "query_resolution_m": resolution,
        "providers": used_providers,
    }
    hit1, miss1, req1 = RANGE_CACHE.counters()
    info["downloaded_mb"] = round((miss1 - miss0) / 1e6, 1)
    info["from_cache_mb"] = round((hit1 - hit0) / 1e6, 1)
    info["http_requests"] = req1 - req0
    return LidarResult(rs(ground), rs(water), rs(bld), rs(brd), rs(veg), rs(unc), rs(bcls), rs(vcls),
                       rs(dens), cov, info)
