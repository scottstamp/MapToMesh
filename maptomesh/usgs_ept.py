"""USGS 3DEP LiDAR from the AWS Entwine Point Tile (EPT) collection (``s3://usgs-lidar-public``).

USGS / Hobu publish every 3DEP point-cloud project (2,000+, kept current) as an Entwine octree
in Web Mercator: ``<project>/ept.json`` (bounds, span), ``ept-hierarchy/<key>.json`` (point
counts per node; -1 = subtree in its own file) and ``ept-data/<key>.laz`` (one small LAZ file
per node). Project footprints come from Hobu's ``resources.geojson``.

For an area we pick the projects whose footprint intersects it, walk each octree down to the
depth whose point spacing matches the grid (the same rule as the COPC ``resolution``) and
fetch the intersecting nodes whole through the persistent range cache. No token needed.
Heights are NAVD88 metres (checked against the Planetary Computer COPC copy of a project).
"""

from __future__ import annotations

import itertools
import json
import math
import re
import threading
import time
from dataclasses import dataclass

import numpy as np
import requests
from pyproj import Transformer

from . import config
from .httpcache import RANGE_CACHE
from .raster import fill_polygon

EPT_BASE = "https://s3-us-west-2.amazonaws.com/usgs-lidar-public/"
RESOURCES_URLS = (
    "https://raw.githubusercontent.com/hobuinc/usgs-lidar/master/boundaries/resources.geojson",
    "https://usgs.entwine.io/boundaries/resources.geojson",
)
RESOURCES_PATH = config.CACHE_DIR / "usgs_resources.geojson"
OVERLAY_PATH = config.CACHE_DIR / "usgs_projects.geojson.gz"
REFRESH_S = 30 * 86400
MAX_NODES = 20000
_WHOLE = 1 << 28  # "read the whole object" (the cache clamps to the real size)


@dataclass
class EptProject:
    name: str        # e.g. "WA_KingCo_1_2021"
    label: str       # display / merge name, e.g. "USGS WA_KingCo_1_2021"
    year: int
    url: str         # base URL ending in "/"
    bbox: tuple[float, float, float, float]
    rings: list[np.ndarray]  # lon/lat rings of the footprint (even-odd)


def project_label(name: str) -> str:
    """Display name shared with the Planetary Computer copy of the same project."""
    return "USGS " + (name[len("USGS_LPC_"):] if name.startswith("USGS_LPC_") else name)


def project_year(name: str) -> int:
    m = re.search(r"(?<!\d)(19[89]\d|20[0-4]\d)(?!\d)", name)
    if m:
        return int(m.group(1))
    m = re.search(r"_[A-Z](\d{2})(?:_|$)", name)  # USGS work-unit suffixes: _B22, _D24
    return 2000 + int(m.group(1)) if m else 0


# ---------------------------------------------------------------- project footprints

def _download_resources() -> None:
    last = None
    for url in RESOURCES_URLS:
        try:
            r = requests.get(url, headers={"User-Agent": config.USER_AGENT}, timeout=(20, 180))
            r.raise_for_status()
            json.loads(r.content)  # complete and valid before replacing the cached copy
            tmp = RESOURCES_PATH.with_suffix(".tmp")
            tmp.write_bytes(r.content)
            tmp.replace(RESOURCES_PATH)
            OVERLAY_PATH.unlink(missing_ok=True)
            return
        except Exception as e:  # noqa: BLE001 - try the mirror
            last = e
    raise RuntimeError(f"USGS 3DEP project list unavailable: {last}")


class ProjectIndex:
    def __init__(self):
        self._lock = threading.Lock()
        self._projects: list[EptProject] | None = None
        self._bbox: np.ndarray | None = None

    def _load(self) -> None:
        stale = not RESOURCES_PATH.exists() or time.time() - RESOURCES_PATH.stat().st_mtime > REFRESH_S
        if stale:
            try:
                _download_resources()
            except RuntimeError:
                if not RESOURCES_PATH.exists():
                    raise  # an old copy is better than nothing
        gj = json.loads(RESOURCES_PATH.read_text(encoding="utf-8"))
        projects, boxes = [], []
        for f in gj["features"]:
            g, props = f.get("geometry"), f.get("properties", {})
            if not g or not props.get("url"):
                continue
            polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
            rings = [np.asarray(r, dtype=np.float64)[:, :2] for poly in polys for r in poly if len(r) >= 3]
            if not rings:
                continue
            allp = np.concatenate(rings)
            bb = (float(allp[:, 0].min()), float(allp[:, 1].min()), float(allp[:, 0].max()), float(allp[:, 1].max()))
            name = props["name"]
            base = props["url"].rsplit("/", 1)[0] + "/"
            projects.append(EptProject(name, project_label(name), project_year(name), base, bb, rings))
            boxes.append(bb)
        self._projects = projects
        self._bbox = np.asarray(boxes)

    def get(self) -> list[EptProject]:
        with self._lock:
            if self._projects is None:
                self._load()
            return self._projects

    def query(self, bbox: tuple[float, float, float, float]) -> list[EptProject]:
        """Projects whose footprint intersects ``bbox`` (lon/lat)."""
        projects = self.get()
        w, s, e, n = bbox
        b = self._bbox
        hit = np.nonzero((b[:, 0] <= e) & (b[:, 2] >= w) & (b[:, 1] <= n) & (b[:, 3] >= s))[0]
        out = []
        N = 64  # coarse raster of the box: exact enough to drop projects that only share a bbox
        for i in hit:
            p = projects[i]
            rings = [np.column_stack([(r[:, 0] - w) / (e - w) * N, (r[:, 1] - s) / (n - s) * N]) for r in p.rings]
            if fill_polygon(rings, (N + 1, N + 1)).any() or any(
                    ((r[:, 0] >= w) & (r[:, 0] <= e) & (r[:, 1] >= s) & (r[:, 1] <= n)).any() for r in p.rings):
                out.append(p)
        return out

    def overlay(self) -> bytes:
        """Footprints as gzipped GeoJSON for the coverage map (built once, cached). The boundaries
        are already coarse (~80 vertices per project): rounding to ~10 m + gzip makes ~1 MB."""
        import gzip

        with self._lock:
            if OVERLAY_PATH.exists() and RESOURCES_PATH.exists() \
                    and OVERLAY_PATH.stat().st_mtime >= RESOURCES_PATH.stat().st_mtime:
                return OVERLAY_PATH.read_bytes()
        feats = []
        for p in self.get():
            rings = [np.round(r, 4).tolist() for r in p.rings]
            feats.append({"type": "Feature", "properties": {"project": p.label, "year": p.year},
                          "geometry": {"type": "MultiPolygon", "coordinates": [[r] for r in rings if len(r) >= 4]}})
        data = gzip.compress(json.dumps({"type": "FeatureCollection", "features": feats},
                                        separators=(",", ":")).encode(), 6)
        OVERLAY_PATH.write_bytes(data)
        return data


INDEX = ProjectIndex()


# ---------------------------------------------------------------- octree

def _json(url: str) -> dict:
    return json.loads(RANGE_CACHE.read(url, 0, _WHOLE))


def node_bytes(project: EptProject, key: str) -> bytes:
    return RANGE_CACHE.read(f"{project.url}ept-data/{key}.laz", 0, _WHOLE)


@dataclass
class Plan:
    project: EptProject
    keys: list[str]
    footprint: list[np.ndarray]  # rings in vertex-grid coordinates (col, row)


def plan(project: EptProject, aoi, grid, resolution: float, max_nodes: int = MAX_NODES) -> Plan:
    """Octree nodes of ``project`` intersecting the AOI down to the depth matching ``resolution``
    (metres on the ground), plus the project footprint on the vertex grid."""
    ept = _json(project.url + "ept.json")
    b = ept["bounds"]
    span = float(ept.get("span", 128))
    size0 = b[3] - b[0]
    # AOI in Web Mercator (scale 1/cos(lat): ground metres are smaller than Mercator metres)
    ring = aoi.outline_local(margin=grid.step * 1.5)
    tf = Transformer.from_crs(aoi.crs, "EPSG:3857", always_xy=True)
    mx, my = tf.transform(ring[:, 0], ring[:, 1])
    box = (mx.min(), my.min(), mx.max(), my.max())
    res_m = resolution / math.cos(math.radians(aoi.lat))
    dmax = 0
    while size0 / span / 2 ** dmax > res_m and dmax < 30:
        dmax += 1

    hier: dict[str, int] = {}
    hier.update(_json(f"{project.url}ept-hierarchy/0-0-0-0.json"))
    keys: list[str] = []
    queue = ["0-0-0-0"]
    while queue:
        k = queue.pop()
        cnt = hier.get(k)
        if cnt is None:
            continue
        if cnt == -1:  # subtree continues in its own hierarchy file (which lists this node too)
            hier.update(_json(f"{project.url}ept-hierarchy/{k}.json"))
            cnt = hier.get(k, 0)
            if cnt == -1:
                continue
        d, X, Y, Z = map(int, k.split("-"))
        s = size0 / 2 ** d
        x0, y0 = b[0] + X * s, b[1] + Y * s
        if x0 > box[2] or x0 + s < box[0] or y0 > box[3] or y0 + s < box[1]:
            continue
        if cnt > 0:
            keys.append(k)
            if len(keys) > max_nodes:
                raise RuntimeError(f"USGS 3DEP project {project.name}: over {max_nodes} octree nodes "
                                   "for this area - pick a smaller area or lower detail.")
        if d < dmax:
            for dx, dy, dz in itertools.product((0, 1), repeat=3):
                ck = f"{d + 1}-{2 * X + dx}-{2 * Y + dy}-{2 * Z + dz}"
                if ck in hier:
                    queue.append(ck)
    # footprint polygon on the vertex grid
    to_local = Transformer.from_crs("EPSG:4326", aoi.crs, always_xy=True)
    fp = []
    for r in project.rings:
        lx, ly = to_local.transform(r[:, 0], r[:, 1])
        fp.append(np.column_stack([(np.asarray(lx) - grid.x0) / grid.step, (np.asarray(ly) - grid.y0) / grid.step]))
    return Plan(project, keys, fp)
