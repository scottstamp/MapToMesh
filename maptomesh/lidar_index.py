"""CanElevation LiDAR project/tile indexes.

NRCan publishes two zipped shapefiles (EPSG:4617, lon/lat):
  * projects index (~8 MB)   -> coverage overlay for the map
  * tiles index    (~230 MB) -> one record per COPC tile with its URL

The tile index is downloaded once and converted into a compact ``.npz``
(bounding boxes + project ids + URL suffixes) that loads in well under a second.
Shapefile parts are parsed directly with numpy so no GDAL is required.
"""

from __future__ import annotations

import json
import re
import shutil
import struct
import threading
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import requests

from . import config

ProgressFn = Callable[[float, str], None]

TILES_NPZ = config.CACHE_DIR / "tile_index.npz"
PROJECTS_GEOJSON = config.CACHE_DIR / "projects.geojson"


def _noop(_f: float, _m: str) -> None:
    pass


def project_year(name: str) -> int:
    """Best-effort acquisition year from a project name (0 if unknown)."""
    years = [int(y) for y in re.findall(r"(?<!\d)(19[89]\d|20[0-4]\d)(?!\d)", name)]
    if years:
        return max(years)
    # Names like ..._20210525_... (yyyymmdd)
    m = re.findall(r"(?<!\d)(20[0-4]\d)[01]\d[0-3]\d(?!\d)", name)
    return max(int(y) for y in m) if m else 0


def download(url: str, dest: Path, progress: ProgressFn = _noop, label: str = "") -> Path:
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60, headers={"User-Agent": config.USER_AGENT}) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0)) or None
        done = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    progress(done / total, f"Downloading {label} {done / 1e6:.0f}/{total / 1e6:.0f} MB")
    shutil.move(tmp, dest)
    return dest


# ---------------------------------------------------------------- shapefile parsing

def _find_member(zf: zipfile.ZipFile, ext: str) -> str:
    for n in zf.namelist():
        if n.lower().endswith(ext):
            return n
    raise FileNotFoundError(f"No {ext} in {zf.filename}")


def _shp_bboxes(zf: zipfile.ZipFile) -> np.ndarray:
    """Per-record bounding boxes (minx, miny, maxx, maxy) from .shx + .shp."""
    shx = zf.read(_find_member(zf, ".shx"))
    n = (len(shx) - 100) // 8
    idx = np.frombuffer(shx, dtype=">i4", offset=100, count=n * 2).reshape(n, 2)
    offsets = idx[:, 0].astype(np.int64) * 2  # 16-bit words -> bytes
    shp = np.frombuffer(zf.read(_find_member(zf, ".shp")), dtype=np.uint8)
    shape_type = shp[(offsets + 8)[:, None] + np.arange(4)].copy().view("<i4").ravel()
    raw = shp[(offsets + 12)[:, None] + np.arange(32)].copy()
    bbox = raw.view("<f8").reshape(n, 4)
    bbox[shape_type == 0] = np.nan  # null shapes
    return bbox


def _dbf_columns(zf: zipfile.ZipFile, wanted: list[str], progress: ProgressFn = _noop,
                 chunk_records: int = 50_000) -> dict[str, list[str]]:
    """Stream selected character columns from a (large) .dbf inside a zip."""
    with zf.open(_find_member(zf, ".dbf")) as f:
        head = f.read(32)
        nrec, hlen, rlen = struct.unpack("<IHH", head[4:12])
        fields_raw = f.read(hlen - 32)
        fields = []
        pos = 1  # deletion flag
        for i in range(0, len(fields_raw) - 1, 32):
            d = fields_raw[i:i + 32]
            if d[0] == 0x0D:
                break
            name = d[:11].split(b"\x00")[0].decode("ascii")
            size = d[16]
            fields.append((name, pos, size))
            pos += size
        out: dict[str, list[str]] = {w: [] for w in wanted}
        sel = [(n, p, s) for (n, p, s) in fields if n in wanted]
        done = 0
        while done < nrec:
            k = min(chunk_records, nrec - done)
            buf = f.read(k * rlen)
            arr = np.frombuffer(buf, dtype=np.uint8).reshape(k, rlen)
            for name, p, s in sel:
                col = arr[:, p:p + s].copy().view(f"S{s}").ravel()
                out[name].extend(x.decode("utf-8", "replace").strip() for x in col)
            done += k
            progress(done / nrec, f"Indexing tiles {done:,}/{nrec:,}")
    return out


# ---------------------------------------------------------------- tile index

@dataclass
class Tile:
    url: str
    project: str
    year: int
    bbox: tuple[float, float, float, float]
    crs: str | None = None   # used when the file carries no CRS (e.g. "EPSG:2180+EPSG:9651")
    provider: str = "nrcan"  # point-cloud provider key (for credits)
    class_map: dict | None = None  # source class -> ASPRS class (e.g. GeoSN uses 20 for non-ground)


class TileIndex:
    def __init__(self, bbox: np.ndarray, project_id: np.ndarray, urls: list[str], projects: list[str]):
        self.bbox = bbox
        self.project_id = project_id
        self.urls = urls
        self.projects = projects
        self.years = [project_year(p) for p in projects]

    @classmethod
    def load(cls, path: Path = TILES_NPZ) -> "TileIndex":
        z = np.load(path, allow_pickle=False)
        urls = bytes(z["urls"]).decode("utf-8").split("\n")
        projects = bytes(z["projects"]).decode("utf-8").split("\n")
        return cls(z["bbox"], z["project_id"], urls, projects)

    def query(self, minx: float, miny: float, maxx: float, maxy: float) -> list[Tile]:
        b = self.bbox
        hit = np.nonzero((b[:, 0] <= maxx) & (b[:, 2] >= minx) & (b[:, 1] <= maxy) & (b[:, 3] >= miny))[0]
        out = []
        for i in hit:
            u = self.urls[i]
            if not u.startswith("http"):
                u = config.LIDAR_BASE + u
            p = self.projects[self.project_id[i]]
            out.append(Tile(u, p, self.years[self.project_id[i]], tuple(float(v) for v in b[i])))
        return out


def build_tile_index(progress: ProgressFn = _noop, zip_path: Path | None = None) -> Path:
    zip_path = zip_path or (config.CACHE_DIR / "tiles_index.zip")
    if not zip_path.exists():
        download(config.TILES_ZIP_URL, zip_path, lambda f, m: progress(0.7 * f, m), "tile index")
    progress(0.7, "Reading tile footprints")
    with zipfile.ZipFile(zip_path) as zf:
        bbox = _shp_bboxes(zf)
        cols = _dbf_columns(zf, ["Project", "URL"], lambda f, m: progress(0.72 + 0.25 * f, m))
    projects = sorted(set(cols["Project"]))
    pid = {p: i for i, p in enumerate(projects)}
    project_id = np.array([pid[p] for p in cols["Project"]], dtype=np.int32)
    urls = [u[len(config.LIDAR_BASE):] if u.startswith(config.LIDAR_BASE) else u for u in cols["URL"]]
    ok = np.isfinite(bbox).all(axis=1)
    if not ok.all():
        bbox, project_id = bbox[ok], project_id[ok]
        urls = [u for u, k in zip(urls, ok) if k]
    np.savez_compressed(TILES_NPZ, bbox=bbox, project_id=project_id,
             urls=np.frombuffer("\n".join(urls).encode("utf-8"), dtype=np.uint8),
             projects=np.frombuffer("\n".join(projects).encode("utf-8"), dtype=np.uint8))
    progress(1.0, f"Tile index ready ({len(urls):,} tiles)")
    try:
        zip_path.unlink()
    except OSError:
        pass
    return TILES_NPZ


# ---------------------------------------------------------------- projects overlay

def _simplify_ring(pts: list, tol: float) -> list:
    """Radial-distance decimation; keeps closure. Good enough for an overlay."""
    if len(pts) <= 8:
        return [list(p) for p in pts]
    a = np.asarray(pts)
    keep = [0]
    last = a[0]
    for i in range(1, len(a) - 1):
        if abs(a[i, 0] - last[0]) > tol or abs(a[i, 1] - last[1]) > tol:
            keep.append(i)
            last = a[i]
    keep.append(len(a) - 1)
    if len(keep) < 4:
        return [list(p) for p in pts]
    return [[round(float(a[i, 0]), 5), round(float(a[i, 1]), 5)] for i in keep]


def build_projects_geojson(progress: ProgressFn = _noop, tol: float = 0.004) -> Path:
    import shapefile  # pyshp

    zpath = config.CACHE_DIR / "projects_index.zip"
    download(config.PROJECTS_ZIP_URL, zpath, progress, "project index")
    feats = []
    with zipfile.ZipFile(zpath) as zf:
        r = shapefile.Reader(
            shp=zf.open(_find_member(zf, ".shp")),
            shx=zf.open(_find_member(zf, ".shx")),
            dbf=zf.open(_find_member(zf, ".dbf")),
        )
        for sr in r.iterShapeRecords():
            s, rec = sr.shape, sr.record.as_dict()
            parts = list(s.parts) + [len(s.points)]
            rings = [_simplify_ring(s.points[parts[i]:parts[i + 1]], tol) for i in range(len(parts) - 1)]
            rings = [rg for rg in rings if len(rg) >= 4]
            if not rings:
                continue
            feats.append({
                "type": "Feature",
                # Rings are emitted as separate polygons; good enough for display.
                "geometry": {"type": "MultiPolygon", "coordinates": [[rg] for rg in rings]},
                "properties": {"project": rec.get("Project", ""), "provider": rec.get("Provider", ""),
                               "year": project_year(rec.get("Project", "")), "url": rec.get("URL", "")},
            })
    PROJECTS_GEOJSON.write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
    try:
        zpath.unlink()
    except OSError:
        pass
    return PROJECTS_GEOJSON


# ---------------------------------------------------------------- singleton manager

class IndexManager:
    """Lazily builds/loads indexes in a background thread, exposing status."""

    def __init__(self):
        self._lock = threading.Lock()
        self._index: TileIndex | None = None
        self.state = "idle"
        self.progress = 0.0
        self.message = ""
        self.error: str | None = None
        self._thread: threading.Thread | None = None

    @property
    def ready(self) -> bool:
        return self._index is not None

    def get(self) -> TileIndex:
        if self._index is None:
            raise RuntimeError("LiDAR tile index is not ready yet")
        return self._index

    def _set(self, f: float, m: str) -> None:
        self.progress, self.message = f, m

    def _run(self) -> None:
        try:
            if not PROJECTS_GEOJSON.exists():
                self.state = "building"
                build_projects_geojson(lambda f, m: self._set(0.0, m))
            if not TILES_NPZ.exists():
                self.state = "building"
                build_tile_index(self._set)
            self._set(1.0, "Loading tile index")
            self._index = TileIndex.load()
            self.state = "ready"
            self._set(1.0, f"{len(self._index.urls):,} LiDAR tiles in {len(self._index.projects)} projects")
        except Exception as e:  # noqa: BLE001
            self.state = "error"
            self.error = f"{type(e).__name__}: {e}"

    def load_now(self) -> bool:
        """Load an already-built index synchronously (job worker processes); False if not built."""
        with self._lock:
            if self._index is None and TILES_NPZ.exists():
                self._index = TileIndex.load()
                self.state = "ready"
        return self.ready

    def start(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            if self.state in ("ready",):
                return
            self.error = None
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()


INDEX = IndexManager()
