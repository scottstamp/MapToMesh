"""Project PLATEAU 3D city models (Japan, MLIT): building roofs from the PLATEAU VIEW 3D Tiles.

Japan has little open LiDAR, so its elevation is GSI's bare-earth DEM and buildings would come
from OpenStreetMap footprints at their tagged (often default) heights. PLATEAU covers ~300
municipalities with surveyed buildings: LoD2 (roof shapes) for ~250 of them, LoD1 (blocks at the
measured height) elsewhere - and the LoD2 tilesets fall back to the LoD1 block for buildings
modelled only in LoD1, so one tileset per city holds every building.

Discovery: PLATEAU VIEW's nationwide tilesets list one child per city (its bounding region and
tileset URL); each city tileset is a tree whose leaves (REPLACE refinement) hold the full-detail
content: b3dm (3D Tiles 1.0) or glb (1.1), glTF with Draco-compressed geometry
(KHR_draco_mesh_compression), decoded by the Rust extension (``maptomesh_native.decode_draco``).

Heights are ellipsoidal (JGD2011; the geoid is ~35-40 m above the ellipsoid in Japan), so each
building is kept relative to its own lowest point and stood on the model's ground: the output is
the roof height above the building's base per grid vertex plus the building it belongs to.

Terms: PLATEAU site policy (compatible with CC BY 4.0); credit the city, survey year and MLIT and
say the data was processed.
"""

from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import math
import re
import struct
import threading
import time
from collections import deque

import numpy as np

from .. import config

INDEXES = (
    ("LoD2", "https://api.plateauview.mlit.go.jp/datacatalog/3dtiles/all-bldg-lod2-notexture-latest/tileset.json"),
    ("LoD1", "https://api.plateauview.mlit.go.jp/datacatalog/3dtiles/all-bldg-lod1-latest/tileset.json"),
)
JAPAN = (122.0, 20.0, 154.0, 46.0)          # lon / lat box: no index lookups elsewhere
MAX_KM2 = 25.0                               # checked before any tile is read
MAX_MB = 250.0                               # geometry to download (central Tokyo: ~30 MB per km2)
MB_PER_KM2 = 25.0                            # for download estimates (dense wards; suburbs less)
MAX_TILES = 1500
WINDOW = 8                                   # tiles downloaded / decoded ahead of the rasteriser
HINT_MB = 48.0                               # browser: geometry announced ahead of the reads
INDEX_TTL = 86400.0                          # the nationwide indexes change with each data release
TILESET_TTL = 30 * 86400.0                   # city tileset URLs carry asset ids: new data, new URL
MAX_FIT_ERR = 0.002                          # m: local map vs pyproj at the check points (the true
                                             # largest error is up to ~2x that: still under 1 cm)
CREDIT = "3D city models: Project PLATEAU (MLIT Japan), {cities}, processed (PLATEAU site policy, CC BY 4.0 compatible)"

_lock = threading.Lock()
_memo: dict[str, tuple[float, object]] = {}   # url -> (time, parsed): index children / tilesets
_MEMO_MAX = 16


# ---------------------------------------------------------------- discovery

def _get_json(url: str, ttl: float = TILESET_TTL):
    """A JSON document (index / tileset), kept on disk for ``ttl`` seconds (``config.CACHE_DIR``,
    which the browser build keeps in its persistent cache). A plain GET: PLATEAU's API answers
    HEAD with 404 and ignores Range, so the range cache can't hold it. A stale copy is used when
    the download fails."""
    import requests
    path = config.CACHE_DIR / "plateau" / f"{hashlib.sha1(url.encode()).hexdigest()}.json"
    stale = None
    try:
        head, body = path.read_bytes().split(b"\n", 1)
        meta = json.loads(head)
        if meta.get("url") == url:
            if time.time() - float(meta.get("t", 0)) < ttl:
                return json.loads(body)
            stale = body
    except (OSError, ValueError):
        pass
    try:
        r = requests.get(url, headers={"User-Agent": config.USER_AGENT}, timeout=60)
        r.raise_for_status()
        out = json.loads(r.content)
    except (requests.RequestException, ValueError):
        if stale is None:
            raise
        return json.loads(stale)
    meta = json.dumps({"url": url, "t": time.time()}).encode()
    config.atomic_write(path, meta + b"\n" + r.content)
    return out


def _memoised(url: str, ttl: float, make):
    """``make()`` for ``url``, memoised in memory for ``ttl`` seconds (at most ``_MEMO_MAX``)."""
    with _lock:
        hit = _memo.get(url)
        if hit and time.time() - hit[0] < ttl:
            return hit[1]
    val = make()
    with _lock:
        _memo[url] = (time.time(), val)
        while len(_memo) > _MEMO_MAX:
            _memo.pop(next(iter(_memo)))
    return val


def _cities(lod: str, url: str) -> list[dict]:
    """[{"code", "name", "year", "url", "region"}] of a nationwide index (refreshed daily)."""
    def make():
        out = []
        for c in _get_json(url, INDEX_TTL)["root"].get("children", []):
            u = (c.get("content") or {}).get("uri") or (c.get("content") or {}).get("url")
            reg = (c.get("boundingVolume") or {}).get("region")
            if not u or not reg:
                continue
            m = re.search(r"/(\d{5})_([a-z0-9-]+)_[a-z]+_(\d{4})_", u)
            out.append({"code": m.group(1) if m else u, "name": m.group(2) if m else "",
                        "year": m.group(3) if m else "", "url": u, "region": reg[:4]})
        return out
    return _memoised(url, INDEX_TTL, make)


def _hits(region, bbox) -> bool:
    w, s, e, n = (math.degrees(v) for v in region)
    return w <= bbox[2] and e >= bbox[0] and s <= bbox[3] and n >= bbox[1]


def cities_for(bbox) -> list[dict]:
    """The city tilesets for a lon/lat bbox: each city's LoD2 tileset, else its LoD1 one."""
    if not (JAPAN[0] <= bbox[2] and bbox[0] <= JAPAN[2] and JAPAN[1] <= bbox[3] and bbox[1] <= JAPAN[3]):
        return []
    chosen: dict[str, dict] = {}
    for lod, url in INDEXES:
        for c in _cities(lod, url):
            if c["code"] not in chosen and _hits(c["region"], bbox):
                chosen[c["code"]] = {**c, "lod": lod}
    return list(chosen.values())


def _leaves(tileset_url: str, bbox, out: list, transform=None, refine="REPLACE", depth=0) -> None:
    """Content URLs (and the tile transform) whose nodes the bbox needs: leaves under REPLACE
    refinement, every node on the way under ADD. External tilesets are followed."""
    from urllib.parse import urljoin
    ts = _memoised(tileset_url, TILESET_TTL, lambda: _get_json(tileset_url, TILESET_TTL))

    def walk(node, tf, ref):
        bv = node.get("boundingVolume") or {}
        if "region" in bv and not _hits(bv["region"][:4], bbox):
            return
        if node.get("transform"):
            m = np.asarray(node["transform"], dtype=np.float64).reshape(4, 4).T   # column-major
            tf = m if tf is None else tf @ m
        ref = node.get("refine", ref)
        kids = node.get("children") or []
        c = node.get("content") or {}
        uri = c.get("uri") or c.get("url")
        if uri and (not kids or ref == "ADD"):
            full = urljoin(tileset_url, uri)
            if full.split("?")[0].endswith(".json"):
                if depth < 4:
                    _leaves(full, bbox, out, tf, ref, depth + 1)
            else:
                out.append((full, tf))
        for k in kids:
            walk(k, tf, ref)

    walk(ts["root"], transform, ts["root"].get("refine", refine))


# ---------------------------------------------------------------- tiles

_CTYPES = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}


def _int(v, what: str) -> int:
    """A non-negative integer from a tile's glTF JSON (anything else: ValueError)."""
    if not isinstance(v, int) or isinstance(v, bool) or v < 0:
        raise ValueError(f"bad {what}: {v!r}")
    return v


_Y_UP = np.array([[1.0, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])   # glTF y-up -> 3D Tiles z-up


def _glb_parts(glb: bytes, rtc=None, tf=None):
    """Meshes of a glb: [(points (n, 3) in tile coordinates (z up; through the tile transform
    ``tf`` if given), feature id per point, faces (m, 3) uint32)]. A malformed glb raises
    ValueError (or KeyError / IndexError / TypeError for JSON of the wrong shape)."""
    from .. import native
    if glb[:4] != b"glTF":
        raise ValueError("not a glb")
    jl, = struct.unpack_from("<I", glb, 12)
    gj = json.loads(glb[20:20 + jl])
    o = 20 + jl
    o += (4 - o % 4) % 4
    binc = memoryview(b"")
    if o + 8 <= len(glb):
        bl, = struct.unpack_from("<I", glb, o)
        binc = memoryview(glb)[o + 8:o + 8 + bl]           # no copy
    rtc = (gj.get("extensions") or {}).get("CESIUM_RTC", {}).get("center", rtc)
    post = _Y_UP
    if rtc is not None:
        t = np.eye(4)
        t[:3, 3] = np.asarray(rtc, dtype=np.float64)
        post = t @ post
    if tf is not None:
        post = tf @ post
    views = gj.get("bufferViews", [])
    accs = gj.get("accessors", [])

    def view(i):
        v = views[_int(i, "bufferView index")]
        s, n = _int(v.get("byteOffset", 0), "byteOffset"), _int(v.get("byteLength"), "byteLength")
        if s + n > len(binc):
            raise ValueError(f"bufferView {i} ({s}+{n}) runs past the {len(binc)}-byte buffer")
        return binc[s:s + n]                        # a memoryview: no copy

    def accessor(i, comps):
        a = accs[_int(i, "accessor index")]
        dt = _CTYPES.get(a.get("componentType"))
        if dt is None:
            raise ValueError(f"accessor {i}: component type {a.get('componentType')!r}")
        if a.get("bufferView") is None:
            raise ValueError(f"accessor {i} has no bufferView")
        data = view(a["bufferView"])
        item = np.dtype(dt).itemsize * comps
        stride = _int(views[a["bufferView"]].get("byteStride") or item, "byteStride")
        count, off = _int(a.get("count"), "count"), _int(a.get("byteOffset", 0), "byteOffset")
        if stride < item:
            raise ValueError(f"accessor {i}: stride {stride} under its {item}-byte elements")
        if count and off + (count - 1) * stride + item > len(data):
            raise ValueError(f"accessor {i} ({count} x {stride} from {off}) runs past its {len(data)}-byte bufferView")
        raw = np.frombuffer(data, np.uint8)[off:]
        rows = np.lib.stride_tricks.as_strided(raw, (count, item), (stride, 1), writeable=False)
        return np.ascontiguousarray(rows).view(dt).reshape(count, comps).astype(np.float64)

    meshes = gj.get("meshes", [])
    nodes = gj.get("nodes", [])
    out = []

    def node_matrix(n):
        if "matrix" in n:
            return np.asarray(n["matrix"], dtype=np.float64).reshape(4, 4).T
        m = np.eye(4)
        if "scale" in n:
            m = np.diag([*n["scale"], 1.0]) @ m
        if "rotation" in n:
            x, y, z, w = n["rotation"]
            r = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                          [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                          [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
            rm = np.eye(4)
            rm[:3, :3] = r
            m = rm @ m
        if "translation" in n:
            t = np.eye(4)
            t[:3, 3] = n["translation"]
            m = t @ m
        return m

    def visit(i, parent, depth=0):
        if depth > 32:
            raise ValueError("glTF node tree too deep (a cycle?)")
        n = nodes[_int(i, "node index")]
        m = parent @ node_matrix(n)
        if "mesh" in n:
            for prim in meshes[n["mesh"]].get("primitives", []):
                if prim.get("mode", 4) != 4:
                    continue
                attrs = prim.get("attributes", {})
                fid_name = next((k for k in ("_BATCHID", "_FEATURE_ID_0", "_FEATURE_ID") if k in attrs), None)
                dr = (prim.get("extensions") or {}).get("KHR_draco_mesh_compression")
                if dr:
                    ids = dr.get("attributes", {})
                    want = [ids["POSITION"]] + ([ids[fid_name]] if fid_name in ids else [])
                    faces, arrs = native.ext.decode_draco(bytes(view(dr["bufferView"])), want)
                    pos = arrs[0]
                    fid = arrs[1][:, 0] if len(arrs) > 1 and arrs[1].shape[1] else np.zeros(len(pos))
                else:
                    pos = accessor(attrs["POSITION"], 3)
                    fid = accessor(attrs[fid_name], 1)[:, 0] if fid_name else np.zeros(len(pos))
                    if "indices" in prim:
                        faces = accessor(prim["indices"], 1)[:, 0].astype(np.int64).reshape(-1, 3)
                    else:
                        faces = np.arange(len(pos) - len(pos) % 3).reshape(-1, 3)
                if len(fid) != len(pos):
                    raise ValueError(f"{len(fid)} feature ids for {len(pos)} points")
                if faces.size and (faces.min() < 0 or faces.max() >= len(pos)):
                    raise ValueError("triangle index out of range")
                tm = post @ m                       # one affine map from glTF to tile coordinates
                p = pos @ tm[:3, :3].T
                p += tm[:3, 3]
                out.append((p, fid.astype(np.int64), np.ascontiguousarray(faces, dtype=np.uint32)))
        for k in n.get("children", []):
            visit(k, m, depth + 1)

    scene = gj.get("scenes", [{}])[gj.get("scene", 0)] if gj.get("scenes") else {"nodes": list(range(len(nodes)))}
    for i in scene.get("nodes", []):
        visit(i, np.eye(4))
    return out


def _layout(head: bytes) -> tuple[int, int, int, int]:
    """(feature table JSON offset, its length, glb offset, glb length) of a tile from its first
    28 bytes. A b3dm's batch table (per-building attributes, often most of the file: flood ranks,
    addresses, ...) lies between the feature table and the glb and is never read."""
    if head[:4] == b"b3dm":
        total, ftj, ftb, btj, btb = struct.unpack_from("<IIIII", head, 8)
        glb = 28 + ftj + ftb + btj + btb
        return 28, ftj, glb, total - glb
    if head[:4] == b"glTF":
        total, = struct.unpack_from("<I", head, 8)
        return 0, 0, 0, total
    raise ValueError(f"unsupported 3D Tiles content {head[:4]!r}")


def _tile_parts(data: bytes):
    """Meshes of a whole tile (b3dm or glb)."""
    fo, fn, go, gn = _layout(data[:28])
    ft = json.loads(data[fo:fo + fn]) if fn else {}
    return _glb_parts(data[go:go + gn], ft.get("RTC_CENTER"))


def _fill_cracks(rel: np.ndarray, bid: np.ndarray, max_cells: int) -> None:
    """Fill (in place) grid vertices inside a roof that no triangle covers: a roof's polygons
    leave hairline cracks between them, and a vertex on one printed as a pit down to the ground.
    Enclosed gaps up to ``max_cells`` take their neighbours' highest value; courtyards stay."""
    from scipy import ndimage
    fin = np.isfinite(rel)
    holes = ndimage.binary_fill_holes(fin) & ~fin
    if not holes.any():
        return
    lab, n = ndimage.label(holes)
    size = np.bincount(lab.ravel(), minlength=n + 1)
    small = (size <= max_cells)
    small[0] = False
    todo = small[lab]
    for _ in range(4):
        if not todo.any():
            break
        v = ndimage.maximum_filter(np.where(fin, rel, -np.inf), size=3)
        i = ndimage.maximum_filter(np.where(fin, bid, 0), size=3)
        take = todo & np.isfinite(v)
        rel[take] = v[take]
        bid[take] = i[take]
        fin |= take
        todo &= ~take


# ---------------------------------------------------------------- coordinates

def _quad(u: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    """The non-constant quadratic monomials of (n, 3) points: (n, 9) x y z xx yy zz xy xz yz."""
    q = np.empty((len(u), 9)) if out is None else out[:len(u)]
    q[:, :3] = u
    x, y, z = q[:, 0], q[:, 1], q[:, 2]
    np.multiply(x, x, out=q[:, 3])
    np.multiply(y, y, out=q[:, 4])
    np.multiply(z, z, out=q[:, 5])
    np.multiply(x, y, out=q[:, 6])
    np.multiply(x, z, out=q[:, 7])
    np.multiply(y, z, out=q[:, 8])
    return q


_g = np.array([-1.0, 0.0, 1.0])
_FIT = np.stack(np.meshgrid(_g, _g, _g, indexing="ij"), -1).reshape(-1, 3)     # 27 nodes of a box
_CHECK = np.stack(np.meshgrid(*[[-0.5, 0.5]] * 3, indexing="ij"), -1).reshape(-1, 3)   # between them
_A_FIT = np.column_stack([np.ones(len(_FIT)), _quad(_FIT)])
_A_CHECK = np.column_stack([np.ones(len(_CHECK)), _quad(_CHECK)])
_PINV = np.linalg.pinv(_A_FIT)                 # least squares on the fit nodes: coefficients = _PINV @ values


class _ToGrid:
    """Earth-centred (EPSG:4978) points -> (column, row) on the vertex grid and ellipsoidal height.

    pyproj per point costs a quarter of a large fetch; over a tile's extent (a few hundred
    metres) the map is smooth, so a quadratic fitted to 27 exactly transformed points of the
    tile's bounding box stands in for it. Its error is checked at 8 more points between the fit
    nodes; a tile where it exceeds ``MAX_FIT_ERR`` is transformed exactly. On Tokyo's tiles (up
    to ~500 m across) the largest error over all 2.7 M points is ~1e-7 m, horizontally and
    vertically."""

    CHUNK = 1 << 15

    def __init__(self, aoi, grid):
        from pyproj import Transformer
        self.to_geo = Transformer.from_crs(4978, 4979, always_xy=True)
        self.to_aoi = Transformer.from_crs(4326, aoi.crs, always_xy=True)
        self.x0, self.y0, self.step = grid.x0, grid.y0, grid.step
        self.fitted = self.exact_parts = 0

    def exact(self, p: np.ndarray) -> np.ndarray:
        """(n, 3): column, row, ellipsoidal height (pyproj per point)."""
        lon, lat, h = self.to_geo.transform(p[:, 0], p[:, 1], p[:, 2])
        x, y = self.to_aoi.transform(lon, lat)
        return np.column_stack([(np.asarray(x) - self.x0) / self.step, (np.asarray(y) - self.y0) / self.step,
                                np.asarray(h, dtype=np.float64)])

    def fit(self, p: np.ndarray):
        """(centre, half-size, (10, 3) coefficients) of the quadratic over ``p``'s bounding box,
        and its largest error (m) at the check points."""
        lo = np.array([p[:, i].min() for i in range(3)])     # per column: much faster than min(0)
        hi = np.array([p[:, i].max() for i in range(3)])
        c, s = (lo + hi) / 2, np.maximum((hi - lo) / 2, 1.0)
        want = self.exact(np.vstack([_FIT, _CHECK]) * s + c)
        coef = _PINV @ want[:len(_FIT)]
        err = np.abs(_A_CHECK @ coef - want[len(_FIT):]) * [self.step, self.step, 1.0]
        return c, s, coef, float(err.max()) if np.isfinite(err).all() else math.inf

    def __call__(self, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(grid coordinates (n, 2), ellipsoidal heights (n,)) of ECEF points (n, 3)."""
        if len(p) > len(_FIT) + len(_CHECK):
            c, s, coef, err = self.fit(p)
            if err <= MAX_FIT_ERR:
                self.fitted += 1
                n = len(p)
                xy, h = np.empty((n, 2)), np.empty(n)
                buf = np.empty((min(n, self.CHUNK), 9))
                for i in range(0, n, self.CHUNK):
                    q = _quad((p[i:i + self.CHUNK] - c) / s, buf)
                    np.matmul(q, coef[1:, :2], out=xy[i:i + len(q)])
                    np.matmul(q, coef[1:, 2], out=h[i:i + len(q)])
                xy += coef[0, :2]
                h += coef[0, 2]
                return xy, h
        self.exact_parts += 1
        g = self.exact(p)
        return np.ascontiguousarray(g[:, :2]), g[:, 2].copy()


# ---------------------------------------------------------------- fetch

def fetch(aoi, grid, cancel=None, progress=lambda f, m: None) -> dict | None:
    """Roof heights above each building's base on the vertex grid:
    ``{"rel": float32 (NaN off buildings), "id": int32 building (0: none), "cities": [...],
    "tiles": n, "credit": str}``, None without PLATEAU data here; ``TooLarge`` when the area is
    too large."""
    from .. import native
    from ..httpcache import RANGE_CACHE, hint

    if not native.enabled() or not hasattr(native.ext, "decode_draco"):
        raise RuntimeError("the native extension (maptomesh_native) with Draco decoding is needed")
    bbox = aoi.lonlat_bbox(margin=grid.step * 2)
    cities = cities_for(bbox)
    if not cities:
        return None
    from . import Cancelled, TooLarge
    if aoi.width_m * aoi.height_m / 1e6 > MAX_KM2:
        raise TooLarge(f"the area is over {MAX_KM2:.0f} km2")
    progress(0.05, f"PLATEAU: {len(cities)} city model(s)")
    tiles: list = []
    used = []
    for c in cities:
        n0 = len(tiles)
        _leaves(c["url"], bbox, tiles)
        if len(tiles) > n0:
            used.append(c)
    if not tiles:
        return None
    if len(tiles) > MAX_TILES:
        raise TooLarge(f"the area needs {len(tiles)} tiles, over {MAX_TILES}")
    shape = grid.vshape
    rel = np.full(shape, np.nan, np.float32)
    bid = np.zeros(shape, np.int32)
    to_grid = _ToGrid(aoi, grid)
    next_id = 1
    failed: list[str] = []
    # 1. every tile's header: where its geometry is, and how much there is
    for u, _ in tiles:
        hint(u, 0, 28)
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        heads = list(ex.map(lambda t: RANGE_CACHE.read(t[0], 0, 28), tiles))
    plan = []
    for (u, tf), head in zip(tiles, heads):
        try:
            plan.append((u, tf, *_layout(head)))
        except (ValueError, struct.error) as e:
            failed.append(f"{u.rsplit('/', 1)[-1]}: {e}")
    mb = sum(gn + fn for _, _, _, fn, _, gn in plan) / 1e6
    if mb > MAX_MB:
        raise TooLarge(f"its 3D tiles hold {mb:.0f} MB of geometry, over {MAX_MB:.0f} MB")
    progress(0.1, f"PLATEAU: {len(plan)} tiles, {mb:.0f} MB")

    # 2. the geometry (feature table + glb) only, a window of tiles at a time: tiles are
    # downloaded and decoded ahead (threads on the server, announced downloads in the browser)
    # while earlier ones are rasterised, and each is dropped once it is (a whole city block's
    # decoded points would otherwise all be held at once)
    def get(item):
        u, tf, fo, fn, go, gn = item
        try:
            ft = json.loads(RANGE_CACHE.read(u, fo, fn)) if fn else {}
            return _glb_parts(RANGE_CACHE.read(u, go, gn), ft.get("RTC_CENTER") if isinstance(ft, dict) else None, tf)
        except (KeyError, IndexError, TypeError, AttributeError, OverflowError, struct.error) as e:
            raise ValueError(f"{type(e).__name__}: {e}") from e
        except BaseException as e:
            if type(e).__name__ == "PanicException":     # a Rust panic decoding a damaged tile
                raise ValueError(f"decoder panic: {e}") from None
            raise

    ends = np.cumsum([gn for *_, gn in plan]) / 1e6 if plan else np.zeros(0)
    hinted = 0

    def hint_ahead(k: int) -> None:      # tile k and what follows, up to two windows / HINT_MB
        nonlocal hinted
        while hinted < len(plan) and (hinted <= k or (hinted < k + 2 * WINDOW
                                                      and ends[hinted] - ends[k] < HINT_MB)):
            u, _, _, _, go, gn = plan[hinted]
            hint(u, go, gn)
            hinted += 1

    todo = iter(plan)
    window: deque = deque()
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        try:
            for k in range(len(plan)):       # in plan order: the same building ids every run
                if cancel is not None and cancel.is_set():
                    raise Cancelled()
                hint_ahead(k)
                while len(window) < WINDOW and (item := next(todo, None)) is not None:
                    window.append((item, ex.submit(get, item)))
                item, fut = window.popleft()
                try:
                    parts = fut.result()
                except ValueError as e:   # one bad tile: skip it
                    failed.append(f"{item[0].rsplit('/', 1)[-1]}: {e}")
                    parts = []
                del fut
                parts.reverse()
                while parts:                 # each mesh dropped once rasterised
                    p, fid, faces = parts.pop()
                    if not len(faces):
                        continue
                    xy, h = to_grid(p)
                    del p
                    # each building relative to its own lowest point (ellipsoidal heights -> above its base)
                    u, inv = np.unique(fid, return_inverse=True)
                    base = np.full(len(u), np.inf)
                    np.minimum.at(base, inv, h)
                    z = (h - base[inv]).astype(np.float32)
                    ids = (next_id + inv).astype(np.int32)
                    next_id += len(u)
                    native.ext.raster_max(xy, z, faces, np.ascontiguousarray(ids[faces[:, 0]]), rel, bid)
                progress(0.1 + 0.9 * (k + 1) / len(plan), f"PLATEAU 3D buildings: tile {k + 1}/{len(plan)}")
        except BaseException:
            for _, f in window:              # don't download / decode what nobody will read
                f.cancel()
            raise
    # a vertex only the building's floor covers (through a crack in the roof) is no roof
    floor = rel < 0.5
    rel[floor] = np.nan
    bid[floor] = 0
    if not np.isfinite(rel).any():
        return None
    _fill_cracks(rel, bid, max(4, int(round(25.0 / grid.step ** 2))))
    names = sorted({f"{c['name']} (FY{c['year']}, {c['lod']})" for c in used})
    return {"rel": rel, "id": bid, "cities": names, "tiles": len(tiles), "failed": failed,
            "credit": CREDIT.format(cities=", ".join(names))}
