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
import json
import math
import re
import struct
import threading
import time

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
_WHOLE = 1 << 28                             # "read the whole file" (the range cache clamps it)
CREDIT = "3D city models: Project PLATEAU (MLIT Japan), {cities}, processed (PLATEAU site policy, CC BY 4.0 compatible)"

_lock = threading.Lock()
_index: dict = {}
_tilesets: dict = {}


# ---------------------------------------------------------------- discovery

def _get_json(url: str):
    from ..httpcache import RANGE_CACHE
    return json.loads(RANGE_CACHE.read(url, 0, _WHOLE).decode("utf-8"))


def _cities(lod: str, url: str) -> list[dict]:
    """[{"code", "name", "year", "url", "region"}] of a nationwide index (memoised for a day)."""
    with _lock:
        hit = _index.get(lod)
        if hit and time.time() - hit[0] < 86400:
            return hit[1]
    out = []
    for c in _get_json(url)["root"].get("children", []):
        u = (c.get("content") or {}).get("uri") or (c.get("content") or {}).get("url")
        reg = (c.get("boundingVolume") or {}).get("region")
        if not u or not reg:
            continue
        m = re.search(r"/(\d{5})_([a-z0-9-]+)_[a-z]+_(\d{4})_", u)
        out.append({"code": m.group(1) if m else u, "name": m.group(2) if m else "",
                    "year": m.group(3) if m else "", "url": u, "region": reg[:4]})
    with _lock:
        _index[lod] = (time.time(), out)
    return out


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
    with _lock:
        ts = _tilesets.get(tileset_url)
    if ts is None:
        ts = _get_json(tileset_url)
        with _lock:
            _tilesets[tileset_url] = ts

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

def _glb_parts(glb: bytes, rtc=None):
    """Meshes of a glb: [(points (n, 3) in tile coordinates (z up), feature id per point, faces)]."""
    from .. import native
    if glb[:4] != b"glTF":
        raise ValueError("not a glb")
    jl, = struct.unpack_from("<I", glb, 12)
    gj = json.loads(glb[20:20 + jl])
    o = 20 + jl
    o += (4 - o % 4) % 4
    binc = b""
    if o + 8 <= len(glb):
        bl, = struct.unpack_from("<I", glb, o)
        binc = glb[o + 8:o + 8 + bl]
    rtc = (gj.get("extensions") or {}).get("CESIUM_RTC", {}).get("center", rtc)
    views = gj.get("bufferViews", [])
    accs = gj.get("accessors", [])

    def view(i):
        v = views[i]
        s = v.get("byteOffset", 0)
        return binc[s:s + v["byteLength"]]

    def accessor(i, comps):
        a = accs[i]
        dt = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}[a["componentType"]]
        v = views[a["bufferView"]]
        stride = v.get("byteStride") or np.dtype(dt).itemsize * comps
        raw = np.frombuffer(view(a["bufferView"]), np.uint8)[a.get("byteOffset", 0):]
        rows = np.lib.stride_tricks.as_strided(raw, (a["count"], np.dtype(dt).itemsize * comps), (stride, 1))
        return np.ascontiguousarray(rows).view(dt).reshape(a["count"], comps).astype(np.float64)

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

    def visit(i, parent):
        n = nodes[i]
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
                    faces, arrs = native.ext.decode_draco(view(dr["bufferView"]), want)
                    pos = arrs[0].astype(np.float64)
                    fid = arrs[1][:, 0] if len(arrs) > 1 and arrs[1].shape[1] else np.zeros(len(pos))
                    faces = faces.astype(np.int64)
                else:
                    pos = accessor(attrs["POSITION"], 3)
                    fid = accessor(attrs[fid_name], 1)[:, 0] if fid_name else np.zeros(len(pos))
                    if "indices" in prim:
                        faces = accessor(prim["indices"], 1)[:, 0].astype(np.int64).reshape(-1, 3)
                    else:
                        faces = np.arange(len(pos) - len(pos) % 3).reshape(-1, 3)
                p = pos @ m[:3, :3].T + m[:3, 3]
                p = np.column_stack([p[:, 0], -p[:, 2], p[:, 1]])     # glTF y-up -> 3D Tiles z-up
                if rtc is not None:
                    p += np.asarray(rtc, dtype=np.float64)
                out.append((p, fid.astype(np.int64), faces))
        for k in n.get("children", []):
            visit(k, m)

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


# ---------------------------------------------------------------- fetch

def fetch(aoi, grid, cancel=None, progress=lambda f, m: None) -> dict | None:
    """Roof heights above each building's base on the vertex grid:
    ``{"rel": float32 (NaN off buildings), "id": int32 building (0: none), "cities": [...],
    "tiles": n, "credit": str}``, None without PLATEAU data here; ``TooLarge`` when the area is
    too large."""
    from pyproj import Transformer

    from .. import native
    from ..httpcache import RANGE_CACHE, hint

    if not native.enabled() or not hasattr(native.ext, "decode_draco"):
        raise RuntimeError("the native extension (maptomesh_native) with Draco decoding is needed")
    bbox = aoi.lonlat_bbox(margin=grid.step * 2)
    cities = cities_for(bbox)
    if not cities:
        return None
    from . import TooLarge
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
    to_geo = Transformer.from_crs(4978, 4979, always_xy=True)
    to_aoi = Transformer.from_crs(4326, aoi.crs, always_xy=True)
    next_id = 1
    done = 0
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
    # 2. the geometry (feature table + glb) only
    for u, _, fo, fn, go, gn in plan:
        hint(u, go, gn)

    def get(item):
        u, _, fo, fn, go, gn = item
        ft = json.loads(RANGE_CACHE.read(u, fo, fn)) if fn else {}
        return _glb_parts(RANGE_CACHE.read(u, go, gn), ft.get("RTC_CENTER"))

    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        futs = [(item, ex.submit(get, item)) for item in plan]
        for item, fut in futs:            # in plan order: the same building ids every run
            if cancel is not None and cancel.is_set():
                raise RuntimeError("Cancelled")
            url, tf = item[:2]
            try:
                parts = fut.result()
            except (ValueError, KeyError, IndexError, struct.error) as e:   # one bad tile: skip it
                failed.append(f"{url.rsplit('/', 1)[-1]}: {e}")
                parts = []
            for p, fid, faces in parts:
                if not len(faces):
                    continue
                if tf is not None:
                    p = p @ tf[:3, :3].T + tf[:3, 3]
                lon, lat, h = to_geo.transform(p[:, 0], p[:, 1], p[:, 2])
                x, y = to_aoi.transform(lon, lat)
                xy = np.column_stack([(np.asarray(x) - grid.x0) / grid.step, (np.asarray(y) - grid.y0) / grid.step])
                h = np.asarray(h, dtype=np.float64)
                # each building relative to its own lowest point (ellipsoidal heights -> above its base)
                u, inv = np.unique(fid, return_inverse=True)
                base = np.full(len(u), np.inf)
                np.minimum.at(base, inv, h)
                z = (h - base[inv]).astype(np.float32)
                ids = (next_id + inv).astype(np.int32)
                next_id += len(u)
                native.ext.raster_max(np.ascontiguousarray(xy), z, np.ascontiguousarray(faces, dtype=np.uint32),
                                      np.ascontiguousarray(ids[faces[:, 0]]), rel, bid)
            done += 1
            progress(0.1 + 0.9 * done / len(plan), f"PLATEAU 3D buildings: tile {done}/{len(plan)}")
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
