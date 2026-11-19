"""OpenStreetMap water / roads / buildings from OpenMapTiles vector tiles.

Uses OpenFreeMap (free, no key, CDN-served; OpenMapTiles schema) instead of
the Overpass API, which is frequently overloaded. Any OpenMapTiles-compatible
TileJSON can be used via ``MAPTOMESH_VECTOR_TILES``.
Tiles are cached on disk; a tiny Mapbox Vector Tile decoder is included so no
protobuf dependency is required.
"""

from __future__ import annotations

import concurrent.futures as cf
import gzip
import json
import math
import os
import time

import numpy as np
import requests

from . import config
from .geo import AOI, Grid
from .raster import fill_polygon, rasterize_lines

TILEJSON = os.environ.get("MAPTOMESH_VECTOR_TILES", "https://tiles.openfreemap.org/planet")
TILE_DIR = config.OSM_CACHE_DIR / "tiles"

# carriageway widths in metres (OpenMapTiles transportation classes)
ROAD_WIDTH = {"motorway": 22, "trunk": 18, "primary": 14, "secondary": 12, "tertiary": 10,
              "minor": 7, "service": 4, "track": 3, "path": 2, "raceway": 12, "busway": 7}
ROAD_LEVELS = {
    "major": {"motorway", "trunk", "primary", "secondary"},
    "medium": {"motorway", "trunk", "primary", "secondary", "tertiary", "minor", "raceway", "busway"},
    "all": set(ROAD_WIDTH),
}
# bridges (brunnel=bridge) of any road class plus railways, at least this wide (decks are wider
# than the carriageway; footbridges get a printable width from min_width_mm)
BRIDGE_WIDTH = {**ROAD_WIDTH, "rail": 10, "transit": 8, "path": 4}
WATERWAY_WIDTH = {"river": 25, "canal": 12, "stream": 3, "drain": 1.5, "ditch": 1.5}
STREAM_LEVELS = {"none": set(), "rivers": {"river", "canal"}, "all": {"river", "canal", "stream"}}
SKIP_WATER = {"swimming_pool"}

_tile_url: tuple[float, str] | None = None


# ---------------------------------------------------------------- MVT decoding

def _varint(b: bytes, i: int) -> tuple[int, int]:
    r = s = 0
    while True:
        c = b[i]; i += 1
        r |= (c & 0x7F) << s
        if c < 0x80:
            return r, i
        s += 7


def _fields(b: bytes):
    i, n = 0, len(b)
    while i < n:
        key, i = _varint(b, i)
        f, wt = key >> 3, key & 7
        if wt == 0:
            v, i = _varint(b, i)
        elif wt == 2:
            ln, i = _varint(b, i)
            v = b[i:i + ln]; i += ln
        elif wt == 1:
            v = b[i:i + 8]; i += 8
        elif wt == 5:
            v = b[i:i + 4]; i += 4
        else:
            raise ValueError("bad wire type")
        yield f, wt, v


def _packed(b: bytes) -> list[int]:
    out, i, n = [], 0, len(b)
    while i < n:
        v, i = _varint(b, i)
        out.append(v)
    return out


def _value(b: bytes):
    for f, wt, v in _fields(b):
        if f == 1:
            return v.decode("utf-8", "replace")
        if f == 2:
            return float(np.frombuffer(v, "<f4")[0])
        if f == 3:
            return float(np.frombuffer(v, "<f8")[0])
        if f in (4, 5):
            return v
        if f == 6:
            return (v >> 1) ^ -(v & 1)
        if f == 7:
            return bool(v)
    return None


def _geometry(cmds: list[int]) -> list[list[tuple[int, int]]]:
    parts, cur = [], []
    x = y = i = 0
    n = len(cmds)
    while i < n:
        c = cmds[i]; i += 1
        cid, cnt = c & 7, c >> 3
        if cid == 1:  # MoveTo
            for _ in range(cnt):
                if cur:
                    parts.append(cur)
                dx, dy = cmds[i], cmds[i + 1]; i += 2
                x += (dx >> 1) ^ -(dx & 1); y += (dy >> 1) ^ -(dy & 1)
                cur = [(x, y)]
        elif cid == 2:  # LineTo
            for _ in range(cnt):
                dx, dy = cmds[i], cmds[i + 1]; i += 2
                x += (dx >> 1) ^ -(dx & 1); y += (dy >> 1) ^ -(dy & 1)
                cur.append((x, y))
        elif cid == 7 and cur:  # ClosePath
            cur.append(cur[0])
    if cur:
        parts.append(cur)
    return parts


def decode_tile(data: bytes, z: int, tx: int, ty: int, layers: set[str]) -> list[dict]:
    """Decode wanted layers to features with lon/lat part arrays."""
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    feats = []
    for f, wt, lb in _fields(data):
        if f != 3 or wt != 2:
            continue
        name, extent, keys, vals, raw = None, 4096, [], [], []
        for lf, lwt, lv in _fields(lb):
            if lf == 1:
                name = lv.decode()
                if name not in layers:
                    break
            elif lf == 2:
                raw.append(lv)
            elif lf == 3:
                keys.append(lv.decode())
            elif lf == 4:
                vals.append(_value(lv))
            elif lf == 5:
                extent = lv
        if name not in layers:
            continue
        scale = 1.0 / (extent * (1 << z))
        for fb in raw:
            tags, gtype, geom = [], 0, []
            for ff, fwt, fv in _fields(fb):
                if ff == 2:
                    tags = _packed(fv)
                elif ff == 3:
                    gtype = fv
                elif ff == 4:
                    geom = _packed(fv)
            if gtype not in (2, 3):
                continue
            props = {keys[tags[k]]: vals[tags[k + 1]] for k in range(0, len(tags) - 1, 2)}
            parts = []
            for p in _geometry(geom):
                a = np.asarray(p, dtype=np.float64)
                X = tx / (1 << z) + a[:, 0] * scale  # normalised web-mercator
                Y = ty / (1 << z) + a[:, 1] * scale
                lon = X * 360.0 - 180.0
                lat = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * Y))))
                parts.append(np.column_stack([lon, lat]))
            feats.append({"layer": name, "type": "polygon" if gtype == 3 else "line", "props": props, "parts": parts})
    return feats


# ---------------------------------------------------------------- fetching

def _tiles_url() -> str:
    global _tile_url
    if _tile_url and time.time() - _tile_url[0] < 6 * 3600:
        return _tile_url[1]
    r = requests.get(TILEJSON, headers={"User-Agent": config.USER_AGENT}, timeout=30)
    r.raise_for_status()
    _tile_url = (time.time(), r.json()["tiles"][0])
    return _tile_url[1]


def _tile_range(bbox, z: int):
    n = 1 << z
    def tx(lon): return int((lon + 180.0) / 360.0 * n)
    def ty(lat): return int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n)
    x0, x1 = tx(bbox[0]), tx(bbox[2])
    y0, y1 = ty(bbox[3]), ty(bbox[1])
    return range(max(x0, 0), min(x1, n - 1) + 1), range(max(y0, 0), min(y1, n - 1) + 1)


def _get_tile(url_tmpl: str, z: int, x: int, y: int) -> bytes:
    path = TILE_DIR / str(z) / str(x) / f"{y}.pbf"
    if path.exists() and time.time() - path.stat().st_mtime < 30 * 86400:
        return path.read_bytes()
    url = url_tmpl.replace("{z}", str(z)).replace("{x}", str(x)).replace("{y}", str(y))
    for attempt in range(3):
        try:
            r = requests.get(url, headers={"User-Agent": config.USER_AGENT}, timeout=30)
            if r.status_code == 404 or r.status_code == 204:
                data = b""
            else:
                r.raise_for_status()
                data = r.content
            config.atomic_write(path, data)
            return data
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(1 + attempt)
    return b""


def fetch_osm(aoi: AOI, grid: Grid, roads: str, water: bool, buildings: bool = False,
              progress=None) -> dict:
    layers = set()
    if water:
        layers |= {"water", "waterway"}
    if roads and roads != "none":
        layers.add("transportation")
    if buildings:
        layers.add("building")
    if not layers:
        return {"features": [], "tiles": 0}
    bbox = aoi.lonlat_bbox(margin=grid.step * 4)
    # pick the most detailed zoom keeping the request modest
    z = 14
    while z > 9:
        xs, ys = _tile_range(bbox, z)
        if len(xs) * len(ys) <= 64:
            break
        z -= 1
    xs, ys = _tile_range(bbox, z)
    url = _tiles_url()
    coords = [(x, y) for x in xs for y in ys]
    feats: list[dict] = []
    done = 0
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(_get_tile, url, z, x, y): (x, y) for x, y in coords}
        for fut in cf.as_completed(futs):
            x, y = futs[fut]
            data = fut.result()
            if data:
                feats += decode_tile(data, z, x, y, layers)
            done += 1
            if progress:
                progress(done / len(coords), f"OpenStreetMap tiles {done}/{len(coords)} (z{z})")
    return {"features": feats, "tiles": len(coords), "zoom": z}


# ---------------------------------------------------------------- rasterising

def rasterize_osm(data: dict, aoi: AOI, grid: Grid, mm_per_m: float, min_width_mm: float = 0.6,
                  road_scale: float = 1.0, road_level: str = "medium", stream_level: str = "all",
                  river_min_width_mm: float = 0.8) -> dict[str, np.ndarray]:
    """Return {'water' (areas), 'waterways' (lines), 'roads', 'bridges', 'buildings'} vertex-grid
    masks. ``bridges`` are road / rail / foot bridges whatever ``road_level`` is (the structure
    matters even when roads aren't shown); pass ``road_level="none"`` to get only those."""
    from pyproj import Transformer

    shape = grid.vshape
    to_local = Transformer.from_crs("EPSG:4326", aoi.crs, always_xy=True)

    def conv(a: np.ndarray) -> np.ndarray:
        x, y = to_local.transform(a[:, 0], a[:, 1])
        return np.column_stack([(x - grid.x0) / grid.step, (y - grid.y0) / grid.step])

    water = np.zeros(shape, bool)
    bld = np.zeros(shape, bool)
    road_lines, water_lines, bridge_lines = [], [], []
    min_hw = (min_width_mm / mm_per_m) / grid.step / 2 if mm_per_m > 0 else 0.5
    wanted_roads = ROAD_LEVELS.get(road_level, set())
    for f in data.get("features", []):
        layer, props = f["layer"], f["props"]
        parts = [conv(p) for p in f["parts"] if len(p) >= 2]
        if not parts:
            continue
        if layer == "water" and f["type"] == "polygon":
            if props.get("class") in SKIP_WATER or props.get("brunnel") == "tunnel":
                continue
            water |= fill_polygon(parts, shape)
        elif layer == "waterway" and f["type"] == "line":
            # rivers / streams as continuous centrelines, at least a printable width wide
            if props.get("class") not in STREAM_LEVELS.get(stream_level, set()):
                continue
            if props.get("brunnel") in ("tunnel", "culvert") or props.get("intermittent") in (1, True, "1"):
                continue
            wm = WATERWAY_WIDTH.get(props.get("class"), 3)
            min_river_hw = (river_min_width_mm / mm_per_m) / grid.step / 2 if mm_per_m > 0 else 1.0
            for p in parts:
                water_lines.append((p, max(wm / grid.step / 2, min_river_hw)))
        elif layer == "transportation" and f["type"] == "line":
            cls = props.get("class")
            if props.get("brunnel") == "bridge" and cls in BRIDGE_WIDTH:
                bw = BRIDGE_WIDTH[cls] * road_scale
                for p in parts:
                    bridge_lines.append((p, max(bw / grid.step / 2, min_hw)))
            if cls not in wanted_roads or props.get("brunnel") == "tunnel":
                continue
            wm = ROAD_WIDTH.get(cls, 6) * road_scale * (0.6 if props.get("ramp") else 1.0)
            for p in parts:
                road_lines.append((p, max(wm / grid.step / 2, min_hw)))
        elif layer == "building" and f["type"] == "polygon":
            bld |= fill_polygon(parts, shape)
    waterways = rasterize_lines(water_lines, shape)
    roads = rasterize_lines(road_lines, shape)
    bridges = rasterize_lines(bridge_lines, shape)
    return {"water": water, "waterways": waterways, "roads": roads, "bridges": bridges, "buildings": bld}
