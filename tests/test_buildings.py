"""Offline checks for the building sources (maptomesh.buildings) and how they merge with the
OpenStreetMap footprints where the elevation data is bare earth.

    python tests/test_buildings.py
"""

from __future__ import annotations

import json
import math
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from maptomesh import native  # noqa: E402
from maptomesh.buildings import plateau  # noqa: E402
from maptomesh.model import _bare_earth_buildings  # noqa: E402


def _glb(positions: np.ndarray, faces: np.ndarray, batch: np.ndarray, translation=None, rtc=None) -> bytes:
    """An uncompressed glb with one triangle primitive (POSITION, _BATCHID, indices)."""
    pos = positions.astype("<f4").tobytes()
    bid = batch.astype("<f4").tobytes()
    idx = faces.astype("<u4").tobytes()
    binc = pos + bid + idx
    gj = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(binc)}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(pos)},
                        {"buffer": 0, "byteOffset": len(pos), "byteLength": len(bid)},
                        {"buffer": 0, "byteOffset": len(pos) + len(bid), "byteLength": len(idx)}],
        "accessors": [{"bufferView": 0, "componentType": 5126, "count": len(positions), "type": "VEC3"},
                      {"bufferView": 1, "componentType": 5126, "count": len(batch), "type": "SCALAR"},
                      {"bufferView": 2, "componentType": 5125, "count": faces.size, "type": "SCALAR"}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "_BATCHID": 1}, "indices": 2}]}],
        "nodes": [{"mesh": 0, **({"translation": translation} if translation else {})}],
        "scenes": [{"nodes": [0]}],
    }
    if rtc is not None:
        gj["extensions"] = {"CESIUM_RTC": {"center": rtc}}
    js = json.dumps(gj).encode()
    js += b" " * ((4 - len(js) % 4) % 4)
    binc += b"\0" * ((4 - len(binc) % 4) % 4)
    body = struct.pack("<I4s", len(js), b"JSON") + js + struct.pack("<I4s", len(binc), b"BIN\0") + binc
    return struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body


def test_glb_axes_and_rtc():
    """glTF is y-up: a point (x, y, z) is at (x, -z, y) in tile coordinates, plus the RTC centre
    and the node translation (applied before the axis change)."""
    pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    g = _glb(pts, np.array([[0, 1, 2]]), np.array([0, 0, 1]), translation=[10.0, 20.0, 30.0], rtc=[100.0, 200.0, 300.0])
    (p, fid, faces), = plateau._glb_parts(g)
    q = pts + [10.0, 20.0, 30.0]
    assert np.allclose(p, np.column_stack([q[:, 0], -q[:, 2], q[:, 1]]) + [100.0, 200.0, 300.0])
    assert fid.tolist() == [0, 0, 1] and faces.tolist() == [[0, 1, 2]]
    # b3dm: header, feature table (RTC_CENTER), batch table, then the glb
    ft = json.dumps({"BATCH_LENGTH": 2, "RTC_CENTER": [1.0, 2.0, 3.0]}).encode()
    ft += b" " * ((8 - (28 + len(ft)) % 8) % 8)
    inner = _glb(pts, np.array([[0, 1, 2]]), np.array([0, 0, 1]))
    b3dm = struct.pack("<4sIIIIII", b"b3dm", 1, 28 + len(ft) + len(inner), len(ft), 0, 0, 0) + ft + inner
    (p2, _, _), = plateau._tile_parts(b3dm)
    assert np.allclose(p2, np.column_stack([pts[:, 0], -pts[:, 2], pts[:, 1]]) + [1.0, 2.0, 3.0])


def _malformed_glbs() -> dict[str, bytes]:
    """Damaged versions of a small uncompressed glb."""
    pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    good = _glb(pts, np.array([[0, 1, 2]]), np.array([0, 0, 1]))
    plateau._glb_parts(good)

    def edit(fn) -> bytes:
        jl, = struct.unpack_from("<I", good, 12)
        gj = json.loads(good[20:20 + jl])
        fn(gj)
        js = json.dumps(gj).encode()
        js += b" " * ((4 - len(js) % 4) % 4)
        rest = good[20 + jl:]
        body = struct.pack("<I4s", len(js), b"JSON") + js + rest
        return struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body

    def count(gj):
        gj["accessors"][0]["count"] = 10 ** 6

    def offset(gj):
        gj["accessors"][0]["byteOffset"] = 1 << 20

    def stride(gj):
        gj["bufferViews"][0]["byteStride"] = 4

    def view(gj):
        gj["bufferViews"][0]["byteLength"] = 1 << 20

    def neg(gj):
        gj["accessors"][0]["count"] = -1

    def ctype(gj):
        gj["accessors"][0]["componentType"] = 9999

    def ids(gj):
        gj["accessors"][1]["count"] = 2

    def cycle(gj):
        gj["nodes"][0]["children"] = [0]

    bad = {"truncated": good[:len(good) - 20], "truncated more": good[:90]}
    bad.update({fn.__name__: edit(fn) for fn in (count, offset, stride, view, neg, ctype, ids, cycle)})
    oob = _glb(pts, np.array([[0, 1, 7]]), np.array([0, 0, 1]))       # an index past the points
    bad["index"] = oob
    return bad


def test_glb_malformed():
    """Damaged uncompressed glbs raise ValueError (the fetch skips the tile) instead of reading
    outside their buffer: truncated files, accessors / bufferViews past the end, bad strides,
    indices past the points, mismatched feature ids, node cycles."""
    for name, g in _malformed_glbs().items():
        try:
            plateau._glb_parts(g)
        except ValueError:
            continue
        raise AssertionError(f"{name}: no error")


def test_to_grid():
    """The fitted local map matches pyproj per point to well under 1 cm over a tile-sized box,
    and falls back to pyproj where it can't (a box far larger than any tile)."""
    from pyproj import Transformer

    from maptomesh.geo import AOI, Grid
    aoi = AOI(35.6812, 139.7671, 1500.0, 1500.0)
    g = Grid.for_aoi(aoi, 600)
    to_ecef = Transformer.from_crs(4979, 4978, always_xy=True)
    rng = np.random.default_rng(1)
    for km, fitted in ((0.5, True), (2.0, True), (40.0, False)):
        tg = plateau._ToGrid(aoi, g)
        d = km / 111.0
        lon = 139.7671 + rng.uniform(-d / 2, d / 2, 5000)
        lat = 35.6812 + rng.uniform(-d / 2, d / 2, 5000)
        p = np.column_stack(to_ecef.transform(lon, lat, rng.uniform(35.0, 400.0, 5000)))
        xy, h = tg(p)
        e = tg.exact(p)
        assert (tg.fitted == 1) == fitted, (km, tg.fitted)
        assert np.hypot(*(xy - e[:, :2]).T).max() * g.step < 0.01 and np.abs(h - e[:, 2]).max() < 0.01, km
        assert xy.flags.c_contiguous and xy.shape == (5000, 2)


def _building_glb(aoi, x: float, y: float, height: float, size: float = 40.0) -> bytes:
    """A glb (CESIUM_RTC) with one box building centred at local (x, y): a flat roof ``height``
    above its base, and walls (edge-on: no area) down to the base."""
    from pyproj import Transformer
    to_ecef = Transformer.from_crs(4979, 4978, always_xy=True)
    hx = np.array([-1, 1, 1, -1]) * size / 2 + x
    hy = np.array([-1, -1, 1, 1]) * size / 2 + y
    lon, lat = aoi.to_lonlat.transform(np.r_[hx, hx], np.r_[hy, hy])
    ecef = np.column_stack(to_ecef.transform(lon, lat, np.r_[np.full(4, 40.0 + height), np.full(4, 40.0)]))
    rtc = ecef.mean(0)
    d = ecef - rtc
    gl = np.column_stack([d[:, 0], d[:, 2], -d[:, 1]])          # 3D Tiles z-up -> glTF y-up
    faces = np.array([[0, 1, 2], [0, 2, 3], [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5]])
    return _glb(gl, faces, np.zeros(8), rtc=rtc.tolist())


def _fake_fetch(aoi, grid, files: dict, window: int):
    """plateau.fetch over in-memory tiles (no index, no network)."""
    from maptomesh import httpcache

    class Cache:
        def read(self, url, start, n, session=None):
            return files[url][start:start + n]

    saved = (plateau.cities_for, plateau._leaves, httpcache.RANGE_CACHE, plateau.WINDOW)
    plateau.cities_for = lambda bbox: [{"code": "13101", "name": "test-ku", "year": "2025", "url": "t", "lod": "LoD2"}]
    plateau._leaves = lambda url, bbox, out: out.extend((u, None) for u in files)
    httpcache.RANGE_CACHE = Cache()
    plateau.WINDOW = window
    try:
        return plateau.fetch(aoi, grid)
    finally:
        plateau.cities_for, plateau._leaves, httpcache.RANGE_CACHE, plateau.WINDOW = saved


def test_fetch_windows():
    """The windowed fetch: buildings at their measured height, a damaged tile skipped and
    reported, and the same building ids whatever the window."""
    if not native.enabled() or not hasattr(native.ext, "decode_draco"):
        print("  (native extension not built: skipped)")
        return
    from maptomesh.geo import AOI, Grid
    aoi = AOI(35.6812, 139.7671, 400.0, 400.0)
    g = Grid.for_aoi(aoi, 200)
    spots = [(-120.0, -100.0, 30.0), (0.0, 0.0, 80.0), (110.0, 90.0, 12.0), (-100.0, 120.0, 45.0)]
    files = {f"https://example.org/t{k}.glb": _building_glb(aoi, x, y, h) for k, (x, y, h) in enumerate(spots)}
    whole = files["https://example.org/t1.glb"]
    files["https://example.org/bad.glb"] = whole[:len(whole) // 2] + bytes(len(whole) - len(whole) // 2)
    a = _fake_fetch(aoi, g, files, 1)
    b = _fake_fetch(aoi, g, files, 8)
    assert np.array_equal(a["id"], b["id"]) and np.array_equal(a["rel"], b["rel"], equal_nan=True)
    assert len(a["failed"]) == 1 and a["failed"][0].startswith("bad.glb"), a["failed"]
    for k, (x, y, h) in enumerate(spots):
        c, r = int(round((x - g.x0) / g.step)), int(round((y - g.y0) / g.step))
        assert abs(a["rel"][r, c] - h) < 0.01, (k, a["rel"][r, c], h)
    assert len(np.unique(a["id"][a["id"] > 0])) == len(spots)


def test_json_cache():
    """Indexes / tilesets: a plain GET kept on disk for its TTL, refreshed after it, and the
    stale copy used when the refresh fails."""
    import tempfile
    import time

    import requests

    from maptomesh import config
    calls = []

    class Resp:
        def __init__(self, body):
            self.content = body

        def raise_for_status(self):
            pass

    def get(url, **kw):
        calls.append(url)
        if fail:
            raise requests.ConnectionError("offline")
        return Resp(json.dumps({"v": len(calls)}).encode())

    saved = (config.CACHE_DIR, requests.get)
    with tempfile.TemporaryDirectory() as d:
        config.CACHE_DIR = Path(d)
        requests.get = get
        try:
            fail = False
            url = "https://api.example.org/datacatalog/tileset.json"
            assert plateau._get_json(url, 3600) == {"v": 1}
            assert plateau._get_json(url, 3600) == {"v": 1} and len(calls) == 1     # from disk
            time.sleep(0.05)
            assert plateau._get_json(url, 0.01) == {"v": 2} and len(calls) == 2     # expired: refreshed
            fail = True
            time.sleep(0.05)
            assert plateau._get_json(url, 0.01) == {"v": 2} and len(calls) == 3     # offline: stale copy
            try:
                plateau._get_json(url + "?other", 3600)
            except requests.ConnectionError:
                pass
            else:
                raise AssertionError("no copy and no network: must raise")
        finally:
            config.CACHE_DIR, requests.get = saved


def test_cancel():
    """A set cancel event always propagates (as ``Cancelled``); an ordinary error that happens to
    say "Cancelled" is just a warning."""
    import threading
    from types import SimpleNamespace

    from maptomesh import buildings
    from maptomesh.buildings import overture
    opts = SimpleNamespace(show_buildings=True)
    elev = SimpleNamespace(arrays={"veg_known": np.zeros((50, 50), bool)})
    grid = SimpleNamespace(step=10.0)

    def boom(*a, **k):
        raise RuntimeError("Cancelled order for the upstream service")

    saved = (plateau.fetch, overture.fetch)
    plateau.fetch = boom
    overture.fetch = lambda *a, **k: None
    try:
        out, _, warnings = buildings.fetch_buildings(opts, elev, None, grid, cancel=threading.Event())
        assert "plateau" not in out and len(warnings) == 1, warnings
        ev = threading.Event()
        ev.set()
        try:
            buildings.fetch_buildings(opts, elev, None, grid, cancel=ev)
        except buildings.Cancelled:
            pass
        else:
            raise AssertionError("cancellation swallowed")

        def cancelled(*a, **k):
            raise buildings.Cancelled()
        plateau.fetch = cancelled
        try:
            buildings.fetch_buildings(opts, elev, None, grid, cancel=None)
        except buildings.Cancelled:
            pass
        else:
            raise AssertionError("Cancelled swallowed")
    finally:
        plateau.fetch, overture.fetch = saved


def test_raster_max():
    if not native.enabled():
        print("  (native extension not built: skipped)")
        return
    oz = np.full((6, 6), np.nan, np.float32)
    oi = np.zeros((6, 6), np.int32)
    xy = np.array([[0, 0], [5, 0], [0, 5], [5, 5], [1, 1], [4, 1], [1, 4]], float)
    z = np.array([2, 2, 2, 2, 9, 9, 9], np.float32)
    tris = np.array([[0, 1, 2], [1, 3, 2], [4, 5, 6], [0, 1, 1]], np.uint32)   # a roof, a tower, a degenerate one
    native.ext.raster_max(xy, z, tris, np.array([1, 1, 2, 3], np.int32), oz, oi)
    assert np.isfinite(oz[:6, :6]).all()
    assert oz[1, 1] == 9 and oi[1, 1] == 2 and oz[5, 5] == 2 and oi[5, 5] == 1
    assert (oi != 3).all()


def test_fill_cracks():
    rel = np.full((12, 12), 20.0, np.float32)
    rel[:, :2] = np.nan
    rel[5, 6] = np.nan                         # a crack in the roof
    rel[8:11, 3:6] = np.nan                    # an enclosed courtyard (9 cells)
    bid = np.where(np.isfinite(rel), 7, 0).astype(np.int32)
    plateau._fill_cracks(rel, bid, 4)
    assert rel[5, 6] == 20.0 and bid[5, 6] == 7
    assert np.isnan(rel[9, 4]) and np.isnan(rel[3, 0])


def test_leaves():
    """REPLACE: only leaves; ADD: every node on the way; regions outside the bbox are skipped."""
    r = math.radians

    def node(w, s, e, n, uri=None, kids=(), refine=None):
        d = {"boundingVolume": {"region": [r(w), r(s), r(e), r(n), 0, 100]}, "geometricError": 0}
        if uri:
            d["content"] = {"uri": uri}
        if kids:
            d["children"] = list(kids)
        if refine:
            d["refine"] = refine
        return d

    ts = {"root": node(139.0, 35.0, 140.0, 36.0, "root.b3dm", refine="REPLACE", kids=[
        node(139.0, 35.0, 139.5, 35.5, "a.b3dm", kids=[node(139.0, 35.0, 139.2, 35.2, "a1.b3dm"),
                                                       node(139.3, 35.3, 139.5, 35.5, "a2.b3dm")]),
        node(139.5, 35.5, 140.0, 36.0, "b.b3dm", refine="ADD", kids=[node(139.6, 35.6, 139.7, 35.7, "b1.b3dm")]),
    ])}
    saved = plateau._get_json
    plateau._get_json = lambda url, ttl=0: ts
    plateau._memo.clear()
    try:
        out = []
        plateau._leaves("https://example.org/city/tileset.json", (139.1, 35.1, 139.65, 35.65), out)
    finally:
        plateau._get_json = saved
        plateau._memo.clear()
    names = sorted(u.rsplit("/", 1)[1] for u, _ in out)
    assert names == ["a1.b3dm", "a2.b3dm", "b.b3dm", "b1.b3dm"], names


def test_merge():
    """Bare-earth buildings: Overture fills OSM's default (5 m) heights but not tagged ones and adds
    footprints OSM lacks; a PLATEAU building replaces the OSM footprint it overlaps and stands on
    its lowest ground."""
    shape = (40, 40)
    G = np.tile(np.linspace(0.0, 3.9, 40), (40, 1))          # ground rising 0.1 m per column
    osm_b = np.zeros(shape, bool)
    osm_h = np.full(shape, np.nan)
    osm_b[2:6, 2:6] = True; osm_h[2:6, 2:6] = 5.0           # untagged (default height)
    osm_b[2:6, 10:14] = True; osm_h[2:6, 10:14] = 20.0      # tagged
    osm_b[20:26, 20:26] = True; osm_h[20:26, 20:26] = 5.0   # also in PLATEAU
    ov = {"h": np.full(shape, np.nan), "mask": np.zeros(shape, bool), "other": np.zeros(shape, bool)}
    ov["h"][2:6, 2:6] = 12.0; ov["h"][2:6, 10:14] = 12.0
    ov["mask"][2:6, 2:14] = True
    ov["other"][10:14, 2:6] = True; ov["h"][10:14, 2:6] = 8.0; ov["mask"][10:14, 2:6] = True
    rel = np.full(shape, np.nan, np.float32)
    ids = np.zeros(shape, np.int32)
    rel[20:27, 20:27] = 30.0; ids[20:27, 20:27] = 4
    osm = {"buildings": osm_b, "building_h": osm_h, "overture": ov,
           "plateau": {"rel": rel, "id": ids, "cities": ["test-ku (FY2025, LoD2)"]}}
    notes = []
    bare = np.ones(shape, bool)
    out = _bare_earth_buildings(np.full(shape, np.nan), G, bare, None, osm, 10.0, notes)
    assert np.allclose(out[2:6, 2:6], G[2:6, 2:6] + 12.0)        # filled from Overture
    assert np.allclose(out[2:6, 10:14], G[2:6, 10:14] + 20.0)    # OSM's tagged height kept
    assert np.allclose(out[10:14, 2:6], G[10:14, 2:6] + 8.0)     # Overture-only footprint
    assert np.allclose(out[20:27, 20:27], G[20, 20] + 30.0)      # PLATEAU, on its lowest ground
    assert len(notes) == 3, notes
    # LiDAR / surface-model cells are left alone
    out2 = _bare_earth_buildings(np.full(shape, np.nan), G, np.zeros(shape, bool), None, osm, 10.0, [])
    assert np.isnan(out2).all()


if __name__ == "__main__":
    for fn in (test_glb_axes_and_rtc, test_glb_malformed, test_to_grid, test_fetch_windows, test_json_cache,
               test_cancel, test_raster_max, test_fill_cracks, test_leaves, test_merge):
        fn()
        print(f"{fn.__name__}: ok")
    print("all passed")
