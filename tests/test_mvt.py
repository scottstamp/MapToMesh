"""The Rust vector tile decoders against the pure-Python ones.

    python tests/test_mvt.py

``maptomesh_native.mvt_decode_osm`` / ``mvt_decode`` must return exactly what
``osm._decode_tile_py`` / ``mvt._decode_py`` return (same order, same Python types, bit-identical
floats and coordinate arrays), and ``mvt.polygons`` must group rings as before: on hand-encoded
tiles covering the corner cases, on a few hundred cached OpenStreetMap tiles (``cache/osm/tiles``)
and on some Overture building tiles (fetched through the range cache). The fallback path must work without the extension, and
malformed (truncated, bit-flipped, random) tiles must raise ``ValueError``, never crash.
Parts needing the extension, cached tiles or the network are skipped when those are missing.
"""

from __future__ import annotations

import gzip
import os
import random
import struct
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from maptomesh import config, mvt, native, osm  # noqa: E402

HAVE = native.ext is not None and all(hasattr(native.ext, f) for f in ("mvt_decode_osm", "mvt_decode"))
OSM_LAYERS = {"water", "waterway", "transportation", "building"}
N_OSM = int(os.environ.get("MVT_TEST_TILES", "300"))


# ---------------------------------------------------------------- encoding

def varint(v: int) -> bytes:
    out = bytearray()
    while True:
        b = v & 0x7F
        v >>= 7
        out.append(b | (0x80 if v else 0))
        if not v:
            return bytes(out)


def field(f: int, v) -> bytes:
    if isinstance(v, int):
        return varint(f << 3) + varint(v)
    return varint(f << 3 | 2) + varint(len(v)) + v


def zz(v: int) -> int:
    return (v << 1) ^ (v >> 63)


def cmd(cid: int, n: int) -> int:
    return cid | n << 3


def feature(tags: list[int], gtype: int, geom: list[int], extra: bytes = b"") -> bytes:
    return (field(1, 7) + field(2, b"".join(map(varint, tags))) + field(3, gtype)
            + field(4, b"".join(map(varint, geom))) + extra)


def layer(name: bytes, feats: list[bytes], keys: list[bytes], values: list[bytes], extent=4096,
          name_last: bool = False) -> bytes:
    body = (b"".join(field(2, f) for f in feats) + b"".join(field(3, k) for k in keys)
            + b"".join(field(4, v) for v in values) + field(5, extent) + field(15, 2))
    return body + field(1, name) if name_last else field(1, name) + body


def sample_tile() -> bytes:
    """Every value type, odd tags, duplicate keys, all geometry types and command quirks."""
    values = [field(1, "Grachtengordel ✓".encode()), field(1, b"bad \xff\xfe utf-8 \xed\xa0\x80"),
              varint(2 << 3 | 5) + struct.pack("<f", 1.1), varint(3 << 3 | 1) + struct.pack("<d", -2.5e-300),
              field(4, 7), field(4, 2**64 - 5), field(5, 300), field(6, zz(-42)), field(6, zz(2**62)),
              field(7, 1), field(7, 0), b"", field(9, 3) + field(1, b"after unknown"),
              varint(2 << 3 | 5) + struct.pack("<f", float("nan"))]
    keys = [b"class", b"name", b"brunnel", b"render_height", b"x"]
    nv = len(values)
    line = [cmd(1, 1), zz(10), zz(20), cmd(2, 3), zz(5), zz(0), zz(0), zz(5), zz(-5), zz(-5),
            cmd(1, 2), zz(100), zz(100), zz(1), zz(1), cmd(2, 1), zz(3), zz(-300)]
    poly = [cmd(1, 1), zz(0), zz(0), cmd(2, 3), zz(4096), zz(0), zz(0), zz(4096), zz(-4096), zz(0), cmd(7, 1),
            cmd(1, 1), zz(10), zz(-4000), cmd(2, 3), zz(0), zz(100), zz(100), zz(0), zz(0), zz(-100), cmd(7, 1),
            cmd(1, 1), zz(2000), zz(2000), cmd(2, 1), zz(1), zz(1), cmd(7, 1),      # short ring
            cmd(3, 2), cmd(7, 3),                                                   # unknown id, ClosePath x3
            cmd(1, 1), zz(50), zz(50), cmd(2, 2), zz(10), zz(0), zz(0), zz(10)]      # unclosed
    lineto_first = [cmd(2, 2), zz(1), zz(2), zz(3), zz(4), cmd(7, 1), cmd(2, 0)]
    feats = [feature([0, 0, 1, 1, 2, 11], 2, line),
             feature([0, 1, 0, 2, 3, 4, 3, 5, 4], 3, poly, extra=field(9, b"ignored") + field(3, 3)),
             feature([1, 6, 2, 7, 3, 8, 4, 10], 1, [cmd(1, 1), zz(1), zz(1)]),       # point: skipped by osm
             feature([3, 11, 4, 12, 0, 13], 3, lineto_first),
             feature([], 2, []), feature([2, nv - 1], 0, [cmd(1, 1), zz(3), zz(3)]),
             feature([4, 3], 3, [cmd(1, 3), zz(0), zz(0), zz(10), zz(0), zz(0), zz(10)])]
    tile = (field(3, layer(b"water", feats, keys, values))
            + field(3, layer(b"mountain_peak", feats[:3], keys, values))
            + field(2, b"not a layer") + field(9, 12345)
            + field(3, layer(b"transportation", feats[::-1], keys, values, extent=512, name_last=True))
            + field(3, layer(b"water", feats[1:2], keys[::-1], values[::-1], extent=8192)))
    return tile


# ---------------------------------------------------------------- comparing

def same(a, b, path="") -> None:
    assert type(a) is type(b), (path, type(a), type(b))
    if isinstance(a, dict):
        assert list(a) == list(b), (path, list(a), list(b))
        for k in a:
            same(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), (path, len(a), len(b))
        for i, (x, y) in enumerate(zip(a, b)):
            same(x, y, f"{path}[{i}]")
    elif isinstance(a, np.ndarray):
        assert a.dtype == b.dtype and a.shape == b.shape and a.flags.c_contiguous == b.flags.c_contiguous, path
        assert a.tobytes() == b.tobytes(), path          # bit-identical
    elif isinstance(a, float):
        assert struct.pack("<d", a) == struct.pack("<d", b) or (a != a and b != b), (path, a, b)
    else:
        assert a == b, (path, a, b)


def osm_both(data: bytes, z: int, x: int, y: int, layers=OSM_LAYERS):
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    return osm._decode_tile_py(data, z, x, y, layers), osm._decode_tile_native(data, z, x, y, layers)


def mvt_both(data: bytes):
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    py = mvt._decode_py(data)
    nat = native.ext.mvt_decode(data)
    same(py, nat)
    for name, lay in py.items():
        for f in lay["features"]:
            same(mvt.polygons(f["rings"]), polygons_ref(f["rings"]), name)
    return py


def polygons_ref(rings):
    """``mvt.polygons`` as it was (it now skips the first ring's area, which can't matter)."""
    out = []
    for r in rings:
        a = mvt.ring_area(r)
        if a > 0 or not out:
            out.append([r])
        elif a < 0:
            out[-1].append(r)
    return out


def osm_tiles(n: int) -> list[tuple[int, int, int, bytes]]:
    root = config.OSM_CACHE_DIR / "tiles"
    files = sorted(root.glob("*/*/*.pbf"))
    files = [p for p in files if p.stat().st_size]
    if len(files) > n:
        files = files[::len(files) // n][:n]
    return [(int(p.parent.parent.name), int(p.parent.name), int(p.stem), p.read_bytes()) for p in files]


# ---------------------------------------------------------------- tests

def test_synthetic() -> None:
    if not HAVE:
        print("  (Rust extension without MVT decoding: skipped)")
        return
    tile = sample_tile()
    for data in (tile, gzip.compress(tile)):
        for layers in (OSM_LAYERS, {"water"}, {"mountain_peak", "transportation"}, set()):
            py, nat = osm_both(data, 14, 8414, 5384, layers)
            same(py, nat)
        assert len(py) == 0 and len(nat) == 0
        py = mvt_both(data)
        assert list(py) == ["water", "mountain_peak", "transportation"] and py["water"]["extent"] == 8192
        same(osm.decode_tile(data, 3, 1, 2, OSM_LAYERS), osm._decode_tile_py(tile, 3, 1, 2, OSM_LAYERS))
        same(mvt.decode(data), mvt._decode_py(tile))
    f = osm._decode_tile_py(tile, 14, 8414, 5384, OSM_LAYERS)[0]["props"]
    assert f == {"class": "Grachtengordel ✓", "name": "bad �� utf-8 ���", "brunnel": None}, f
    feats = mvt._decode_py(tile)["mountain_peak"]["features"]
    p = feats[2]["props"]
    assert p == {"name": 300, "brunnel": -42, "render_height": 2**62, "x": False}, p
    p = feats[1]["props"]
    assert p == {"class": 1.100000023841858, "render_height": 2**64 - 5}, p
    rings = feats[1]["rings"]
    assert [len(r) for r in rings] == [4, 4], "ring rules"
    # polygons(): outer ring + hole; the first ring starts a polygon even when clockwise
    assert mvt.polygons(rings) == [[rings[0], rings[1]]]
    for rr in ([rings[1], rings[0], rings[1]], [rings[0], rings[0][:2], rings[1]], [rings[1][:2]], []):
        same(mvt.polygons(rr), polygons_ref(rr))
    assert mvt.polygons([[(0.0, 0.0), (1.5, 0.0), (0.0, 1.0)], [[0, 0], [0, 1], [1, 0]]]) == \
        [[[(0.0, 0.0), (1.5, 0.0), (0.0, 1.0)], [[0, 0], [0, 1], [1, 0]]]]


def test_osm_tiles() -> None:
    if not HAVE:
        print("  (Rust extension without MVT decoding: skipped)")
        return
    tiles = osm_tiles(N_OSM)
    if not tiles:
        print(f"  (no cached tiles under {config.OSM_CACHE_DIR / 'tiles'}: skipped)")
        return
    t_py = t_nat = 0.0
    nf = npts = 0
    for i, (z, x, y, data) in enumerate(tiles):
        raw = gzip.decompress(data) if data[:2] == b"\x1f\x8b" else data
        t = time.perf_counter()
        py = osm._decode_tile_py(raw, z, x, y, OSM_LAYERS)
        t_py += time.perf_counter() - t
        t = time.perf_counter()
        nat = osm._decode_tile_native(raw, z, x, y, OSM_LAYERS)
        t_nat += time.perf_counter() - t
        same(py, nat, f"{z}/{x}/{y}")
        nf += len(py)
        npts += sum(len(p) for f in py for p in f["parts"])
        if i % 3 == 0:
            mvt_both(raw)                                # the generic decoder on the same tiles
    print(f"  {len(tiles)} tiles, {nf} features, {npts} points identical; "
          f"Python {t_py:.2f} s, Rust {t_nat:.2f} s")


def test_overture_tiles() -> None:
    if not HAVE:
        print("  (Rust extension without MVT decoding: skipped)")
        return
    try:
        from maptomesh.buildings import overture
        a, _ = overture._archive()
        got = a.get_many([(14, 14552, 6451), (14, 14553, 6451), (14, 14552, 6452),
                          (14, 4560, 5999), (14, 4561, 5999), (14, 4560, 6000)])
    except Exception as ex:                              # offline
        print(f"  (Overture tiles unavailable: {ex!r}: skipped)")
        return
    n = 0
    for key, data in got.items():
        if not data:
            continue
        py = mvt_both(data)
        same(py, mvt.decode(data), str(key))
        for f in py["building"]["features"]:
            same(mvt.polygons(f["rings"]), polygons_ref(f["rings"]), str(key))
        n += len(py["building"]["features"])
    print(f"  {len(got)} tiles, {n} buildings identical")


def test_fallback() -> None:
    """No extension, or one without the MVT functions: the Python decoders run."""
    tile = gzip.compress(sample_tile())
    saved = native.ext
    try:
        for ext in (None, type("Old", (), {})()):
            native.ext = ext
            same(osm.decode_tile(tile, 14, 8414, 5384, OSM_LAYERS),
                 osm._decode_tile_py(sample_tile(), 14, 8414, 5384, OSM_LAYERS))
            d = mvt.decode(tile)
            same(d, mvt._decode_py(sample_tile()))
            assert mvt.polygons(d["mountain_peak"]["features"][1]["rings"])
    finally:
        native.ext = saved
    os.environ["MAPTOMESH_NATIVE"] = "0"
    try:
        same(mvt.decode(tile), mvt._decode_py(sample_tile()))
    finally:
        del os.environ["MAPTOMESH_NATIVE"]


def test_malformed() -> None:
    if not HAVE:
        print("  (Rust extension without MVT decoding: skipped)")
        return
    rng = random.Random(1234)
    tiles = [sample_tile()] + [gzip.decompress(d) if d[:2] == b"\x1f\x8b" else d for *_, d in osm_tiles(8)]
    raised = runs = 0

    def run(data: bytes) -> None:
        nonlocal raised, runs
        for fn in (lambda: native.ext.mvt_decode_osm(data, OSM_LAYERS | {"mountain_peak", "poi"}),
                   lambda: native.ext.mvt_decode(data)):
            runs += 1
            try:
                fn()
            except ValueError:
                raised += 1

    for data in tiles:
        n = len(data)
        for cut in sorted({rng.randrange(1, n) for _ in range(60)}):
            run(data[:cut])
        for _ in range(150):
            b = bytearray(data)
            for _ in range(rng.choice((1, 1, 2, 8))):
                i = rng.randrange(n)
                b[i] ^= 1 << rng.randrange(8)
            run(bytes(b))
    for _ in range(300):
        run(bytes(rng.randrange(256) for _ in range(rng.randrange(1, 200))))
    def water(feats=(), keys=(b"k",), values=(field(1, b"v"),), extent=4096) -> bytes:
        return field(3, layer(b"water", list(feats), list(keys), list(values), extent))

    for bad, osm_raises, mvt_raises in (
            (b"\x1a\xff\xff\xff\xff\xff\xff\xff\xff\xff\x01", True, True),   # huge length
            (b"\x1a" + b"\xff" * 11, True, True),                             # varint overflow
            (b"\x1a\x05ab", True, True), (b"\x1a\x02\x08", True, True),       # truncated
            (b"\x1b\x00", True, True),                                        # wire type 3
            (water([feature([0, 5], 2, [9, 2, 2])]), True, True),             # tag value index
            (water([feature([0, 0], 3, [9, 2])]), True, True),                # truncated geometry
            (water(extent=0), True, False),                                   # zero extent
            (water(keys=[b"\xff"]), True, False)):                            # strict / "replace" keys
        for fn, want in ((lambda: native.ext.mvt_decode_osm(bad, OSM_LAYERS), osm_raises),
                         (lambda: native.ext.mvt_decode(bad), mvt_raises)):
            try:
                fn()
                assert not want, f"accepted {bad!r}"
            except ValueError:
                assert want, f"rejected {bad!r}"
    print(f"  {runs} decodes of damaged tiles: {raised} raised ValueError, the rest decoded, no crash")


if __name__ == "__main__":
    for fn in (test_synthetic, test_osm_tiles, test_overture_tiles, test_fallback, test_malformed):
        fn()
        print(f"{fn.__name__}: ok")
    print("all passed")
