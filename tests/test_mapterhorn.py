"""Offline checks for the Mapterhorn reader pieces and the provider ranking.

    python tests/test_mapterhorn.py

PMTiles tile ids against the spec's reference values, a directory and tile lookup in a small
in-memory archive (root + leaf directory), the MVT polygon decoder on a hand-encoded tile, the
effective-resolution ranking of elevation providers, and the blend-band erosion.
"""

from __future__ import annotations

import gzip
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from maptomesh import dem, httpcache, mvt, pmtiles  # noqa: E402


def varint(v: int) -> bytes:
    out = bytearray()
    while True:
        b = v & 0x7F
        v >>= 7
        out.append(b | (0x80 if v else 0))
        if not v:
            return bytes(out)


def directory(entries: list[tuple[int, int, int, int]]) -> bytes:
    """entries: (tile_id, run_length, offset, length), sorted by tile id."""
    out = varint(len(entries))
    last = 0
    for tid, _, _, _ in entries:
        out += varint(tid - last)
        last = tid
    out += b"".join(varint(r) for _, r, _, _ in entries)
    out += b"".join(varint(n) for _, _, _, n in entries)
    prev_end = None
    for _, _, off, n in entries:
        out += varint(0 if prev_end == off else off + 1)
        prev_end = off + n
    return out


def test_tileid() -> None:
    t = pmtiles.zxy_to_tileid
    assert [t(0, 0, 0), t(1, 0, 0), t(1, 0, 1), t(1, 1, 1), t(1, 1, 0), t(2, 0, 0)] == [0, 1, 2, 3, 4, 5]
    assert t(12, 3423, 1763) == 19078479
    ids = sorted(t(3, x, y) for x in range(8) for y in range(8))
    assert ids == list(range(t(3, 0, 0), t(3, 0, 0) + 64)), "zoom 3 ids must be a permutation"


class FakeCache:
    def __init__(self, blob: bytes):
        self.blob, self.reads = blob, 0

    def read(self, key, start, n):
        self.reads += 1
        return self.blob[start:start + n]

    def prefetch(self, key, ranges, workers=8):
        pass


def test_archive() -> None:
    tiles = {(2, x, y): f"tile {x},{y}".encode() for x in range(4) for y in range(4)}
    order = sorted(tiles, key=lambda k: pmtiles.zxy_to_tileid(*k))
    data, entries, off = b"", [], 0
    for k in order:
        body = gzip.compress(tiles[k])
        entries.append((pmtiles.zxy_to_tileid(*k), 1, off, len(body)))
        data += body
        off += len(body)
    leaf = gzip.compress(directory(entries[8:]))            # second half behind a leaf directory
    root = gzip.compress(directory(entries[:8] + [(entries[8][0], 0, 0, len(leaf))]))
    root_off = pmtiles.HEADER_LEN
    leaf_off = root_off + len(root)
    data_off = leaf_off + len(leaf)
    hdr = b"PMTiles" + bytes([3]) + struct.pack("<8Q", root_off, len(root), 0, 0, leaf_off, len(leaf),
                                                data_off, len(data))
    hdr += struct.pack("<3Q", 16, 16, 16) + bytes([1, 2, 2, 4, 2, 2]) + struct.pack("<4i", 0, 0, 0, 0)
    hdr += bytes([2]) + struct.pack("<2i", 0, 0)
    assert len(hdr) == pmtiles.HEADER_LEN, len(hdr)
    blob = hdr + root + leaf + data
    real = httpcache.RANGE_CACHE
    httpcache.RANGE_CACHE = FakeCache(blob)
    try:
        a = pmtiles.Archive("https://example.invalid/test.pmtiles", "v1")
        assert a.key.endswith("#v1")
        got = a.get_many(list(tiles) + [(3, 0, 0), (1, 0, 0)])
    finally:
        httpcache.RANGE_CACHE = real
    assert got == tiles, sorted(set(got) ^ set(tiles))


def encode_tile(features: list[tuple[dict, list[list[tuple[int, int]]]]], extent: int = 4096) -> bytes:
    def field(n, wt, payload):
        return varint((n << 3) | wt) + payload

    def lenf(n, b):
        return field(n, 2, varint(len(b)) + b)

    def zz(v):
        return (v << 1) ^ (v >> 31)

    keys, values, feats = [], [], b""
    for props, rings in features:
        tags = []
        for k, v in props.items():
            if k not in keys:
                keys.append(k)
            if v not in values:
                values.append(v)
            tags += [keys.index(k), values.index(v)]
        geom, x, y = [], 0, 0
        for r in rings:
            geom.append(1 | (1 << 3))
            geom += [zz(r[0][0] - x), zz(r[0][1] - y)]
            x, y = r[0]
            geom.append(2 | ((len(r) - 1) << 3))
            for px, py in r[1:]:
                geom += [zz(px - x), zz(py - y)]
                x, y = px, py
            geom.append(7 | (1 << 3))
        f = lenf(2, b"".join(varint(t) for t in tags)) + field(3, 0, varint(3)) \
            + lenf(4, b"".join(varint(g) for g in geom))
        feats += lenf(2, f)
    layer = field(15, 0, varint(2)) + lenf(1, b"coverage") + feats
    layer += b"".join(lenf(3, k.encode()) for k in keys)
    layer += b"".join(lenf(4, lenf(1, v.encode())) for v in values)
    layer += field(5, 0, varint(extent))
    return gzip.compress(lenf(3, layer))


def test_mvt() -> None:
    outer = [(0, 0), (100, 0), (100, 100), (0, 100)]          # clockwise with y down: exterior
    hole = [(10, 10), (10, 20), (20, 20), (20, 10)]
    other = [(200, 200), (300, 200), (300, 300)]
    tile = encode_tile([({"source": "dk"}, [outer, hole]), ({"source": "glo30"}, [other])])
    layer = mvt.decode(tile)["coverage"]
    assert layer["extent"] == 4096
    f0, f1 = layer["features"]
    assert f0["props"] == {"source": "dk"} and f1["props"] == {"source": "glo30"}
    polys = mvt.polygons(f0["rings"])
    assert len(polys) == 1 and len(polys[0]) == 2, polys
    assert polys[0][0] == outer and polys[0][1] == hole
    assert mvt.ring_area(outer) == 10000 and mvt.ring_area(hole) == -100


class P(dem.DemProvider):
    def __init__(self, key, res, has_dsm=False, dsm_only=False, prefer=1):
        self.key, self.res_m, self.has_dsm, self.dsm_only, self.prefer = key, res, has_dsm, dsm_only, prefer


def test_rank() -> None:
    off_dsm = P("official 1 m + DSM", 1.0, has_dsm=True)
    off_dtm = P("official 1 m DTM", 1.0)
    shared = P("Mapterhorn 2 m", 2.0, prefer=0)
    shared_jp = P("Mapterhorn 1 m (licence warning)", 1.0, prefer=2)
    surf = P("surface model 2 m", 2.0, dsm_only=True)
    coarse = P("official 10 m", 10.0)
    ps = [coarse, surf, shared_jp, shared, off_dtm, off_dsm]
    keys = lambda step, dsm: [p.key for p in dem.ordered(ps, step, dsm)]  # noqa: E731
    # 5 m grid, bare terrain: everything <= 2.5 m ties; shared tiles first, the licence-warning copy last
    k = keys(5.0, False)
    assert k[0] == "Mapterhorn 2 m" and set(k[1:3]) == {"official 1 m + DSM", "official 1 m DTM"} \
        and k[3] == "Mapterhorn 1 m (licence warning)", k
    # surface job: the provider with a DSM first
    assert keys(5.0, True)[:3] == ["official 1 m + DSM", "Mapterhorn 2 m", "official 1 m DTM"], keys(5.0, True)
    # 1.6 m grid: the 1 m models are finer than Mapterhorn's 2 m here
    k = keys(1.6, False)
    assert set(k[:2]) == {"official 1 m + DSM", "official 1 m DTM"} and k.index("Mapterhorn 2 m") == 3, k
    # bare earth before a surface model at equal effective resolution; coarse last
    k = keys(5.0, False)
    assert k.index("surface model 2 m") == 4 and k[-1] == "official 10 m"
    # no grid: native resolution order (as before)
    assert keys(None, True)[:2] == ["official 1 m + DSM", "official 1 m DTM"]


def test_erode() -> None:
    from maptomesh.providers.mapterhorn import _erode
    m = np.zeros((50, 50), bool)
    m[:, :30] = True                                   # covered on the west, seam at column 30
    e = _erode(m, 10.0, 1.0)
    assert e[:, :19].all() and not e[:, 21:].any()      # 10 m band at the seam removed
    assert e[0, :19].all()                              # the grid's own edges are not seams
    assert _erode(np.ones((5, 5), bool), 10.0, 1.0).all()


if __name__ == "__main__":
    for fn in (test_tileid, test_archive, test_mvt, test_rank, test_erode):
        fn()
        print(f"{fn.__name__}: ok")
    print("all passed")
