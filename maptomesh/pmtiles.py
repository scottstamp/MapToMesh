"""Minimal reader for PMTiles v3 archives over HTTP range requests.

https://github.com/protomaps/PMTiles/blob/main/spec/v3/spec.md

Everything goes through the persistent range cache (``httpcache.RANGE_CACHE``): the header and
directories are read once, tiles are fetched in parallel with neighbouring ranges merged (tiles of
one area sit close together in an archive: they are stored in Hilbert-curve order).

An archive that is replaced upstream must not be read through stale cached directories: pass
``version`` (e.g. the file's md5 from a release manifest) and the cache keys its blocks by
``url#version``; the fragment is not sent to the server.
"""

from __future__ import annotations

import bisect
import gzip
import struct
import threading
from dataclasses import dataclass

HEADER_LEN = 127
FIRST_READ = 16384          # header + root directory are always within the first 16 KiB
COMPRESSION = {0: "unknown", 1: "none", 2: "gzip", 3: "brotli", 4: "zstd"}


def zxy_to_tileid(z: int, x: int, y: int) -> int:
    """Tile id: tiles of lower zooms first, then Hilbert order within the zoom."""
    if z > 31:
        raise ValueError("zoom too high")
    n = 1 << z
    if not (0 <= x < n and 0 <= y < n):
        raise ValueError(f"tile {z}/{x}/{y} outside the zoom")
    acc = ((1 << (2 * z)) - 1) // 3          # number of tiles in zooms 0 .. z-1
    d, s = 0, n >> 1
    while s > 0:
        rx = 1 if x & s else 0
        ry = 1 if y & s else 0
        d += s * s * ((3 * rx) ^ ry)
        if ry == 0:                          # rotate the quadrant
            if rx == 1:
                x, y = n - 1 - x, n - 1 - y
            x, y = y, x
        s >>= 1
    return acc + d


def _varint(buf: bytes, pos: int) -> tuple[int, int]:
    out = shift = 0
    while True:
        b = buf[pos]
        pos += 1
        out |= (b & 0x7F) << shift
        if not b & 0x80:
            return out, pos
        shift += 7


@dataclass
class Header:
    root_offset: int
    root_length: int
    meta_offset: int
    meta_length: int
    leaf_offset: int
    leaf_length: int
    data_offset: int
    data_length: int
    internal_compression: int
    tile_compression: int
    tile_type: int
    min_zoom: int
    max_zoom: int
    bounds: tuple[float, float, float, float]


def parse_header(b: bytes) -> Header:
    if len(b) < HEADER_LEN or b[:7] != b"PMTiles":
        raise ValueError("not a PMTiles archive")
    if b[7] != 3:
        raise ValueError(f"PMTiles version {b[7]} (only 3 is supported)")
    (ro, rl, mo, ml, lo, ll, do, dl) = struct.unpack_from("<8Q", b, 8)
    ic, tc, tt, z0, z1 = b[97], b[98], b[99], b[100], b[101]
    w, s, e, n = (v / 1e7 for v in struct.unpack_from("<4i", b, 102))
    return Header(ro, rl, mo, ml, lo, ll, do, dl, ic, tc, tt, z0, z1, (w, s, e, n))


def parse_directory(raw: bytes) -> tuple[list[int], list[int], list[int], list[int]]:
    """(tile_ids, run_lengths, lengths, offsets) of a decompressed directory."""
    n, p = _varint(raw, 0)
    ids, runs, lens, offs = [0] * n, [0] * n, [0] * n, [0] * n
    last = 0
    for i in range(n):
        v, p = _varint(raw, p)
        last += v
        ids[i] = last
    for i in range(n):
        runs[i], p = _varint(raw, p)
    for i in range(n):
        lens[i], p = _varint(raw, p)
    for i in range(n):
        v, p = _varint(raw, p)
        offs[i] = offs[i - 1] + lens[i - 1] if (v == 0 and i > 0) else v - 1
    return ids, runs, lens, offs


class Archive:
    """One remote archive. Thread-safe; directories are memoised per instance."""

    def __init__(self, url: str, version: str = ""):
        from .httpcache import RANGE_CACHE
        self.url = url
        self.key = f"{url}#{version}" if version else url   # cache identity (fragment is not sent)
        self._cache = RANGE_CACHE
        self._dirs: dict[tuple[int, int], tuple] = {}
        self._lock = threading.Lock()
        first = self._read(0, FIRST_READ)
        self.header = parse_header(first)
        if self.header.internal_compression not in (1, 2):
            raise ValueError(f"unsupported directory compression "
                             f"{COMPRESSION.get(self.header.internal_compression)}")
        h = self.header
        if h.root_offset + h.root_length <= len(first):
            raw = first[h.root_offset:h.root_offset + h.root_length]
            self._dirs[(h.root_offset, h.root_length)] = parse_directory(self._decompress(raw))

    def _read(self, start: int, n: int) -> bytes:
        return self._cache.read(self.key, start, n)

    def _decompress(self, raw: bytes) -> bytes:
        return gzip.decompress(raw) if self.header.internal_compression == 2 else raw

    def _directory(self, offset: int, length: int) -> tuple:
        d = self._dirs.get((offset, length))
        if d is None:
            d = parse_directory(self._decompress(self._read(offset, length)))
            with self._lock:
                self._dirs[(offset, length)] = d
        return d

    def locate(self, z: int, x: int, y: int) -> tuple[int, int] | None:
        """(absolute offset, length) of a tile, None if the archive has no such tile."""
        h = self.header
        if z < h.min_zoom or z > h.max_zoom:
            return None
        tid = zxy_to_tileid(z, x, y)
        offset, length = h.root_offset, h.root_length
        for _ in range(4):   # the spec allows at most 3 levels of leaf directories
            ids, runs, lens, offs = self._directory(offset, length)
            i = bisect.bisect_right(ids, tid) - 1
            if i < 0:
                return None
            if runs[i] == 0:                      # leaf directory
                offset, length = h.leaf_offset + offs[i], lens[i]
                continue
            if tid < ids[i] + runs[i]:
                return h.data_offset + offs[i], lens[i]
            return None
        raise ValueError("PMTiles directory nesting too deep")

    def get_many(self, tiles: list[tuple[int, int, int]], workers: int = 8) -> dict[tuple, bytes]:
        """Raw tile bytes for every (z, x, y) the archive has. Ranges are prefetched in parallel."""
        locs = {t: self.locate(*t) for t in tiles}
        found = {t: loc for t, loc in locs.items() if loc is not None}
        self._cache.prefetch(self.key, list(found.values()), workers=workers)
        out = {}
        for t, (off, n) in found.items():
            data = self._read(off, n)
            if self.header.tile_compression == 2:
                data = gzip.decompress(data)
            out[t] = data
        return out
