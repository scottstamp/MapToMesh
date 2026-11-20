"""Minimal Mapbox Vector Tile (MVT 2.x) decoder: polygon features and their string properties.

https://github.com/mapbox/vector-tile-spec/tree/master/2.1

Only what reading coverage footprints needs: layers, features, string / numeric tags and polygon
geometry (rings in tile pixel coordinates, 0..extent, y down).

The Rust extension decodes tiles when it is built (``maptomesh_native.mvt_decode``: the same
output); the pure-Python decoder here is the fallback.
"""

from __future__ import annotations

import gzip
import struct

from . import native


def _varint(b: bytes, p: int) -> tuple[int, int]:
    out = shift = 0
    while True:
        c = b[p]
        p += 1
        out |= (c & 0x7F) << shift
        if not c & 0x80:
            return out, p
        shift += 7


def _fields(b: bytes):
    """(field number, wire type, value) of a protobuf message; value is bytes for wire type 2."""
    p, n = 0, len(b)
    while p < n:
        key, p = _varint(b, p)
        f, wt = key >> 3, key & 7
        if wt == 0:
            v, p = _varint(b, p)
        elif wt == 1:
            v, p = b[p:p + 8], p + 8
        elif wt == 2:
            ln, p = _varint(b, p)
            v, p = b[p:p + ln], p + ln
        elif wt == 5:
            v, p = b[p:p + 4], p + 4
        else:
            raise ValueError(f"unsupported protobuf wire type {wt}")
        yield f, wt, v


def _packed(b: bytes) -> list[int]:
    out, p = [], 0
    while p < len(b):
        v, p = _varint(b, p)
        out.append(v)
    return out


def _value(b: bytes):
    for f, wt, v in _fields(b):
        if f == 1:
            return v.decode("utf-8", "replace")
        if f == 2:
            return struct.unpack("<f", v)[0]
        if f == 3:
            return struct.unpack("<d", v)[0]
        if f in (4, 5):
            return v
        if f == 6:
            return (v >> 1) ^ -(v & 1)
        if f == 7:
            return bool(v)
    return None


def _rings(cmds: list[int]) -> list[list[tuple[int, int]]]:
    rings, cur = [], []
    x = y = 0
    i = 0
    while i < len(cmds):
        cid, count = cmds[i] & 7, cmds[i] >> 3
        i += 1
        if cid == 7:                     # ClosePath
            if len(cur) >= 3:
                rings.append(cur)
            cur = []
            continue
        for _ in range(count):
            dx, dy = cmds[i], cmds[i + 1]
            i += 2
            x += (dx >> 1) ^ -(dx & 1)
            y += (dy >> 1) ^ -(dy & 1)
            if cid == 1:                 # MoveTo starts a ring
                if len(cur) >= 3:
                    rings.append(cur)
                cur = [(x, y)]
            else:
                cur.append((x, y))
    return rings


def ring_area(r: list[tuple[int, int]]) -> float:
    """Shoelace area in tile coordinates (y down): exterior rings are positive."""
    a = 0.0
    for (x0, y0), (x1, y1) in zip(r, r[1:] + r[:1]):
        a += x0 * y1 - x1 * y0
    return a / 2.0


def polygons(rings: list[list[tuple[int, int]]]) -> list[list[list[tuple[int, int]]]]:
    """Group rings into polygons: an exterior ring followed by its holes."""
    out = []
    for r in rings:
        if not out:                      # the first ring starts a polygon whatever its area
            out.append([r])
            continue
        a = ring_area(r)
        if a > 0:
            out.append([r])
        elif a < 0:
            out[-1].append(r)
    return out


def decode(data: bytes) -> dict[str, dict]:
    """{layer name: {"extent": int, "features": [{"props": {...}, "type": int, "rings": [...]}]}}"""
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    if native.enabled() and hasattr(native.ext, "mvt_decode"):
        return native.ext.mvt_decode(bytes(data))
    return _decode_py(data)


def _decode_py(data: bytes) -> dict[str, dict]:
    layers = {}
    for f, _, lb in _fields(data):
        if f != 3:
            continue
        name, extent, keys, values, feats = "", 4096, [], [], []
        for lf, _, lv in _fields(lb):
            if lf == 1:
                name = lv.decode("utf-8", "replace")
            elif lf == 2:
                feats.append(lv)
            elif lf == 3:
                keys.append(lv.decode("utf-8", "replace"))
            elif lf == 4:
                values.append(_value(lv))
            elif lf == 5:
                extent = lv
        out = []
        for fb in feats:
            tags, gtype, geom = [], 0, []
            for ff, wt, fv in _fields(fb):
                if ff == 2:
                    tags = _packed(fv) if wt == 2 else [fv]
                elif ff == 3:
                    gtype = fv
                elif ff == 4:
                    geom = _packed(fv)
            props = {keys[tags[k]]: values[tags[k + 1]] for k in range(0, len(tags) - 1, 2)}
            out.append({"props": props, "type": gtype, "rings": _rings(geom) if gtype == 3 else []})
        layers[name] = {"extent": extent, "features": out}
    return layers
