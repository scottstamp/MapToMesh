"""GeoTIFF / Cloud-Optimized GeoTIFF reading without GDAL (the browser build has no usable GDAL;
see docs/plans/browser-wasm.md).

Covers what the elevation sources publish: classic TIFF and BigTIFF in either byte order, tiled or
stripped, chunky or planar bands, 8-64 bit integer / float samples, overviews (reduced-resolution
IFDs; masks are skipped), georeferencing (ModelPixelScale + ModelTiepoint or ModelTransformation,
PixelIsPoint), the CRS from GeoKeys (EPSG code, or a WKT / ESRI PE string in the citations) and
GDAL_NODATA. Only the tiles under a window are read: through ``httpcache.RANGE_CACHE`` for URLs,
prefetched as one batch.

Tiles are decoded by the Rust extension (``maptomesh_native.decode_tile``: LZW, Deflate, ZSTD,
PackBits, predictors 2 and 3). Without it: Deflate via zlib, LZW and PackBits in Python (slow),
ZSTD with the ``zstandard`` module if installed. LERC and JPEG are not supported.
"""

from __future__ import annotations

import math
import struct
import zlib

import numpy as np

# TIFF field types: struct code, size
_TYPES = {1: ("B", 1), 2: ("s", 1), 3: ("H", 2), 4: ("I", 4), 5: ("I", 8), 6: ("b", 1), 7: ("B", 1),
          8: ("h", 2), 9: ("i", 4), 10: ("i", 8), 11: ("f", 4), 12: ("d", 8), 13: ("I", 4),
          16: ("Q", 8), 17: ("q", 8), 18: ("Q", 8)}
_DTYPES = {(1, 1): "u1", (1, 2): "u2", (1, 4): "u4", (1, 8): "u8", (2, 1): "i1", (2, 2): "i2",
           (2, 4): "i4", (2, 8): "i8", (3, 4): "f4", (3, 8): "f8"}
COMPRESSION_NAMES = {1: "none", 5: "LZW", 7: "JPEG", 8: "Deflate", 32946: "Deflate", 32773: "PackBits",
                     34887: "LERC", 50000: "ZSTD", 50001: "WebP"}
SUPPORTED = {1, 5, 8, 32946, 32773, 50000}


class BytesSource:
    """A whole file in memory (a service response)."""

    def __init__(self, data: bytes):
        self.data = data

    def read(self, off: int, n: int) -> bytes:
        return self.data[off:off + n]

    def prefetch(self, ranges) -> None:
        pass


class UrlSource:
    """A remote file read in byte ranges through the shared range cache."""

    def __init__(self, url: str):
        from .httpcache import RANGE_CACHE
        self.url = url
        self.cache = RANGE_CACHE

    def read(self, off: int, n: int) -> bytes:
        return self.cache.read(self.url, off, n)

    def prefetch(self, ranges) -> None:
        self.cache.prefetch(self.url, list(ranges))


class _Lazy:
    """A tag value too large to read up front (tile offsets / byte counts of a big image)."""

    def __init__(self, tif: GeoTiff, typ: int, count: int, offset: int):
        self.tif, self.typ, self.count, self.offset = tif, typ, count, offset
        self.code, self.size = _TYPES[typ]

    def slice(self, a: int, b: int) -> np.ndarray:
        a, b = max(0, a), min(self.count, b)
        if b <= a:
            return np.zeros(0, np.int64)
        raw = self.tif._read(self.offset + a * self.size, (b - a) * self.size)
        dt = np.dtype(self.tif.bo + {"H": "u2", "I": "u4", "Q": "u8"}[self.code])
        return np.frombuffer(raw, dt).astype(np.int64)


class Image:
    """One full-resolution image or overview (an IFD)."""

    def __init__(self, tif: GeoTiff, tags: dict):
        g = lambda t, d=None: tags.get(t, d)  # noqa: E731
        first = lambda t, d=None: (g(t)[0] if isinstance(g(t), (list, tuple)) else g(t)) if t in tags else d  # noqa: E731
        self.width, self.height = int(first(256)), int(first(257))
        bps = g(258, [1])
        self.bytes = int((bps[0] if isinstance(bps, (list, tuple)) else bps)) // 8
        self.spp = int(first(277, 1))
        fmt = g(339, [1])
        self.fmt = int(fmt[0] if isinstance(fmt, (list, tuple)) else fmt)
        self.compression = int(first(259, 1))
        self.predictor = int(first(317, 1))
        self.planar = int(first(284, 1))
        self.subfile = int(first(254, 0))
        if 322 in tags:
            self.tw, self.th = int(first(322)), int(first(323))
            self.offsets, self.counts = tags[324], tags[325]
            self.strips = False
        else:
            self.tw = self.width
            self.th = int(first(278, self.height))
            self.offsets, self.counts = tags[273], tags[279]
            self.strips = True
        self.across = -(-self.width // self.tw)
        self.down = -(-self.height // self.th)
        self.lerc = g(50674)
        key = (self.fmt, self.bytes)
        if key not in _DTYPES:
            raise ValueError(f"unsupported sample format {self.fmt} / {8 * self.bytes} bits")
        self.dtype = np.dtype(tif.bo + _DTYPES[key])

    def _entries(self, arr, a: int, b: int) -> np.ndarray:
        if isinstance(arr, _Lazy):
            return arr.slice(a, b)
        arr = np.atleast_1d(np.asarray(arr, np.int64))
        return arr[a:b]


class GeoTiff:
    def __init__(self, source):
        self.src = source
        self._head = source.read(0, 1 << 16)
        bo = self._head[:2]
        if bo == b"II":
            self.bo = "<"
        elif bo == b"MM":
            self.bo = ">"
        else:
            raise ValueError("not a TIFF file")
        magic = struct.unpack(self.bo + "H", self._head[2:4])[0]
        self.big = magic == 43
        if magic == 43:
            first = struct.unpack(self.bo + "Q", self._head[8:16])[0]
        elif magic == 42:
            first = struct.unpack(self.bo + "I", self._head[4:8])[0]
        else:
            raise ValueError("not a TIFF file")
        ifds = []
        off, seen = first, set()
        while off and off not in seen and len(ifds) < 64:
            seen.add(off)
            tags, off = self._ifd(off)
            ifds.append(tags)
        if not ifds:
            raise ValueError("TIFF without images")
        main = ifds[0]
        self.images = [Image(self, main)]
        for t in ifds[1:]:
            sub = t.get(254, 0)
            sub = sub[0] if isinstance(sub, (list, tuple)) else sub
            if int(sub) & 4:                 # transparency mask
                continue
            if int(sub) & 1:                 # reduced resolution: an overview
                self.images.append(Image(self, t))
        self.images[1:] = sorted(self.images[1:], key=lambda im: -im.width)
        self.transform0 = self._transform(main)
        self.crs = self._crs(main)
        nd = main.get(42113)
        self.nodata = None
        if nd is not None:
            try:
                self.nodata = float(str(nd).strip("\x00 ").strip())
            except ValueError:
                pass

    # ------------------------------------------------------------ low level
    def _read(self, off: int, n: int) -> bytes:
        if off + n <= len(self._head):
            return self._head[off:off + n]
        return self.src.read(off, n)

    def _ifd(self, off: int) -> tuple[dict, int]:
        bo = self.bo
        if self.big:
            n = struct.unpack(bo + "Q", self._read(off, 8))[0]
            esize, cnt_fmt, val_size, p = 20, "Q", 8, off + 8
        else:
            n = struct.unpack(bo + "H", self._read(off, 2))[0]
            esize, cnt_fmt, val_size, p = 12, "I", 4, off + 2
        raw = self._read(p, n * esize + val_size)
        tags = {}
        for i in range(n):
            e = raw[i * esize:(i + 1) * esize]
            tag, typ = struct.unpack(bo + "HH", e[:4])
            count = struct.unpack(bo + cnt_fmt, e[4:4 + val_size])[0]
            if typ not in _TYPES:
                continue
            code, size = _TYPES[typ]
            total = count * size
            field = e[4 + val_size:]
            if total <= val_size:
                data = field[:total]
            else:
                voff = struct.unpack(bo + cnt_fmt, field)[0]
                if tag in (273, 279, 324, 325) and count > 64:
                    tags[tag] = _Lazy(self, typ, count, voff)
                    continue
                data = self._read(voff, total)
            tags[tag] = self._decode_value(typ, count, data)
        nxt = struct.unpack(bo + cnt_fmt, raw[n * esize:n * esize + val_size])[0]
        return tags, nxt

    def _decode_value(self, typ: int, count: int, data: bytes):
        code, size = _TYPES[typ]
        if typ == 2:
            return data.rstrip(b"\x00").decode("latin-1")
        if typ in (5, 10):
            v = struct.unpack(self.bo + code * (2 * count), data)
            return [v[2 * i] / v[2 * i + 1] if v[2 * i + 1] else 0.0 for i in range(count)]
        if typ == 7:
            return data
        return list(struct.unpack(self.bo + code * count, data))

    # ------------------------------------------------------------ georeferencing
    def _transform(self, t: dict):
        from affine import Affine
        if 34264 in t:
            m = t[34264]
            tf = Affine(m[0], m[1], m[3], m[4], m[5], m[7])
        elif 33550 in t and 33922 in t:
            sx, sy = t[33550][0], t[33550][1]
            i, j, _, x, y, _ = t[33922][:6]
            tf = Affine(sx, 0.0, x - i * sx, 0.0, -sy, y + j * sy)
        else:
            tf = Affine.identity()
        keys = self._geokeys(t)
        if keys.get(1025) == 2:              # PixelIsPoint: GDAL shifts to the pixel corner
            tf = tf * Affine.translation(-0.5, -0.5)
        return tf

    def _geokeys(self, t: dict) -> dict:
        d = t.get(34735)
        if not d or len(d) < 4:
            return {}
        out = {}
        doubles, ascii_ = t.get(34736) or [], t.get(34737) or ""
        for k in range(d[3]):
            key, loc, count, val = d[4 + 4 * k:8 + 4 * k]
            if loc == 0:
                out[key] = val
            elif loc == 34737:
                out[key] = ascii_[val:val + count].rstrip("|\x00")
            elif loc == 34736:
                out[key] = doubles[val:val + count]
        return out

    def _crs(self, t: dict):
        """EPSG code (int), WKT (str) or None."""
        keys = self._geokeys(t)
        for key in (3072, 2048):             # ProjectedCSType, GeographicType
            v = keys.get(key)
            if isinstance(v, int) and 0 < v < 32767:
                return v
        for key in (3073, 1026, 2049):       # citations: GDAL / ESRI put WKT there for custom CRSs
            v = keys.get(key)
            if isinstance(v, str):
                s = v.split("ESRI PE String = ", 1)[-1]
                if s.lstrip().startswith(("PROJCS[", "GEOGCS[", "PROJCRS[", "GEOGCRS[", "COMPOUNDCRS[")):
                    return s
        return None

    # ------------------------------------------------------------ public
    @property
    def factors(self) -> list[int]:
        """Overview decimation factors (like rasterio's ``overviews()``)."""
        w = self.images[0].width
        return [max(1, round(w / im.width)) for im in self.images[1:]]

    def shape(self, level: int = 0) -> tuple[int, int]:
        im = self.images[level]
        return im.height, im.width

    def transform(self, level: int = 0):
        from affine import Affine
        base, im = self.images[0], self.images[level]
        return self.transform0 * Affine.scale(base.width / im.width, base.height / im.height)

    def read(self, level: int = 0, window: tuple[int, int, int, int] | None = None, band: int = 1,
             nodata: float | None = None) -> np.ndarray:
        """Pixels of ``window`` = (row0, row1, col0, col1) at ``level`` as float32, NaN where nodata."""
        im = self.images[level]
        r0, r1, c0, c1 = window or (0, im.height, 0, im.width)
        r0, c0 = max(0, r0), max(0, c0)
        r1, c1 = min(im.height, r1), min(im.width, c1)
        out = np.full((max(0, r1 - r0), max(0, c1 - c0)), np.nan, np.float32)
        if out.size == 0:
            return out
        if im.compression not in SUPPORTED:
            raise ValueError(f"unsupported TIFF compression {COMPRESSION_NAMES.get(im.compression, im.compression)}")
        if not 1 <= band <= im.spp:
            raise ValueError(f"band {band} not in 1..{im.spp}")
        nd = self.nodata if nodata is None else nodata
        ty0, ty1 = r0 // im.th, (r1 - 1) // im.th + 1
        tx0, tx1 = c0 // im.tw, (c1 - 1) // im.tw + 1
        per_band = im.across * im.down
        boff = (band - 1) * per_band if im.planar == 2 else 0
        jobs = []
        for ty in range(ty0, ty1):
            a = boff + ty * im.across + tx0
            offs = im._entries(im.offsets, a, a + (tx1 - tx0))
            cnts = im._entries(im.counts, a, a + (tx1 - tx0))
            for k, tx in enumerate(range(tx0, tx1)):
                jobs.append((ty, tx, int(offs[k]), int(cnts[k])))
        self.src.prefetch([(o, n) for _, _, o, n in jobs if n > 0])
        spp = im.spp if im.planar == 1 else 1
        bi = band - 1 if im.planar == 1 else 0
        for ty, tx, off, n in jobs:
            rows = min(im.th, im.height - ty * im.th) if im.strips else im.th
            if n <= 0:
                continue                      # sparse tile: nodata
            raw = decode_tile(self.src.read(off, n), im.compression, im.predictor, im.tw, rows,
                              im.bytes, spp, self.bo == ">")
            dt = im.dtype.newbyteorder("<") if im.predictor == 3 else im.dtype
            tile = np.frombuffer(raw, dt).reshape(rows, im.tw, spp)[:, :, bi]
            y0, x0 = ty * im.th, tx * im.tw
            ys0, ys1 = max(r0, y0), min(r1, y0 + rows)
            xs0, xs1 = max(c0, x0), min(c1, x0 + im.tw)
            if ys1 <= ys0 or xs1 <= xs0:
                continue
            out[ys0 - r0:ys1 - r0, xs0 - c0:xs1 - c0] = tile[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0]
        if nd is not None and math.isfinite(nd):
            out[out == np.float32(nd)] = np.nan
        return out


# ---------------------------------------------------------------- tile decoding

def decode_tile(data: bytes, compression: int, predictor: int, width: int, rows: int, nbytes: int,
                spp: int = 1, big_endian: bool = False) -> bytes:
    from . import native
    if native.ext is not None and hasattr(native.ext, "decode_tile"):
        return native.ext.decode_tile(data, compression, predictor, width, rows, nbytes, spp, big_endian)
    return _decode_tile_py(data, compression, predictor, width, rows, nbytes, spp, big_endian)


def _decode_tile_py(data, compression, predictor, width, rows, nbytes, spp, big_endian) -> bytes:
    expected = width * rows * nbytes * spp
    if compression == 1:
        raw = bytes(data)
    elif compression in (8, 32946):
        raw = zlib.decompress(data)
    elif compression == 5:
        raw = _lzw(data, expected)
    elif compression == 32773:
        raw = _packbits(data, expected)
    elif compression == 50000:
        import zstandard   # not in the default install: the Rust extension covers ZSTD
        raw = zstandard.ZstdDecompressor().decompress(data, max_output_size=expected)
    else:
        raise ValueError(f"unsupported TIFF compression {compression}")
    raw = raw[:expected].ljust(expected, b"\x00")
    if predictor == 1:
        return raw
    if predictor == 2:
        bo = ">" if big_endian else "<"
        a = np.frombuffer(raw, np.dtype(bo + f"u{nbytes}")).reshape(rows, width, spp)
        return np.cumsum(a, axis=1, dtype=a.dtype).tobytes()
    if predictor == 3:
        b = np.frombuffer(raw, np.uint8).reshape(rows, width * nbytes, spp)
        b = np.cumsum(b, axis=1, dtype=np.uint8).reshape(rows, nbytes, width * spp)
        return np.ascontiguousarray(b.transpose(0, 2, 1)[:, :, ::-1]).tobytes()
    raise ValueError(f"unsupported TIFF predictor {predictor}")


def _lzw(data: bytes, expected: int) -> bytes:
    """TIFF LZW (most significant bit first, 9-12 bit codes, early change)."""
    out = bytearray()
    table = [bytes([i]) for i in range(256)] + [b"", b""]
    bits, nbits, pos, prev = 0, 0, 0, None
    width = 9
    n = len(data)
    while len(out) < expected:
        while nbits < width and pos < n:
            bits = (bits << 8) | data[pos]
            pos += 1
            nbits += 8
        if nbits < width:
            break
        code = (bits >> (nbits - width)) & ((1 << width) - 1)
        nbits -= width
        if code == 256:                       # clear
            table = table[:258]
            width = 9
            prev = None
            continue
        if code == 257:                       # end of information
            break
        if prev is None:
            entry = table[code]
        elif code < len(table):
            entry = table[code]
            table.append(prev + entry[:1])
        else:
            entry = prev + prev[:1]
            table.append(entry)
        out += entry
        prev = entry
        if len(table) + 1 >= (1 << width) and width < 12:
            width += 1
    return bytes(out)


def _packbits(data: bytes, expected: int) -> bytes:
    out = bytearray()
    i = 0
    while i < len(data) and len(out) < expected:
        n = data[i] - 256 if data[i] > 127 else data[i]
        i += 1
        if n >= 0:
            out += data[i:i + n + 1]
            i += n + 1
        elif n != -128:
            out += bytes([data[i]]) * (1 - n)
            i += 1
    return bytes(out)
