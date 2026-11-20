"""Generic readers that put remote elevation rasters onto the AOI vertex grid.

Every reader returns a float64 array of the vertex grid shape ``grid.vshape`` with row 0 at the
southern edge and NaN where the source has no data. Sources can be:

* Cloud-Optimized GeoTIFFs (any HTTP URL): only the blocks of the overview level matching the grid
  are read, through the persistent range cache (``read_cog``).
* GeoTIFF bytes from an elevation service - WCS GetCoverage, ArcGIS ``exportImage``, WMS GetMap with
  a GeoTIFF format (``read_geotiff_bytes`` + ``service_get``; requests are split to the service's
  pixel limit by ``service_tiles``).
* XYZ elevation tiles (PNG/WebP encoded heights, Web Mercator; ``read_xyz``).
"""

from __future__ import annotations

import hashlib
import math
import os
import time
from typing import Callable, Iterable

import numpy as np
import requests

from . import config
from .geo import AOI, Grid

# GDAL over HTTP: no directory probing for sidecar files, merged range requests, block cache
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("GDAL_HTTP_MULTIRANGE", "YES")
os.environ.setdefault("GDAL_HTTP_MERGE_CONSECUTIVE_RANGES", "YES")
os.environ.setdefault("VSI_CACHE", "TRUE")
os.environ.setdefault("GDAL_CACHEMAX", "512")

SERVICE_CACHE = config.CACHE_DIR / "dem_services"
SERVICE_CACHE_DAYS = 30
SERVICE_CACHE_BYTES = int(float(os.environ.get("MAPTOMESH_SERVICE_CACHE_GB", "2")) * 1e9)
_writes = 0


def _trim_service_cache() -> None:
    """Evict the oldest-written responses beyond the cache's size cap (MAPTOMESH_SERVICE_CACHE_GB)."""
    files = [(f.stat().st_mtime, f.stat().st_size, f) for f in SERVICE_CACHE.glob("*.bin")]
    total = sum(sz for _, sz, _ in files)
    if total <= SERVICE_CACHE_BYTES:
        return
    for _, sz, f in sorted(files):
        try:
            f.unlink()
        except OSError:
            continue
        total -= sz
        if total <= SERVICE_CACHE_BYTES * 0.9:
            break


def _dst_transform(grid: Grid):
    from affine import Affine   # rasterio's own (also in the browser, where rasterio isn't)
    # pixel centres on vertices; raster rows run north -> south
    return Affine(grid.step, 0, grid.x0 - grid.step / 2, 0, -grid.step,
                  grid.y0 + grid.ny * grid.step + grid.step / 2)


def _dst_crs(aoi: AOI):
    from rasterio.crs import CRS as RCRS
    return RCRS.from_wkt(aoi.crs.to_wkt())


def _crs(c):
    from rasterio.crs import CRS as RCRS
    if c is None or isinstance(c, RCRS):
        return c
    if isinstance(c, int):
        return RCRS.from_epsg(c)
    return RCRS.from_user_input(c)


def src_bbox(aoi: AOI, grid: Grid, crs, margin_cells: float = 3.0) -> tuple[float, float, float, float]:
    """Bounding box of the AOI (plus a margin) in ``crs`` (EPSG int, string or CRS)."""
    from pyproj import CRS, Transformer
    c = CRS.from_user_input(crs if not hasattr(crs, "to_wkt") else crs.to_wkt())
    ring = aoi.outline_local(margin=grid.step * margin_cells, n=24)
    tf = Transformer.from_crs(aoi.crs, c, always_xy=True)
    x, y = tf.transform(ring[:, 0], ring[:, 1])
    return float(np.min(x)), float(np.min(y)), float(np.max(x)), float(np.max(y))


def reproject_array(arr: np.ndarray, transform, crs, aoi: AOI, grid: Grid, nodata=None,
                    resampling: str = "bilinear") -> np.ndarray:
    """A source array (rows north -> south) with its affine transform and CRS -> vertex grid."""
    src = np.asarray(arr, dtype=np.float32)
    if nodata is not None and np.isfinite(nodata):
        src = np.where(src == nodata, np.nan, src)
    try:
        from rasterio.enums import Resampling
        from rasterio.warp import reproject
    except ImportError:   # in the browser (Pyodide): no GDAL
        return _reproject_numpy(src, transform, crs, aoi, grid, resampling)
    H, W = grid.vshape
    dst = np.full((H, W), np.nan, dtype=np.float32)
    reproject(source=src, destination=dst, src_transform=transform, src_crs=_crs(crs), src_nodata=np.nan,
              dst_transform=_dst_transform(grid), dst_crs=_dst_crs(aoi), dst_nodata=np.nan,
              resampling=getattr(Resampling, resampling))
    return dst[::-1].astype(np.float64)


def _open(url: str, **kw):
    import rasterio
    from .httpcache import CachedHttpFile
    if url.startswith("http"):
        try:
            return rasterio.open(url, opener=lambda path, mode="rb": CachedHttpFile(url), **kw)
        except (TypeError, ValueError):  # rasterio without opener support
            return rasterio.open(url, **kw)
    return rasterio.open(url, **kw)


def _prefetch_blocks(url: str, src, band: int, aoi: AOI, grid: Grid, crs, max_blocks: int = 20_000) -> None:
    """Fetch the COG tiles the warp is about to read, in parallel (GDAL reads them one at a time
    through ``CachedHttpFile``, about 6 MB/s instead of the link's speed). Only warms the range
    cache: tiles missed here (or a failure) are simply read by GDAL as before."""
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds
    from .httpcache import RANGE_CACHE

    try:
        h = grid.step / 2   # the destination's pixel edges (pixel centres sit on the vertices)
        b = (grid.x0 - h, grid.y0 - h, grid.x0 + grid.nx * grid.step + h, grid.y0 + grid.ny * grid.step + h)
        w, s, e, n = transform_bounds(_dst_crs(aoi), crs, *b, densify_pts=21)
        win = from_bounds(w, s, e, n, transform=src.transform)
        bh, bw = src.block_shapes[band - 1]
        pad = 2   # resampling kernel
        r0 = max(0, int(math.floor(win.row_off)) - pad) // bh
        r1 = min(src.height - 1, int(math.ceil(win.row_off + win.height)) + pad) // bh
        c0 = max(0, int(math.floor(win.col_off)) - pad) // bw
        c1 = min(src.width - 1, int(math.ceil(win.col_off + win.width)) + pad) // bw
        if r1 < r0 or c1 < c0 or (r1 - r0 + 1) * (c1 - c0 + 1) > max_blocks:
            return
        ranges = []
        for by in range(r0, r1 + 1):
            for bx in range(c0, c1 + 1):
                off = src.get_tag_item(f"BLOCK_OFFSET_{bx}_{by}", "TIFF", bidx=band)
                size = src.get_tag_item(f"BLOCK_SIZE_{bx}_{by}", "TIFF", bidx=band)
                if off and size and int(size) > 0:   # 0 / missing: sparse (empty) tile
                    ranges.append((int(off), int(size)))
        RANGE_CACHE.prefetch(url, ranges)
    except Exception:  # noqa: BLE001 - an optimisation only
        return


def read_cog(url: str, aoi: AOI, grid: Grid, src_crs=None, nodata=None, band: int = 1,
             scale: float = 1.0) -> np.ndarray:
    """Window of a (Cloud-Optimized) GeoTIFF on the vertex grid, from the coarsest overview that is
    still finer than half a grid cell. ``src_crs`` / ``nodata`` override missing file metadata."""
    H, W = grid.vshape
    if grid.read_blocks:   # only the rectangles under the gaps (Grid.read_blocks)
        out = np.full((H, W), np.nan)
        for r0, r1, c0, c1 in grid.read_blocks:
            sub = Grid(int(c1 - c0 - 1), int(r1 - r0 - 1), grid.step, grid.x0 + c0 * grid.step, grid.y0 + r0 * grid.step)
            out[r0:r1, c0:c1] = read_cog(url, aoi, sub, src_crs, nodata, band, scale)
        return out
    try:
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.warp import reproject
    except ImportError:   # in the browser (Pyodide): no GDAL
        out = _read_cog_nogdal(url, aoi, grid, src_crs, nodata, band)
        return out * scale if scale != 1.0 else out
    dst = np.full((H, W), np.nan, dtype=np.float32)
    with _open(url) as src0:
        res = abs(src0.res[0])
        crs = _crs(src_crs) or src0.crs
        if crs is not None and crs.is_geographic:  # degrees -> metres (roughly, for the LOD choice)
            res *= 111_320.0 * math.cos(math.radians(aoi.lat))
        level = -1
        for k, f in enumerate(src0.overviews(band)):
            if res * f <= grid.step / 2:
                level = k
    kw = {"overview_level": level} if level >= 0 else {}
    with _open(url, **kw) as src:
        if url.startswith("http"):
            _prefetch_blocks(url, src, band, aoi, grid, _crs(src_crs) or src.crs)
        nd = nodata if nodata is not None else src.nodata
        reproject(source=rasterio.band(src, band), destination=dst, src_crs=_crs(src_crs) or src.crs,
                  src_nodata=nd, dst_nodata=np.nan, dst_transform=_dst_transform(grid), dst_crs=_dst_crs(aoi),
                  resampling=Resampling.bilinear if level < 0 else Resampling.average)
    out = dst[::-1].astype(np.float64)
    return out * scale if scale != 1.0 else out


def _crs_of(c):
    """pyproj CRS from an EPSG int, a string (WKT, "EPSG:x") or a rasterio CRS."""
    from pyproj import CRS
    return CRS.from_user_input(c.to_wkt() if hasattr(c, "to_wkt") else c)


MAX_WINDOW_PX = 64_000_000   # GDAL-free reads: larger windows (no overview fine enough) are refused


def _read_cog_nogdal(url: str, aoi: AOI, grid: Grid, src_crs=None, nodata=None, band: int = 1) -> np.ndarray:
    """``read_cog`` with ``maptomesh.geotiff`` instead of GDAL (overview choice as above)."""
    from .geotiff import GeoTiff, UrlSource
    g = GeoTiff(UrlSource(url))
    crs = src_crs if src_crs is not None else g.crs
    if crs is None:
        raise ValueError(f"{url}: no CRS in the GeoTIFF (pass src_crs)")
    level = _nogdal_level(g, crs, aoi, grid)
    return _read_tiff_window(g, level, crs, aoi, grid, nodata, band, "bilinear" if level == 0 else "average")


def _nogdal_level(g, crs, aoi: AOI, grid: Grid) -> int:
    """The coarsest overview with pixels at most half a grid cell (0 = full resolution)."""
    res = abs(g.transform0.a)
    if _crs_of(crs).is_geographic:
        res *= 111_320.0 * math.cos(math.radians(aoi.lat))
    level = 0
    for k, f in enumerate(g.factors, 1):
        if res * f <= grid.step / 2:
            level = k
    return level


def _tiff_window(g, level: int, crs, aoi: AOI, grid: Grid) -> tuple[int, int, int, int] | None:
    """(row0, row1, col0, col1) of the area at ``level``, None if outside the file."""
    tf = g.transform(level)
    h, w = g.shape(level)
    x0, y0, x1, y1 = src_bbox(aoi, grid, crs)
    inv = ~tf
    cols, rows = zip(*(inv * (x, y) for x in (x0, x1) for y in (y0, y1)))
    c0, c1 = max(0, int(math.floor(min(cols))) - 1), min(w, int(math.ceil(max(cols))) + 1)
    r0, r1 = max(0, int(math.floor(min(rows))) - 1), min(h, int(math.ceil(max(rows))) + 1)
    if c1 <= c0 or r1 <= r0:
        return None
    return r0, r1, c0, c1


def cog_download_bytes(url: str, aoi: AOI, grid: Grid, src_crs=None) -> int:
    """Bytes the GDAL-free reader (the browser) downloads to read the area from ``url``: the
    compressed tiles of the window at its overview level, from the file's header (fetched; cached),
    plus the header itself. 0 if the file doesn't exist or misses the area."""
    from .geotiff import GeoTiff, UrlSource
    try:
        g = GeoTiff(UrlSource(url))
    except Exception:  # noqa: BLE001 - missing tile (sea): the job skips it too
        return 0
    crs = src_crs if src_crs is not None else g.crs
    if crs is None:
        return 0
    level = _nogdal_level(g, crs, aoi, grid)
    win = _tiff_window(g, level, crs, aoi, grid)
    if win is None:
        return 0
    r0, r1, c0, c1 = win
    im = g.images[level]
    ty0, ty1 = r0 // im.th, (r1 - 1) // im.th + 1
    tx0, tx1 = c0 // im.tw, (c1 - 1) // im.tw + 1
    total = 1 << 16
    for ty in range(ty0, ty1):
        a = ty * im.across + tx0
        total += int(np.asarray(im._entries(im.counts, a, a + (tx1 - tx0)), np.int64).sum())
    return total


def _read_tiff_window(g, level: int, crs, aoi: AOI, grid: Grid, nodata, band: int, resampling: str) -> np.ndarray:
    from affine import Affine
    tf = g.transform(level)
    win = _tiff_window(g, level, crs, aoi, grid)
    if win is None:
        return np.full(grid.vshape, np.nan)
    r0, r1, c0, c1 = win
    if (r1 - r0) * (c1 - c0) > MAX_WINDOW_PX:
        raise RuntimeError(f"{(r1 - r0) * (c1 - c0) / 1e6:.0f} M pixels to read (no coarser overview)")
    arr = g.read(level, (r0, r1, c0, c1), band, nodata)
    return reproject_array(arr, tf * Affine.translation(c0, r0), crs, aoi, grid, resampling=resampling)


class RasterBytes:
    """Band of a raster file held in memory: ``arr`` (as stored; NaN where nodata on the GDAL-free
    path), ``nodata``, ``crs`` (rasterio CRS, EPSG int or WKT), ``transform``, ``res``."""

    def __init__(self, arr, nodata, crs, transform):
        self.arr, self.nodata, self.crs, self.transform = arr, nodata, crs, transform
        self.res = abs(transform.a)


def read_raster_bytes(data: bytes, ext: str = ".tif", band: int = 1) -> RasterBytes:
    """A GeoTIFF / ESRI ASCII grid from a service or a zip: GDAL when present, else
    ``maptomesh.geotiff`` / a plain-text parser (the browser)."""
    try:
        from rasterio.io import MemoryFile
    except ImportError:
        if data.lstrip()[:5].lower() == b"ncols":
            return _ascii_grid(data)
        from .geotiff import BytesSource, GeoTiff
        g = GeoTiff(BytesSource(data))
        return RasterBytes(g.read(0, None, band), g.nodata, g.crs, g.transform0)
    with MemoryFile(data, ext=ext) as mf, mf.open() as src:
        return RasterBytes(src.read(band), src.nodata, src.crs, src.transform)


def _ascii_grid(data: bytes) -> RasterBytes:
    """ESRI ASCII grid (ncols, nrows, xllcorner / xllcenter, cellsize, NODATA_value, then the
    rows); the format has no CRS (callers pass src_crs)."""
    from affine import Affine
    lines = data.split(b"\n", 8)
    head, i = {}, 7
    for k, ln in enumerate(lines[:7]):
        parts = ln.split()
        if len(parts) == 2 and parts[0][:1].isalpha():
            head[parts[0].decode().lower()] = float(parts[1])
        else:
            i = k
            break
    body = b"\n".join(lines[i:])
    nc, nr, cs = int(head["ncols"]), int(head["nrows"]), head["cellsize"]
    x0 = head["xllcorner"] if "xllcorner" in head else head.get("xllcenter", 0.0) - cs / 2
    y0 = head["yllcorner"] if "yllcorner" in head else head.get("yllcenter", 0.0) - cs / 2
    arr = np.array(body.split(), dtype=np.float32)[:nc * nr].reshape(nr, nc)
    nd = head.get("nodata_value")
    if nd is not None:
        arr[arr == np.float32(nd)] = np.nan
    return RasterBytes(arr, nd, None, Affine(cs, 0, x0, 0, -cs, y0 + nr * cs))


def read_geotiff_bytes(data: bytes, aoi: AOI, grid: Grid, src_crs=None, nodata=None,
                       band: int = 1) -> np.ndarray:
    """A GeoTIFF returned by a service (WCS / exportImage / WMS) -> vertex grid."""
    try:
        from rasterio.io import MemoryFile
    except ImportError:   # in the browser (Pyodide): no GDAL
        from .geotiff import BytesSource, GeoTiff
        g = GeoTiff(BytesSource(data))
        crs = src_crs if src_crs is not None else g.crs
        if crs is None:
            raise ValueError("service GeoTIFF has no CRS (pass src_crs)")
        return reproject_array(g.read(0, None, band, nodata), g.transform0, crs, aoi, grid)
    with MemoryFile(data) as mf, mf.open() as src:
        arr = src.read(band).astype(np.float32)
        nd = nodata if nodata is not None else src.nodata
        crs = _crs(src_crs) or src.crs
        if crs is None:
            raise ValueError("service GeoTIFF has no CRS (pass src_crs)")
        return reproject_array(arr, src.transform, crs, aoi, grid, nodata=nd)


def _reproject_numpy(src: np.ndarray, transform, crs, aoi: AOI, grid: Grid, resampling: str,
                     lattice: int = 8, rows: int = 256) -> np.ndarray:
    """``reproject_array`` without GDAL: bilinear (or nearest) sampling, NaN-aware like GDAL's
    (weights renormalised over valid neighbours), NaN outside the source. Source positions are
    projected exactly on a lattice of every ``lattice``-th vertex and interpolated between (the
    projections are smooth over a few cells), then sampled ``rows`` grid rows at a time."""
    from pyproj import CRS, Transformer
    from scipy import ndimage
    H, W = grid.vshape
    k = lattice
    ci, cj = np.arange(0, H + k, k), np.arange(0, W + k, k)
    xx, yy = np.meshgrid(grid.x0 + cj * grid.step, grid.y0 + ci * grid.step)
    c = crs.to_wkt() if hasattr(crs, "to_wkt") else crs
    sx, sy = Transformer.from_crs(aoi.crs, CRS.from_user_input(c), always_xy=True).transform(xx, yy)
    inv = ~transform
    scol = inv.a * sx + inv.b * sy + inv.c - 0.5        # pixel-centre coordinates
    srow = inv.d * sx + inv.e * sy + inv.f - 0.5
    valid = np.isfinite(src)
    vals = np.where(valid, src, 0.0).astype(np.float32)
    wts = valid.astype(np.float32)
    order = 0 if resampling == "nearest" else 1
    if order == 1:
        # Downsampling: GDAL widens the bilinear tent to the output pixel (half-width s source
        # pixels). Pre-filter with a tent of half-width sqrt(s^2 - 1) so that, followed by the 1-pixel
        # tent of map_coordinates, the spread (variance) matches.
        s_col = float(np.median(np.abs(np.diff(scol, axis=1)))) / k
        s_row = float(np.median(np.abs(np.diff(srow, axis=0)))) / k
        for axis, s in ((1, s_col), (0, s_row)):
            a = math.sqrt(s * s - 1.0) if s > 1.0 else 0.0
            r = int(math.floor(a))
            if r >= 1:
                kern = np.maximum(0.0, 1.0 - np.abs(np.arange(-r, r + 1)) / a).astype(np.float32)
                kern /= kern.sum()
                vals = ndimage.convolve1d(vals, kern, axis=axis, mode="constant", cval=0.0)
                wts = ndimage.convolve1d(wts, kern, axis=axis, mode="constant", cval=0.0)
    sh, sw = src.shape
    out = np.full((H, W), np.nan, np.float32)
    jj = np.arange(W) / k
    for i0 in range(0, H, rows):
        ii = np.arange(i0, min(H, i0 + rows)) / k
        I, J = np.meshgrid(ii, jj, indexing="ij")
        c_ = ndimage.map_coordinates(scol, [I, J], order=1)
        r_ = ndimage.map_coordinates(srow, [I, J], order=1)
        v = ndimage.map_coordinates(vals, [r_, c_], order=order, mode="grid-constant", cval=0.0, prefilter=False)
        w = ndimage.map_coordinates(wts, [r_, c_], order=order, mode="grid-constant", cval=0.0, prefilter=False)
        ok = (c_ >= -0.5) & (c_ <= sw - 0.5) & (r_ >= -0.5) & (r_ <= sh - 0.5) & (w > 1e-3)
        blk = np.full(v.shape, np.nan, np.float32)
        blk[ok] = v[ok] / w[ok]
        out[i0:i0 + len(ii)] = blk
    return out.astype(np.float64)   # rows already south -> north, like the vertex grid


def merge_fill(dst: np.ndarray, part: np.ndarray) -> np.ndarray:
    """Fill NaNs of ``dst`` from ``part`` (pieces of one source read in several requests)."""
    take = np.isnan(dst) & np.isfinite(part)
    dst[take] = part[take]
    return dst


# ---------------------------------------------------------------- services

def service_get(url: str, params: dict | None = None, timeout: float = 120.0, cache: bool = True,
                headers: dict | None = None, method: str = "GET", data=None, allow_xml: bool = False) -> bytes:
    """HTTP request to an elevation service, cached on disk for 30 days (same AOI and detail ->
    no repeated request). Raises for HTTP errors and, unless ``allow_xml`` (index / capabilities
    documents), for XML/HTML/JSON error documents returned instead of raster data."""
    key_src = url + "?" + "&".join(f"{k}={v}" for k, v in sorted((params or {}).items())) + (str(data) if data else "")
    key = hashlib.sha1(key_src.encode()).hexdigest()[:24]
    path = SERVICE_CACHE / f"{key}.bin"
    if cache and path.exists() and time.time() - path.stat().st_mtime < SERVICE_CACHE_DAYS * 86400:
        return path.read_bytes()
    h = {"User-Agent": config.USER_AGENT}
    h.update(headers or {})
    s = requests.Session()
    s.mount("https://", requests.adapters.HTTPAdapter(max_retries=requests.adapters.Retry(
        total=3, connect=3, backoff_factor=1.0, status_forcelist=(429, 500, 502, 503, 504), allowed_methods=None)))
    r = s.request(method, url, params=params, headers=h, timeout=(20, timeout), data=data)
    r.raise_for_status()
    body = r.content
    head = body[:200].lstrip().lower()
    if not allow_xml and head.startswith((b"<?xml", b"<html", b"<!doctype", b"<serviceexception", b"<ows:",
                                          b"{\"error")) \
            and "tiff" not in r.headers.get("Content-Type", "").lower():
        raise RuntimeError(f"service error from {url}: {body[:300]!r}")
    if cache:
        SERVICE_CACHE.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(body)
        tmp.replace(path)
        global _writes
        _writes += 1
        if _writes % 50 == 1:
            _trim_service_cache()
    return body


def service_tiles(bbox: tuple[float, float, float, float], res: float, max_px: int
                  ) -> Iterable[tuple[tuple[float, float, float, float], int, int]]:
    """Split ``bbox`` (source CRS units) at pixel size ``res`` into requests of at most ``max_px``
    pixels per side. Yields (bbox, width, height)."""
    x0, y0, x1, y1 = bbox
    W = max(2, int(math.ceil((x1 - x0) / res)))
    H = max(2, int(math.ceil((y1 - y0) / res)))
    nx, ny = int(math.ceil(W / max_px)), int(math.ceil(H / max_px))
    for j in range(ny):
        for i in range(nx):
            w = min(max_px, W - i * max_px)
            h = min(max_px, H - j * max_px)
            bx0 = x0 + i * max_px * res
            by0 = y0 + j * max_px * res
            yield (bx0, by0, bx0 + w * res, by0 + h * res), w, h


def request_res(native_res: float, grid: Grid, scale: float = 1.0) -> float:
    """Pixel size to request from a service: at least half a grid cell (no point in more detail),
    never finer than the data. ``scale`` converts metres to source CRS units (e.g. degrees)."""
    return max(native_res, grid.step / 2) * scale


def read_service(fetch_tile: Callable[[tuple, int, int], bytes], bbox, res: float, max_px: int,
                 aoi: AOI, grid: Grid, src_crs=None, nodata=None, cancel=None) -> np.ndarray:
    """Read an elevation service in pixel-limited pieces and merge them on the vertex grid."""
    out = np.full(grid.vshape, np.nan)
    for b, w, h in service_tiles(bbox, res, max_px):
        if cancel is not None and cancel.is_set():
            raise RuntimeError("Cancelled")
        merge_fill(out, read_geotiff_bytes(fetch_tile(b, w, h), aoi, grid, src_crs=src_crs, nodata=nodata))
    return out


# ---------------------------------------------------------------- XYZ tiles

def _lonlat_to_tile(lon: float, lat: float, z: int) -> tuple[float, float]:
    n = 2 ** z
    x = (lon + 180.0) / 360.0 * n
    lat_r = math.radians(max(min(lat, 85.0511), -85.0511))
    y = (1.0 - math.asinh(math.tan(lat_r)) / math.pi) / 2.0 * n
    return x, y


def read_xyz(template: str, zoom: int, decode: Callable[[bytes], np.ndarray | None], aoi: AOI, grid: Grid,
             tile_px: int = 256, cancel=None, max_tiles: int = 400) -> np.ndarray:
    """Mosaic of XYZ elevation tiles (Web Mercator) at ``zoom`` -> vertex grid. ``decode(bytes)``
    returns a (tile_px, tile_px) float array (NaN = no data) or None for a missing tile; 404s count
    as missing. Tiles go through the persistent range cache."""
    from .httpcache import RANGE_CACHE

    w, s, e, n = aoi.lonlat_bbox(margin=grid.step * 3)
    tx0, ty0 = _lonlat_to_tile(w, n, zoom)
    tx1, ty1 = _lonlat_to_tile(e, s, zoom)
    xs = range(int(math.floor(tx0)), int(math.floor(tx1)) + 1)
    ys = range(int(math.floor(ty0)), int(math.floor(ty1)) + 1)
    if len(xs) * len(ys) > max_tiles:
        raise RuntimeError(f"{len(xs) * len(ys)} tiles at zoom {zoom} (limit {max_tiles})")
    mosaic = np.full((len(ys) * tile_px, len(xs) * tile_px), np.nan, dtype=np.float32)
    from .httpcache import hint
    for ty in ys:
        for tx in xs:
            hint(template.format(z=zoom, x=tx, y=ty), 0, 1 << 24)
    for j, ty in enumerate(ys):
        for i, tx in enumerate(xs):
            if cancel is not None and cancel.is_set():
                raise RuntimeError("Cancelled")
            url = template.format(z=zoom, x=tx, y=ty)
            try:
                data = RANGE_CACHE.read(url, 0, 1 << 24)
            except requests.HTTPError as ex:
                if ex.response is not None and ex.response.status_code == 404:
                    continue
                raise
            arr = decode(data) if data else None
            if arr is not None:
                mosaic[j * tile_px:(j + 1) * tile_px, i * tile_px:(i + 1) * tile_px] = arr
    # Web Mercator transform of the mosaic
    R = 6378137.0
    world = 2 * math.pi * R
    px = world / (2 ** zoom * tile_px)
    left = -world / 2 + xs[0] * tile_px * px
    top = world / 2 - ys[0] * tile_px * px
    return reproject_array(mosaic, Affine(px, 0, left, 0, -px, top), 3857, aoi, grid)


def decode_png(data: bytes) -> np.ndarray:
    """RGB(A) pixels of a PNG/WebP tile as a (H, W, C) uint8 array."""
    try:
        from rasterio.io import MemoryFile
    except ImportError:   # in the browser (Pyodide): Pillow (palettes expanded, transparency kept)
        import io

        from PIL import Image
        im = Image.open(io.BytesIO(data))
        alpha = im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info
        return np.asarray(im.convert("RGBA" if alpha else "RGB"))
    with MemoryFile(data) as mf, mf.open() as src:
        return np.moveaxis(src.read(), 0, -1)


def terrarium(data: bytes) -> np.ndarray:
    """Mapzen/Mapterhorn Terrarium encoding: h = R*256 + G + B/256 - 32768."""
    p = decode_png(data).astype(np.float64)
    return p[..., 0] * 256 + p[..., 1] + p[..., 2] / 256 - 32768


def zoom_for(res_m: float, lat: float, tile_px: int = 256, max_zoom: int = 17) -> int:
    """Lowest zoom whose pixels are at least as fine as ``res_m`` at ``lat``."""
    world = 2 * math.pi * 6378137.0 * math.cos(math.radians(lat))
    z = int(math.ceil(math.log2(world / (tile_px * max(res_m, 0.01)))))
    return max(0, min(max_zoom, z))
