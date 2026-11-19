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
    from rasterio.transform import Affine
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
    from rasterio.enums import Resampling
    from rasterio.warp import reproject
    H, W = grid.vshape
    dst = np.full((H, W), np.nan, dtype=np.float32)
    src = np.asarray(arr, dtype=np.float32)
    if nodata is not None and np.isfinite(nodata):
        src = np.where(src == nodata, np.nan, src)
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
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    H, W = grid.vshape
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


def read_geotiff_bytes(data: bytes, aoi: AOI, grid: Grid, src_crs=None, nodata=None,
                       band: int = 1) -> np.ndarray:
    """A GeoTIFF returned by a service (WCS / exportImage / WMS) -> vertex grid."""
    from rasterio.io import MemoryFile
    with MemoryFile(data) as mf, mf.open() as src:
        arr = src.read(band).astype(np.float32)
        nd = nodata if nodata is not None else src.nodata
        crs = _crs(src_crs) or src.crs
        if crs is None:
            raise ValueError("service GeoTIFF has no CRS (pass src_crs)")
        return reproject_array(arr, src.transform, crs, aoi, grid, nodata=nd)


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
    from rasterio.transform import Affine
    from .httpcache import RANGE_CACHE

    w, s, e, n = aoi.lonlat_bbox(margin=grid.step * 3)
    tx0, ty0 = _lonlat_to_tile(w, n, zoom)
    tx1, ty1 = _lonlat_to_tile(e, s, zoom)
    xs = range(int(math.floor(tx0)), int(math.floor(tx1)) + 1)
    ys = range(int(math.floor(ty0)), int(math.floor(ty1)) + 1)
    if len(xs) * len(ys) > max_tiles:
        raise RuntimeError(f"{len(xs) * len(ys)} tiles at zoom {zoom} (limit {max_tiles})")
    mosaic = np.full((len(ys) * tile_px, len(xs) * tile_px), np.nan, dtype=np.float32)
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
    from rasterio.io import MemoryFile
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
