"""Asian elevation models.

* Japan: GSI (国土地理院) elevation tiles, XYZ PNG tiles on cyberjapandata.gsi.go.jp. One provider
  merges GSI's own chain: DEM1A 1 m (aerial LiDAR) -> DEM5A 5 m (LiDAR) -> DEM5B / DEM5C 5 m
  (photogrammetry) -> DEM10B 10 m (nationwide). GSI classifies the elevation tiles as "not a basic
  survey result, usable with source credit only" (no Survey Act application; PDL1.0 allows
  commercial use). Heights: x = R*2^16 + G*2^8 + B, h = x * 0.01 m (x - 2^24 when x > 2^23), x = 2^23
  is no data. Bare earth (DTM).
* Hong Kong: CEDD 2020 airborne LiDAR 0.5 m DTM + DSM, one zipped GeoTIFF per 750 x 600 m sheet
  (HK1980 Grid, EPSG:2326, heights in mPD) on bulkdata.csdi.gov.hk, found through an ArcGIS
  FeatureServer sheet index.
"""

from __future__ import annotations

import io
import json
import math
import warnings
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests

from ..dem import DemArrays, DemProvider, register, want_dsm
from ..dem_readers import merge_fill, reproject_array, service_get, zoom_for
from ..httpcache import RANGE_CACHE

WORKERS = 4                      # parallel requests per host (be gentle)


def _cancelled(cancel) -> None:
    if cancel.is_set():
        raise RuntimeError("Cancelled")


def _get(url: str, max_bytes: int) -> bytes | None:
    """Whole small file through the persistent range cache; None for 404."""
    try:
        return RANGE_CACHE.read(url, 0, max_bytes) or None
    except requests.HTTPError as ex:
        if ex.response is not None and ex.response.status_code == 404:
            return None
        raise


# ---------------------------------------------------------------- Japan (GSI)

GSI_URL = "https://cyberjapandata.gsi.go.jp/xyz/{layer}/{z}/{x}/{y}.png"
GSI_DETAIL = (  # layer, max zoom, native resolution (m), name - best first
    ("dem1a_png", 17, 1.0, "DEM1A"),
    ("dem5a_png", 15, 5.0, "DEM5A"),
    ("dem5b_png", 15, 5.0, "DEM5B"),
    ("dem5c_png", 15, 5.0, "DEM5C"),
)
GSI_BASE = ("dem_png", 14, 10.0, "DEM10B")   # nationwide: land mask + last fallback
GSI_MAX_TILES = 256              # per level; the zoom is lowered beyond this
TILE_PX = 256
WEB_MERC = 2 * math.pi * 6378137.0


def _decode_gsi(data: bytes) -> np.ndarray | None:
    from rasterio.errors import NotGeoreferencedWarning
    from rasterio.io import MemoryFile
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with MemoryFile(data) as mf, mf.open() as src:
            px = src.read()
            if src.count == 1:                    # paletted PNG
                cmap = src.colormap(1)
                lut = np.array([cmap.get(i, (0, 0, 0, 0)) for i in range(256)], np.uint8)
                px = np.moveaxis(lut[px[0]], -1, 0)
    rgb = px[:3].astype(np.int64)
    x = (rgb[0] << 16) | (rgb[1] << 8) | rgb[2]
    h = np.where(x < 2 ** 23, x, x - 2 ** 24).astype(np.float32) * 0.01
    nodata = x == 2 ** 23
    if px.shape[0] == 4:
        nodata |= px[3] == 0
    h[nodata] = np.nan
    return None if nodata.all() else h


def _tile_xy(lon, lat, z: int):
    """Fractional XYZ tile coordinates (vectorised)."""
    n = 2.0 ** z
    x = (np.asarray(lon, np.float64) + 180.0) / 360.0 * n
    lat_r = np.radians(np.clip(np.asarray(lat, np.float64), -85.0511, 85.0511))
    y = (1.0 - np.arcsinh(np.tan(lat_r)) / np.pi) / 2.0 * n
    return x, y


def _tiles_for(aoi, grid, mask: np.ndarray, z: int) -> list[tuple[int, int]]:
    """Tiles at zoom ``z`` holding the vertices in ``mask``, padded by a couple of grid cells so
    that interpolation near a tile edge has its neighbours. The mask is pooled into blocks much
    smaller than a tile first (tiles are >= 64 grid cells wide), so large grids stay cheap."""
    H, W = mask.shape
    tile_m = WEB_MERC * math.cos(math.radians(aoi.lat)) / 2 ** z
    s = max(1, int(tile_m / grid.step / 8))
    Hs, Ws = -(-H // s), -(-W // s)
    pool = np.zeros((Hs * s, Ws * s), bool)
    pool[:H, :W] = mask
    jj, ii = np.nonzero(pool.reshape(Hs, s, Ws, s).any(axis=(1, 3)))
    if not len(ii):
        return []
    lon, lat = aoi.to_lonlat.transform(grid.x0 + (ii * s + (s - 1) / 2) * grid.step,
                                       grid.y0 + (jj * s + (s - 1) / 2) * grid.step)
    x, y = _tile_xy(lon, lat, z)
    pad = (0.75 * s + 2) * grid.step / tile_m
    keys = [np.floor(x + dx).astype(np.int64) * (1 << 30) + np.floor(y + dy).astype(np.int64)
            for dx in (-pad, 0.0, pad) for dy in (-pad, 0.0, pad)]
    return [(int(k >> 30), int(k & ((1 << 30) - 1))) for k in np.unique(np.concatenate(keys))]


def _bbox_tile_count(bbox, z: int) -> int:
    w, s, e, n = bbox
    x0, y0 = _tile_xy(w, n, z)
    x1, y1 = _tile_xy(e, s, z)
    return (int(math.floor(x1)) - int(math.floor(x0)) + 1) * (int(math.floor(y1)) - int(math.floor(y0)) + 1)


def _gsi_zoom(res: float, max_zoom: int, aoi, grid, bbox) -> int:
    """No finer than half a grid cell (or the data), lowered until the tile count is reasonable."""
    z = min(max_zoom, zoom_for(max(res, grid.step / 2), aoi.lat))
    while z > 1 and _bbox_tile_count(bbox, z) > GSI_MAX_TILES:
        z -= 1
    return z


def _gsi_tiles(layer: str, z: int, tiles, cancel) -> dict[tuple[int, int], np.ndarray]:
    """Decoded tiles of one layer (4 requests at a time); missing / empty tiles are left out."""
    def one(t):
        if cancel.is_set():
            return t, None
        data = _get(GSI_URL.format(layer=layer, z=z, x=t[0], y=t[1]), 1 << 22)
        return t, (_decode_gsi(data) if data else None)

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        got = {t: a for t, a in ex.map(one, tiles) if a is not None}
    _cancelled(cancel)
    return got


def _gsi_with_data(layer: str, z: int, tiles, cancel, up: int = 3) -> list[tuple[int, int]]:
    """The tiles of ``layer`` that hold data, judged from their parent tiles ``up`` zoom levels
    coarser (GSI builds every zoom level): one parent request instead of up to 64 child requests
    (404s) where a model has no coverage, e.g. DEM1A outside its survey areas."""
    zp = max(0, z - up)
    f = 2 ** (z - zp)
    if f == 1:
        return list(tiles)
    parents = _gsi_tiles(layer, zp, sorted({(tx // f, ty // f) for tx, ty in tiles}), cancel)
    b = TILE_PX // f
    keep = []
    for tx, ty in tiles:
        a = parents.get((tx // f, ty // f))
        if a is None:
            continue
        r0, c0 = (ty % f) * b, (tx % f) * b
        if np.isfinite(a[max(0, r0 - 1):r0 + b + 1, max(0, c0 - 1):c0 + b + 1]).any():
            keep.append((tx, ty))
    return keep


def _gsi_level(layer: str, z: int, tiles, aoi, grid, cancel) -> tuple[np.ndarray | None, int, int]:
    """Fetch tiles of one layer and put the mosaic on the vertex grid.
    Returns (grid array or None, tiles requested, tiles with data)."""
    from rasterio.transform import Affine

    got = _gsi_tiles(layer, z, tiles, cancel)
    if not got:
        return None, len(tiles), 0
    xs = [t[0] for t in got]
    ys = [t[1] for t in got]
    tx0, ty0 = min(xs), min(ys)
    mosaic = np.full(((max(ys) - ty0 + 1) * TILE_PX, (max(xs) - tx0 + 1) * TILE_PX), np.nan, np.float32)
    for (tx, ty), a in got.items():
        r, c = (ty - ty0) * TILE_PX, (tx - tx0) * TILE_PX
        mosaic[r:r + TILE_PX, c:c + TILE_PX] = a
    px = WEB_MERC / (2 ** z * TILE_PX)
    tf = Affine(px, 0, -WEB_MERC / 2 + tx0 * TILE_PX * px, 0, -px, WEB_MERC / 2 - ty0 * TILE_PX * px)
    return reproject_array(mosaic, tf, 3857, aoi, grid), len(tiles), len(got)


class GsiDem(DemProvider):
    key = "gsi_dem"
    label = "GSI 1-10 m (Japan)"
    res_m = 1.0                  # best level; DEM5A/5B/5C/DEM10B fill the gaps inside ``fetch``
    attribution = ("出典：国土地理院 標高タイル（https://maps.gsi.go.jp/development/ichiran.html）を加工して作成 / "
                   "Source: Geospatial Information Authority of Japan (GSI) elevation tiles, processed")
    licence = ("国土地理院コンテンツ利用規約 / Public Data License 1.0 (CC BY 4.0 compatible, commercial use allowed); "
               "elevation tiles need source credit only, no Survey Act application "
               "(https://maps.gsi.go.jp/development/ichiran.html, https://www.gsi.go.jp/kikakuchousei/kikakuchousei40182.html)")
    regions = (
        (139.3, 41.3, 145.9, 45.6),   # Hokkaido
        (130.8, 33.0, 142.2, 41.6),   # Honshu, Shikoku, Izu islands
        (129.4, 30.9, 132.1, 34.0),   # Kyushu
        (128.55, 32.5, 129.75, 34.75),  # Goto, Tsushima, Iki
        (128.3, 27.0, 131.4, 30.9),   # Satsunan islands
        (122.9, 24.0, 129.4, 27.95),  # Okinawa, Sakishima
        (138.9, 24.0, 142.4, 33.0),   # southern Izu, Ogasawara
    )
    max_area_km2 = 10_000.0      # zoom follows the grid step, so the tile count stays bounded
    sample = (35.3606, 138.7274, 3000)   # Mt Fuji

    def fetch(self, aoi, grid, progress, cancel):
        bbox = aoi.lonlat_bbox(margin=grid.step * 3)
        H, W = grid.vshape
        log = []

        # DEM10B first: it covers all of Japan's land, so it is both the land mask (no tiles are
        # requested over the sea or outside Japan) and the last fallback.
        layer, maxz, res, name = GSI_BASE
        zb = _gsi_zoom(res, maxz, aoi, grid, bbox)
        progress(0.0, f"GSI {name} z{zb}")
        tiles = _gsi_with_data(layer, zb, _tiles_for(aoi, grid, np.ones((H, W), bool), zb), cancel)
        base, nreq, nok = _gsi_level(layer, zb, tiles, aoi, grid, cancel) if tiles else (None, 0, 0)
        if base is None or not np.isfinite(base).any():
            return DemArrays(np.full((H, W), np.nan), None, f"no GSI data ({nreq} tiles)")
        land = np.isfinite(base)

        out = np.full((H, W), np.nan)
        for k, (layer, maxz, res, name) in enumerate(GSI_DETAIL):
            remaining = np.isnan(out) & land
            if remaining.sum() < max(4, 2e-4 * remaining.size):
                break
            z = _gsi_zoom(res, maxz, aoi, grid, bbox)
            progress((k + 1) / (len(GSI_DETAIL) + 1), f"GSI {name} z{z}")
            cand = _tiles_for(aoi, grid, remaining, z)
            tiles = _gsi_with_data(layer, z, cand, cancel)
            part, nreq, nok = _gsi_level(layer, z, tiles, aoi, grid, cancel) if tiles else (None, 0, 0)
            if part is None:
                log.append(f"{name} z{z} none ({len(cand)} tiles checked)")
                continue
            before = np.isnan(out).sum()
            merge_fill(out, part)
            log.append(f"{name} z{z} {nok}/{nreq} tiles {(before - np.isnan(out).sum()) / out.size:.0%}")
        before = np.isnan(out).sum()
        merge_fill(out, base)
        log.append(f"{GSI_BASE[3]} z{zb} {(before - np.isnan(out).sum()) / out.size:.0%}")
        return DemArrays(out, None, "; ".join(log))


# ---------------------------------------------------------------- Hong Kong (CEDD)

HK_INDEX = ("https://services3.arcgis.com/6j1KwZfY2fZrfNMR/arcgis/rest/services/"
            "Digital_Terrain_Model_from_2020_LiDAR_Survey/FeatureServer/0/query")
HK_NATIVE = 0.5


def _hk_sheet(data: bytes | None, aoi, grid) -> np.ndarray | None:
    """Zipped sheet GeoTIFF -> vertex grid (block average when the grid is much coarser)."""
    from rasterio.io import MemoryFile
    if not data:
        return None
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        name = next((n for n in zf.namelist() if n.lower().endswith((".tif", ".tiff"))), None)
        if name is None:
            return None
        tif = zf.read(name)
    with MemoryFile(tif) as mf, mf.open() as src:
        arr = src.read(1)
        nd = src.nodata if src.nodata is not None else -9999.0
        crs = src.crs or 2326
        tf = src.transform
    resampling = "average" if grid.step > 2 * HK_NATIVE else "bilinear"
    return reproject_array(arr, tf, crs, aoi, grid, nodata=nd, resampling=resampling)


class HongKongLidar2020(DemProvider):
    key = "hk_cedd_lidar2020"
    label = "CEDD LiDAR 2020 0.5 m (Hong Kong)"
    res_m = 0.5
    has_dsm = True
    attribution = ("Source: Civil Engineering and Development Department, Government of the Hong Kong SAR - "
                   "2020 Airborne LiDAR Survey DTM/DSM (CSDI Portal / DATA.GOV.HK)")
    licence = ("DATA.GOV.HK Terms and Conditions: free commercial and non-commercial use with attribution "
               "(https://data.gov.hk/en/terms-and-conditions)")
    regions = ((113.82, 22.13, 114.52, 22.57),)
    max_area_km2 = 25.0          # whole ~5.7 MB sheets (0.45 km2 each), DTM + DSM: ~0.3 GB at 25 km2
    sample = (22.2700, 114.1500, 1000)   # Victoria Peak / Pok Fu Lam

    def fetch(self, aoi, grid, progress, cancel):
        w, s, e, n = aoi.lonlat_bbox(margin=grid.step * 3)
        body = service_get(HK_INDEX, {
            "geometry": f"{w:.6f},{s:.6f},{e:.6f},{n:.6f}", "geometryType": "esriGeometryEnvelope",
            "inSR": "4326", "spatialRel": "esriSpatialRelIntersects", "outFields": "TILE_NAME,URL",
            "returnGeometry": "false", "resultRecordCount": "2000", "f": "json"}, timeout=60)
        js = json.loads(body)
        if js.get("exceededTransferLimit"):
            raise RuntimeError("Hong Kong sheet index: too many sheets for one query")
        sheets = sorted({(f["attributes"].get("TILE_NAME") or "", f["attributes"]["URL"])
                         for f in js.get("features", []) if f.get("attributes", {}).get("URL")})
        dtm = np.full(grid.vshape, np.nan)
        dsm = np.full(grid.vshape, np.nan)
        if not sheets:
            return DemArrays(dtm, dsm, "no sheets")
        jobs = [(name, kind, url if kind == "dtm" else url.replace("/2020dtm/", "/2020dsm/"))
                for name, url in sheets for kind in (("dtm", "dsm") if want_dsm() else ("dtm",))]

        def one(job):
            if cancel.is_set():
                return job, None
            return job, _get(job[2], 1 << 26)

        missing = []
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            for k, ((name, kind, url), data) in enumerate(ex.map(one, jobs)):
                _cancelled(cancel)
                progress(k / len(jobs), f"Hong Kong LiDAR sheet {name} {kind.upper()}")
                part = _hk_sheet(data, aoi, grid)
                if part is None:
                    missing.append(f"{name} {kind}")
                    continue
                merge_fill(dtm if kind == "dtm" else dsm, part)
        detail = f"{len(sheets)} sheets: " + ", ".join(nm for nm, _ in sheets)
        if missing:
            detail += f"; missing {', '.join(missing)}"
        return DemArrays(dtm, dsm if np.isfinite(dsm).any() else None, detail)


register(GsiDem())
register(HongKongLidar2020())
