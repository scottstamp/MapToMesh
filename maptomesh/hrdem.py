"""HRDEM (1 m / 2 m LiDAR-derived DTM + DSM mosaics) via the CCMEO STAC API.

Faster than raw point clouds for large areas (mountains, islands) because GDAL
reads the Cloud Optimized GeoTIFF overviews matching the grid resolution.
No classification is available: buildings and trees are both "DSM - DTM".
"""

from __future__ import annotations

import os
import threading
from typing import Callable

import numpy as np
import requests

from . import config
from .geo import AOI, Grid

ProgressFn = Callable[[float, str], None]

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.tiff")
os.environ.setdefault("GDAL_HTTP_MULTIRANGE", "YES")
os.environ.setdefault("GDAL_HTTP_MERGE_CONSECUTIVE_RANGES", "YES")
os.environ.setdefault("VSI_CACHE", "TRUE")
os.environ.setdefault("GDAL_CACHEMAX", "512")

COLLECTIONS = ("hrdem-mosaic-1m", "hrdem-mosaic-2m")


def stac_items(bbox: tuple[float, float, float, float], collection: str) -> list[dict]:
    r = requests.get(
        f"{config.STAC_API}/search",
        params={"collections": collection, "bbox": ",".join(f"{v:.6f}" for v in bbox), "limit": 50},
        headers={"User-Agent": config.USER_AGENT}, timeout=60,
    )
    r.raise_for_status()
    return r.json().get("features", [])


def _read_to_grid(url: str, aoi: AOI, grid: Grid) -> np.ndarray:
    import rasterio
    from rasterio.crs import CRS as RCRS
    from rasterio.enums import Resampling
    from rasterio.transform import Affine
    from rasterio.warp import reproject

    H, W = grid.vshape
    # pixel centres on vertices; raster rows run north -> south
    dst_transform = Affine(grid.step, 0, grid.x0 - grid.step / 2, 0, -grid.step, grid.y0 + grid.ny * grid.step + grid.step / 2)
    dst = np.full((H, W), np.nan, dtype=np.float32)
    with rasterio.open(url) as src0:
        res = abs(src0.res[0])
        ovr = src0.overviews(1)
        level = -1
        for k, f in enumerate(ovr):  # coarsest overview still finer than step / 2
            if res * f <= grid.step / 2:
                level = k
    kw = {"overview_level": level} if level >= 0 else {}
    with rasterio.open(url, **kw) as src:
        reproject(
            source=rasterio.band(src, 1), destination=dst,
            src_nodata=src.nodata, dst_nodata=np.nan,
            dst_transform=dst_transform, dst_crs=RCRS.from_wkt(aoi.crs.to_wkt()),
            resampling=Resampling.bilinear if level < 0 else Resampling.average,
        )
    return dst[::-1].astype(np.float64)  # row 0 = south


def fetch_hrdem(aoi: AOI, grid: Grid, progress: ProgressFn, cancel: threading.Event | None = None) -> dict:
    bbox = aoi.lonlat_bbox(margin=grid.step * 2)
    shape = grid.vshape
    dtm = np.full(shape, np.nan); dsm = np.full(shape, np.nan)
    used = []
    for coll in COLLECTIONS:
        progress(0.05, f"Searching {coll}")
        items = stac_items(bbox, coll)
        for k, it in enumerate(items):
            for name, arr in (("dtm", dtm), ("dsm", dsm)):
                if cancel is not None and cancel.is_set():
                    raise RuntimeError("Cancelled")
                href = it["assets"].get(name, {}).get("href")
                if not href:
                    continue
                progress(0.1 + 0.85 * (k + (name == "dsm") * 0.5) / max(len(items), 1),
                         f"Reading {it['id']} {name.upper()}")
                vals = _read_to_grid(href, aoi, grid)
                fill = np.isnan(arr) & np.isfinite(vals)
                arr[fill] = vals[fill]
            used.append(it["id"])
        if np.isfinite(dtm).mean() > 0.98:
            break
    if not np.isfinite(dtm).any():
        raise RuntimeError("No HRDEM coverage for this area.")
    return {"dtm": dtm, "dsm": dsm, "info": {"items": used, "coverage": float(np.isfinite(dtm).mean())}}
