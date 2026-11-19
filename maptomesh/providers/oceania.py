"""New Zealand and Australia elevation models.

* LINZ national 1 m LiDAR DEM + DSM (New Zealand): public COGs (LERC) in the ``nz-elevation`` AWS
  bucket, one file per NZTopo50 map sheet (24 x 36 km in NZTM2000). The sheet list comes from the
  static STAC collection (cached on disk); sheet extents follow the regular Topo50 grid.
* LINZ contour-interpolated 8 m DEM (New Zealand, complete coverage): same layout, gap filler.
* Queensland DEM time-series ImageServer: 0.5-1 m LiDAR DTMs (newest first), read with
  ``exportImage`` in GDA2020 / MGA; the SRTM / bathymetry fill of the service is excluded.
* Geoscience Australia 5 m LiDAR DEM (coastal zone, Murray-Darling floodplains, towns; surveys
  2001-2015): WCS 1.0.0, EPSG:4283 only. Outside its coverage the service returns 0.0 with no
  nodata tag, so exact zeros are treated as no data.

Not used: ELVIS (browser-only API, requester-pays buckets), Vicmap / Landgate (licensed),
NSW 5 m (photogrammetric, slow), MDBA services (no licence stated).
"""

from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path

import numpy as np

from .. import config
from ..dem import DemArrays, DemProvider, register, want_dsm
from ..dem_readers import merge_fill, read_cog, read_service, request_res, service_get, src_bbox

# ---------------------------------------------------------------- New Zealand (LINZ)

LINZ_BUCKET = "https://nz-elevation.s3.ap-southeast-2.amazonaws.com/"
LINZ_INDEX_DIR = config.CACHE_DIR / "linz"
LINZ_INDEX_DAYS = 30
LINZ_LICENCE = "CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)"
NZ_REGIONS = ((172.5, -41.8, 178.7, -34.0),    # North Island
              (166.3, -47.6, 174.5, -40.3))    # South Island, Stewart Island

# NZTopo50 sheets: two letters for the row (north -> south, I and O skipped), two digits for the
# column (west -> east). Calibrated on BQ31 = NZTM (1732000, 5406000) - (1756000, 5442000) and
# checked against the COG bounds of AS21, BD35, BW21, CG09, CJ10.
_TOPO50_LETTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ"
_SHEET_W, _SHEET_H = 24_000, 36_000
_REF_ROW, _REF_COL, _REF_X0, _REF_Y1 = 24 + _TOPO50_LETTERS.index("Q"), 31, 1_732_000, 5_442_000

_index_lock = threading.Lock()


def topo50_bbox(code: str) -> tuple[int, int, int, int]:
    """NZTM2000 extent (xmin, ymin, xmax, ymax) of a Topo50 sheet such as ``BQ31``."""
    r = _TOPO50_LETTERS.index(code[0]) * 24 + _TOPO50_LETTERS.index(code[1])
    x0 = _REF_X0 + (int(code[2:]) - _REF_COL) * _SHEET_W
    y1 = _REF_Y1 - (r - _REF_ROW) * _SHEET_H
    return x0, y1 - _SHEET_H, x0 + _SHEET_W, y1


def topo50_sheets(bbox: tuple[float, float, float, float]) -> list[str]:
    """Topo50 sheet codes intersecting an NZTM2000 bbox."""
    x0, y0, x1, y1 = bbox
    c0 = _REF_COL + math.floor((x0 - _REF_X0) / _SHEET_W)
    c1 = _REF_COL + math.floor((x1 - _REF_X0) / _SHEET_W)
    r0 = _REF_ROW + math.floor((_REF_Y1 - y1) / _SHEET_H)
    r1 = _REF_ROW + math.floor((_REF_Y1 - y0) / _SHEET_H)
    out = []
    for r in range(max(r0, 0), min(r1, 24 * 24 - 1) + 1):
        for c in range(max(c0, 0), min(c1, 99) + 1):
            out.append(_TOPO50_LETTERS[r // 24] + _TOPO50_LETTERS[r % 24] + f"{c:02d}")
    return out


def linz_sheet_index(prefix: str) -> dict[str, tuple[int, int, int, int]]:
    """{sheet code: NZTM bbox} of a LINZ national collection, e.g.
    ``new-zealand/new-zealand/dem_1m/2193/``. Read from the static STAC collection.json once and
    kept under ``CACHE_DIR/linz`` (refreshed after 30 days; a stale copy is used if LINZ is down)."""
    path = LINZ_INDEX_DIR / (prefix.strip("/").replace("/", "_") + ".json")
    with _index_lock:
        stale = None
        if path.exists():
            try:
                stale = {k: tuple(v) for k, v in json.loads(path.read_text())["sheets"].items()}
            except (ValueError, KeyError, TypeError):
                stale = None
            if stale and time.time() - path.stat().st_mtime < LINZ_INDEX_DAYS * 86400:
                return stale
        try:
            coll = json.loads(service_get(LINZ_BUCKET + prefix + "collection.json", timeout=60, cache=False))
        except Exception:  # noqa: BLE001 - offline / LINZ down: keep using the old index
            if stale:
                return stale
            raise
        codes = sorted(Path(lk["href"]).stem for lk in coll.get("links", []) if lk.get("rel") == "item")
        sheets = {c: topo50_bbox(c) for c in codes}
        LINZ_INDEX_DIR.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"title": coll.get("title"), "updated": coll.get("updated"),
                                   "licence": coll.get("license"), "sheets": sheets}))
        tmp.replace(path)
        return sheets


def _linz_read(layers: list[tuple[str, np.ndarray]], aoi, grid, progress, cancel, name: str) -> list[str]:
    """Read every sheet of each (collection prefix, output array) that touches the AOI."""
    bbox = src_bbox(aoi, grid, 2193)
    wanted = topo50_sheets(bbox)
    used: list[str] = []
    jobs = []
    for prefix, arr in layers:
        index = linz_sheet_index(prefix)
        jobs += [(prefix, arr, s) for s in wanted if s in index]
    for k, (prefix, arr, sheet) in enumerate(jobs):
        if cancel.is_set():
            raise RuntimeError("Cancelled")
        progress(k / max(len(jobs), 1), f"{name} sheet {sheet} {prefix.split('/')[-3]}")
        merge_fill(arr, read_cog(f"{LINZ_BUCKET}{prefix}{sheet}.tiff", aoi, grid))
        if sheet not in used:
            used.append(sheet)
    return used


class Linz1m(DemProvider):
    key = "linz_1m"
    label = "LINZ LiDAR 1 m (New Zealand)"
    res_m = 1.0
    has_dsm = True
    # The national 1 m DEM / DSM name LINZ as sole licensor (the regional survey collections credit
    # their councils separately; the merged product is credited to LINZ).
    attribution = ("Sourced from Toitū Te Whenua Land Information New Zealand (New Zealand LiDAR 1m DEM "
                   "and DSM) and licensed for reuse under CC BY 4.0.")
    licence = LINZ_LICENCE
    regions = NZ_REGIONS
    max_area_km2 = 2500.0
    sample = (-41.2900, 174.7550, 3000)   # Wellington: CBD, Botanic Garden, town belt

    DEM = "new-zealand/new-zealand/dem_1m/2193/"
    DSM = "new-zealand/new-zealand/dsm_1m/2193/"

    def fetch(self, aoi, grid, progress, cancel):
        dtm = np.full(grid.vshape, np.nan)
        dsm = np.full(grid.vshape, np.nan)
        used = _linz_read([(self.DEM, dtm)] + ([(self.DSM, dsm)] if want_dsm() else []),
                          aoi, grid, progress, cancel, "LINZ 1 m")
        dsm = np.where(np.isfinite(dtm), dsm, np.nan)   # surface only where the ground is known
        return DemArrays(dtm, dsm if np.isfinite(dsm).any() else None,
                         "NZTopo50 sheets " + ", ".join(used) if used else "")


class Linz8m(DemProvider):
    key = "linz_8m"
    label = "LINZ contour DEM 8 m (New Zealand)"
    res_m = 8.0
    attribution = ("Sourced from Toitū Te Whenua Land Information New Zealand (New Zealand Contour-Interpolated "
                   "8m DEM, originally created by Geographx) and licensed for reuse under CC BY 4.0.")
    licence = LINZ_LICENCE
    regions = NZ_REGIONS
    max_area_km2 = 20_000.0
    sample = (-39.2800, 175.5700, 8000)   # Ruapehu

    DEM = "new-zealand/new-zealand-contour/dem_8m/2193/"

    def fetch(self, aoi, grid, progress, cancel):
        dtm = np.full(grid.vshape, np.nan)
        used = _linz_read([(self.DEM, dtm)], aoi, grid, progress, cancel, "LINZ 8 m")
        return DemArrays(dtm, None, "NZTopo50 sheets " + ", ".join(used) if used else "")


# ---------------------------------------------------------------- Queensland

QLD_SERVICE = "https://spatial-img.information.qld.gov.au/arcgis/rest/services/Elevation/DEM_TimeSeries_AllUsers/ImageServer"
# product_type 8 = LiDAR DTM projects (0.5-1 m; a few coarse ones excluded by lowps). Excludes the
# SRTM 30 m and bathymetry layers the service mosaics underneath, so Copernicus / GA take over there.
QLD_WHERE = "product_type=8 AND lowps<=2"
QLD_MOSAIC = json.dumps({"mosaicMethod": "esriMosaicAttribute", "sortField": "ZOrder", "sortValue": "0",
                         "ascending": True, "mosaicOperation": "MT_FIRST", "where": QLD_WHERE},
                        separators=(",", ":"))
QLD_MAX_PX = 2048   # service limit 4096 x 2048


class QldDem(DemProvider):
    key = "qld_lidar_1m"
    label = "Queensland LiDAR 1 m"
    res_m = 1.0
    attribution = ("Includes material © State of Queensland (Department of Natural Resources and Mines, "
                   "Manufacturing, and Regional and Rural Development) and © Commonwealth of Australia "
                   "(Geoscience Australia), Queensland Elevation DEM Public Time Series Service, CC BY 3.0 AU.")
    licence = "CC BY 3.0 AU (https://creativecommons.org/licenses/by/3.0/au/)"
    regions = ((138.0, -26.0, 153.6, -9.0),      # north of the SA / NT corner
               (141.0, -29.2, 153.6, -26.0))     # south-west to the NSW border
    max_area_km2 = 2500.0
    sample = (-27.4757, 152.9575, 3000)   # Mt Coot-tha, Brisbane

    def _projects(self, epsg: int, bbox) -> str:
        """Names / years of the LiDAR projects under the AOI, for the job log (best effort)."""
        try:
            body = service_get(QLD_SERVICE + "/query", params={
                "geometry": ",".join(f"{v:.1f}" for v in bbox), "geometryType": "esriGeometryEnvelope",
                "inSR": epsg, "spatialRel": "esriSpatialRelIntersects", "where": QLD_WHERE,
                "outFields": "name,year,zorder", "returnGeometry": "false", "f": "json"}, timeout=60)
            feats = [f["attributes"] for f in json.loads(body).get("features", [])]
            feats.sort(key=lambda a: abs(a.get("zorder") or 0))
            return ", ".join(f"{a['name']}" for a in feats[:8])
        except Exception:  # noqa: BLE001
            return ""

    def fetch(self, aoi, grid, progress, cancel):
        zone = int(math.floor((aoi.lon + 180) / 6)) + 1
        epsg = 7800 + max(54, min(56, zone))      # GDA2020 / MGA zone 54-56
        bbox = src_bbox(aoi, grid, epsg)
        res = request_res(self.res_m, grid)
        n = [0]

        def tile(b, w, h):
            n[0] += 1
            progress(0.1, f"Queensland LiDAR DTM request {n[0]}")
            return service_get(QLD_SERVICE + "/exportImage", params={
                "bbox": ",".join(f"{v:.2f}" for v in b), "bboxSR": epsg, "imageSR": epsg,
                "size": f"{w},{h}", "format": "tiff", "pixelType": "F32", "noData": -9999,
                "noDataInterpretation": "esriNoDataMatchAny", "interpolation": "RSP_BilinearInterpolation",
                "mosaicRule": QLD_MOSAIC, "f": "image"}, timeout=180)

        dtm = read_service(tile, bbox, res, QLD_MAX_PX, aoi, grid, nodata=-9999.0, cancel=cancel)
        detail = f"EPSG:{epsg} at {res:.1f} m, {n[0]} request(s)"
        if np.isfinite(dtm).any():
            proj = self._projects(epsg, bbox)
            detail += f"; projects (newest first): {proj}" if proj else ""
        return DemArrays(dtm, None, detail)


# ---------------------------------------------------------------- Australia (Geoscience Australia)

GA_WCS = "https://services.ga.gov.au/gis/services/DEM_LiDAR_5m_2025/MapServer/WCSServer"
GA_MAX_PX = 2000


class Ga5m(DemProvider):
    key = "ga_lidar_5m"
    label = "Geoscience Australia LiDAR 5 m"
    res_m = 5.0
    attribution = ("© Commonwealth of Australia (Geoscience Australia) 2025, Digital Elevation Model (DEM) of "
                   "Australia derived from LiDAR 5 Metre Grid, CC BY 4.0.")
    licence = "CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)"
    regions = ((112.9, -39.2, 153.7, -9.8),      # mainland
               (143.5, -43.7, 148.6, -39.4))     # Tasmania, Bass Strait islands
    max_area_km2 = 10_000.0
    sample = (-42.8800, 147.3000, 3000)   # Hobart, foot of kunanyi / Mt Wellington

    def fetch(self, aoi, grid, progress, cancel):
        bbox = src_bbox(aoi, grid, 4283)                     # GDA94 lon / lat (the only CRS that works)
        res = request_res(self.res_m, grid, scale=1 / 111_320.0)   # degrees; finer than needed in x
        n = [0]

        def tile(b, w, h):
            n[0] += 1
            progress(0.1, f"GA 5 m WCS request {n[0]}")
            return service_get(GA_WCS, params={
                "service": "WCS", "version": "1.0.0", "request": "GetCoverage", "coverage": "1",
                "crs": "EPSG:4283", "bbox": ",".join(f"{v:.7f}" for v in b), "width": w, "height": h,
                "format": "GeoTIFF"}, timeout=180)

        # nearest-neighbour server side; 0.0 = outside the LiDAR coverage (no nodata tag)
        dtm = read_service(tile, bbox, res, GA_MAX_PX, aoi, grid, src_crs=4283, nodata=0.0, cancel=cancel)
        return DemArrays(dtm, None, f"WCS DEM_LiDAR_5m_2025, {res * 111_320:.1f} m, {n[0]} request(s)")


register(Linz1m())
register(Linz8m())
register(QldDem())
register(Ga5m())
