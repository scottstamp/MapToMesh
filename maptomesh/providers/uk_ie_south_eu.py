"""UK, Ireland, Spain and South Tyrol: national / regional LiDAR elevation models.

DEMs (``best_dems`` asks the finest first and fills only remaining gaps):

* Scotland - Scottish Remote Sensing Portal (SRSP): 0.5 m DTM / DSM GeoTIFFs (mostly COGs) in the
  public ``srsp-open-data`` S3 bucket, found through the portal's product search API. National LiDAR
  Programme (2025-27) first, then the newest phases. Collections whose use constraints are not plain
  OGL v3 (Non-Commercial Government Licence; Phase II / III "Fugro retain the commercial copyright")
  are only used where no unrestricted collection has data, and then add a licence warning to the
  model (classified from the collection metadata). Phase I / II-style 1 m tiles without
  overviews are only read for small areas (``SRSP_HEAVY_MAX_KM2``).
* South Tyrol (IT) - Provincia di Bolzano WCS: DTM / DSM 0.5 m (settlements and valley floors) and
  2.5 m (whole province), CC0.
* Wales - DataMapWales: one all-Wales 1 m DTM COG and one DSM COG (2020-23 survey), OGL v3.
* England - Environment Agency LIDAR Composite DTM + first-return DSM 1 m (2022 composite) via the
  Defra WCS (server-side clip and scaling), OGL v3.
* Ireland - Geological Survey Ireland open LiDAR tiles (GSI, OPW, TII, ...): 1-2 m DTM / DSM in small
  zips (GeoTIFF or ASCII grid) listed by ArcGIS coverage layers, CC BY 4.0. Patchy coverage.
* Spain - IGN MDT05 5 m through the INSPIRE WCS (integer metres; above Copernicus as a fallback).
* Tuscany (IT) - Regione Toscana Geoscopio WMS: 1 m LiDAR DTM 2019-21 (Italian Remote Sensing, partial
  coverage) as float32 GeoTIFF from GetMap; 0 outside the surveys. CC BY-NC-SA: a non-commercial
  warning goes with the model, and it ranks after unrestricted models of equal detail.

Not implemented: CNIG LiDAR / MDT02 (portal only), Portugal DGT (login), Northern Ireland (blocked
downloads), MASE PST (web app), Sardinia / Emilia-Romagna WCS (unreliable).
"""

from __future__ import annotations

import io
import json
import math
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ..dem import DemArrays, DemProvider, register, want_dsm
from ..dem_readers import (merge_fill, read_cog, reproject_array, request_res, service_get, service_tiles,
                           src_bbox)

WORKERS = 4   # parallel requests per provider (be gentle)

OGL = "Open Government Licence v3.0 (https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/)"
CC_BY = "Creative Commons Attribution 4.0 (https://creativecommons.org/licenses/by/4.0/)"
CC0 = "CC0 1.0 Public Domain Dedication (https://creativecommons.org/publicdomain/zero/1.0/)"

WALES_REGIONS = ((-5.40, 51.35, -2.62, 52.10), (-4.85, 52.00, -2.70, 53.45))
ENGLAND_REGIONS = ((-6.50, 49.85, 1.80, 51.38), (-3.25, 51.38, 1.80, 53.55), (-3.65, 53.55, 0.20, 55.82))
SCOTLAND_REGIONS = ((-7.75, 54.60, -0.70, 59.40), (-2.20, 59.45, -0.70, 60.90))
IRELAND_REGIONS = ((-10.70, 51.35, -5.95, 55.45),)
SPAIN_REGIONS = ((-9.35, 35.90, 3.35, 43.80), (1.15, 38.60, 4.35, 40.10), (-18.20, 27.60, -13.40, 29.45),
                 (-5.40, 35.85, -5.25, 35.93), (-2.98, 35.26, -2.92, 35.32))
SOUTH_TYROL_REGIONS = ((10.38, 46.21, 12.48, 47.10),)


def _cancelled(cancel) -> None:
    if cancel is not None and cancel.is_set():
        raise RuntimeError("Cancelled")


def _nanify(a: np.ndarray, nodata=None, zero_is_nodata: bool = False, min_valid: float = -1e5) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32).copy()
    bad = ~np.isfinite(a) | (np.abs(a) > 1e5) | (a < min_valid)   # -3.4e38 fill values, -32767, ...
    if nodata is not None and np.isfinite(nodata):
        bad |= a == np.float32(nodata)
    if zero_is_nodata:
        bad |= a == 0
    a[bad] = np.nan
    return a


def _raster_to_grid(data: bytes, aoi, grid, src_crs=None, ext: str = ".tif", zero_is_nodata: bool = False,
                    nodata=None, min_valid: float = -1e5) -> np.ndarray:
    """Raster bytes (GeoTIFF / ASCII grid) -> vertex grid, with every kind of fill value as NaN."""
    from ..dem_readers import read_raster_bytes
    src = read_raster_bytes(data, ext)
    arr = _nanify(src.arr, src.nodata if nodata is None else nodata, zero_is_nodata, min_valid)
    crs = src_crs if src_crs is not None else src.crs
    if crs is None:
        raise ValueError("raster has no CRS")
    return reproject_array(arr, src.transform, crs, aoi, grid, nodata=None)


def _snap(b: tuple, res: float) -> tuple:
    x0, y0, x1, y1 = b
    return (math.floor(x0 / res) * res, math.floor(y0 / res) * res,
            math.ceil(x1 / res) * res, math.ceil(y1 / res) * res)


def _read_wcs(url: str, coverage: str, crs: int, native: float, aoi, grid, cancel, axes=("E", "N"),
              scaling: str = "geoserver", max_px: int = 2000, zero_is_nodata: bool = False,
              extra: dict | None = None) -> np.ndarray:
    """WCS 2.0.1 GetCoverage over the AOI, split into <= max_px requests, scaled server-side when the
    grid is coarser than the data, all on the vertex grid."""
    res = request_res(native, grid)
    bbox = _snap(src_bbox(aoi, grid, crs), native)
    out = np.full(grid.vshape, np.nan)
    for (x0, y0, x1, y1), w, h in service_tiles(bbox, res, max_px):
        _cancelled(cancel)
        p = {"service": "WCS", "version": "2.0.1", "request": "GetCoverage", "CoverageId": coverage,
             "format": "image/tiff", "subset": [f"{axes[0]}({x0:.3f},{x1:.3f})", f"{axes[1]}({y0:.3f},{y1:.3f})"]}
        if res > native * 1.0001:
            if scaling == "geoserver":
                p["SCALESIZE"] = (f"http://www.opengis.net/def/axis/OGC/1/i({w}),"
                                  f"http://www.opengis.net/def/axis/OGC/1/j({h})")
            else:   # MapServer: axis labels
                p["scalesize"] = f"{axes[0].lower()}({w}),{axes[1].lower()}({h})"
        p.update(extra or {})
        data = service_get(url, p, timeout=180)
        merge_fill(out, _raster_to_grid(data, aoi, grid, src_crs=crs, zero_is_nodata=zero_is_nodata))
    return out


def _lonlat_to_cells(aoi, grid, lon, lat):
    """Lon/lat points -> (col, row) on the vertex grid (row 0 = south)."""
    from pyproj import Transformer
    tf = Transformer.from_crs(4326, aoi.crs, always_xy=True)
    x, y = tf.transform(np.asarray(lon, float), np.asarray(lat, float))
    return (x - grid.x0) / grid.step, (y - grid.y0) / grid.step


def _has_gap(arr: np.ndarray, aoi, grid, lon, lat) -> bool:
    """Whether ``arr`` still has NaNs inside the lon/lat bounding box of a tile footprint."""
    c, r = _lonlat_to_cells(aoi, grid, lon, lat)
    H, W = arr.shape
    c0, c1 = max(0, int(math.floor(np.min(c))) - 1), min(W - 1, int(math.ceil(np.max(c))) + 1)
    r0, r1 = max(0, int(math.floor(np.min(r))) - 1), min(H - 1, int(math.ceil(np.max(r))) + 1)
    if c0 > c1 or r0 > r1:
        return False
    return bool(np.isnan(arr[r0:r1 + 1, c0:c1 + 1]).any())


# ---------------------------------------------------------------------------------------------- Wales

class WalesLidar(DemProvider):
    key = "wales_lidar_1m"
    label = "Welsh Government LiDAR 1 m 2020-23 (Wales)"
    res_m = 1.0
    has_dsm = True
    attribution = ("Contains Welsh Government LiDAR data (2020-23, DataMapWales) licensed under the "
                   "Open Government Licence v3.0.")
    licence = OGL
    regions = WALES_REGIONS
    max_area_km2 = 2500.0
    sample = (53.0685, -4.0763, 3000)   # Yr Wyddfa / Snowdon

    DTM = "https://dmwproductionblob.blob.core.windows.net/cogs/lidar/wales_dtm_32bit_cog.tif"
    DSM = "https://dmwproductionblob.blob.core.windows.net/cogs/lidar/wales_dsm_32bit_cog.tif"

    def fetch(self, aoi, grid, progress, cancel):
        progress(0.0, "Wales LiDAR DTM")
        dtm = read_cog(self.DTM, aoi, grid)
        _cancelled(cancel)
        dsm = None
        if want_dsm():
            progress(0.5, "Wales LiDAR DSM")
            dsm = read_cog(self.DSM, aoi, grid)
        return DemArrays(dtm, dsm, "wales_dtm_32bit_cog + wales_dsm_32bit_cog")


# -------------------------------------------------------------------------------------------- England

class EnglandEaLidar(DemProvider):
    key = "ea_lidar_composite_1m"
    read_kind = "service"   # download estimates (maptomesh.estimate)
    label = "Environment Agency LiDAR 1 m (England)"
    res_m = 1.0
    has_dsm = True
    attribution = ("Contains public sector information licensed under the Open Government Licence v3.0. "
                   "© Environment Agency copyright and/or database right 2022. All rights reserved.")
    licence = OGL
    regions = ENGLAND_REGIONS
    max_area_km2 = 1000.0
    sample = (51.3811, -2.3590, 3000)   # Bath

    BASE = "https://environment.data.gov.uk/spatialdata"
    DTM = ("lidar-composite-digital-terrain-model-dtm-1m",
           "13787b9a-26a4-4775-8523-806d13af58fc__Lidar_Composite_Elevation_DTM_1m")
    DSM = ("lidar-composite-digital-surface-model-first-return-dsm-1m",
           "df4e3ec3-315e-48aa-aaaf-b5ae74d7b2bb__Lidar_Composite_Elevation_FZ_DSM_1m")

    def fetch(self, aoi, grid, progress, cancel):
        out = []
        for k, (name, (slug, cov)) in enumerate((("DTM", self.DTM), ("first-return DSM", self.DSM))[:2 if want_dsm() else 1]):
            progress(k / 2, f"Environment Agency LiDAR {name}")
            # outside the survey footprint the WCS answers with zeros and no nodata tag
            out.append(_read_wcs(f"{self.BASE}/{slug}/wcs", cov, 27700, 1.0, aoi, grid, cancel,
                                 zero_is_nodata=True, extra={"geotiff:compression": "Deflate"}))
        return DemArrays(out[0], out[1] if len(out) > 1 else None, "EA LIDAR Composite 2022 DTM + first-return DSM (WCS)")


# ------------------------------------------------------------------------------------------- Scotland

SRSP_API = "https://api.remotesensing.data.gov.scot"


def _srsp_priority(name: str) -> float | None:
    """Lower = newer / better. None = don't use (25 cm duplicate of the 50 cm set)."""
    if "/25cm" in name:
        return None
    if "national-lidar-programme" in name:
        return 0
    if "/coastal/" in name:
        return 1
    if "orkney" in name:
        return 2
    if "outerheb" in name:
        return 3
    m = re.search(r"/phase-(\d+)/", name)
    if m:
        return 10 - int(m.group(1))          # phase 6 -> 4 ... phase 1 -> 9
    if "/hes/" in name:
        return 12
    return 5


SRSP_HEAVY = ("/phase-1/", "/phase-2/", "orkney")   # 1 m tiles without overviews / stripped files
SRSP_HEAVY_MAX_KM2 = 12.0   # those are read at full resolution: ~6 MB per km2 per model


# Collections with licence restrictions are used too (the user's call), but only where no
# unrestricted collection has data (priority offsets), and the model then carries a warning.
SRSP_RESTRICTED_OFFSET = {"commercial copyright": 100.0, "non-commercial": 200.0}


def _srsp_restriction(uc: str) -> str | None:
    if "non-commercial" in uc or "open government licence" not in uc:
        return "non-commercial"
    if "commercial copyright" in uc:
        return "commercial copyright"
    return None


def _srsp_note(name: str, restriction: str) -> str:
    short = name.split("lidar/")[-1].rsplit("/", 1)[0]   # one note per collection, not per DTM / DSM
    if restriction == "non-commercial":
        return (f"Scottish LiDAR collection '{short}' is under the Non-Commercial Government Licence (or a "
                "non-OGL licence): prints of this area must not be sold or used commercially; terms: "
                "https://www.nationalarchives.gov.uk/doc/non-commercial-government-licence/version/2/")
    return (f"Scottish LiDAR collection '{short}' is released under the Open Government Licence, but its "
            "terms state that the survey contractor (Fugro) retains the commercial copyright: check the "
            "terms before selling prints of this area: https://remotesensingdata.gov.scot/")


def _srsp_collections() -> list[tuple[float, str, str, str | None]]:
    """(priority, kind, name, restriction) of every SRSP LiDAR DTM / DSM collection; restricted
    collections sort after every unrestricted one."""
    data = json.loads(service_get(f"{SRSP_API}/search/collection/scotland-gov/*", timeout=90))
    out = []
    for c in data.get("result", []):
        name = c.get("name", "")
        if not name.startswith("scotland-gov/lidar/") or "/laz" in name:
            continue
        kind = "dtm" if "/dtm" in name else "dsm" if "/dsm" in name else None
        if kind is None:
            continue
        uc = json.dumps(c.get("metadata") or {}).lower()
        restriction = _srsp_restriction(uc)
        pr = _srsp_priority(name)
        if pr is not None:
            out.append((pr + SRSP_RESTRICTED_OFFSET.get(restriction, 0.0), kind, name, restriction))
    return sorted(out, key=lambda c: c[:3])


def _srsp_search(collections: list[str], bbox: tuple) -> list[dict]:
    w, s, e, n = bbox
    wkt = f"POLYGON(({w} {s},{e} {s},{e} {n},{w} {n},{w} {s}))"
    items, off, limit = [], 0, 500
    while True:
        body = json.dumps({"collections": collections, "footprint": wkt, "limit": limit, "offset": off})
        res = json.loads(service_get(f"{SRSP_API}/search/product", method="POST", data=body, timeout=90,
                                     headers={"Content-Type": "application/json"})).get("result", [])
        items += res
        if len(res) < limit or off > 5000:
            return items
        off += limit


def _footprint_lonlat(item: dict):
    pts = np.asarray([p for poly in item.get("footprint", {}).get("coordinates", []) for ring in poly
                      for p in ring], float)
    return (pts[:, 0], pts[:, 1]) if len(pts) else (None, None)


class ScotlandSrsp(DemProvider):
    key = "srsp_lidar_50cm"
    label = "Scottish Remote Sensing Portal LiDAR 0.5-1 m (Scotland)"
    res_m = 0.5
    has_dsm = True
    attribution = ("Contains public sector information licensed under the Open Government Licence v3.0: "
                   "Crown copyright Scottish Government, SEPA, Scottish Water and partners "
                   "(Scottish Remote Sensing Portal LiDAR).")
    licence = OGL
    regions = SCOTLAND_REGIONS
    max_area_km2 = 100.0
    sample = (55.9441, -3.1618, 3000)   # Arthur's Seat, Edinburgh

    def fetch(self, aoi, grid, progress, cancel):
        area = aoi.width_m * aoi.height_m / 1e6
        colls = [c for c in _srsp_collections() if area <= SRSP_HEAVY_MAX_KM2 or not any(h in c[2] for h in SRSP_HEAVY)]
        if not colls:
            raise RuntimeError("no SRSP collections")
        bbox = aoi.lonlat_bbox(margin=grid.step * 3)
        items = _srsp_search([c[2] for c in colls], bbox)
        prio = {name: pr for pr, _, name, _r in colls}
        restricted = {name: r for _p, _k, name, r in colls if r}
        notes: list[str] = []
        groups: dict[float, dict[str, list]] = {}
        for it in items:
            name = it.get("collectionName", "")
            if name not in prio:
                continue
            kind = "dtm" if "/dtm" in name else "dsm"
            groups.setdefault(prio[name], {"dtm": [], "dsm": []})[kind].append(it)
        dtm = np.full(grid.vshape, np.nan)
        dsm = np.full(grid.vshape, np.nan)
        used = []
        wants = want_dsm()
        steps = max(1, len(groups))
        with ThreadPoolExecutor(WORKERS) as ex:
            for g, pr in enumerate(sorted(groups)):
                for kind, arr in (("dtm", dtm), ("dsm", dsm))[:2 if wants else 1]:
                    todo = []
                    for it in groups[pr][kind]:
                        lon, lat = _footprint_lonlat(it)
                        if lon is None or _has_gap(arr, aoi, grid, lon, lat):
                            url = it["data"]["product"]["http"]["url"]
                            todo.append(url)
                    _cancelled(cancel)
                    if not todo:
                        continue
                    progress(g / steps, f"SRSP {groups[pr][kind][0]['collectionName'].split('lidar/')[-1]}: "
                                        f"{len(todo)} tiles")
                    cname = groups[pr][kind][0]["collectionName"]
                    for url, part in zip(todo, ex.map(lambda u: read_cog(u, aoi, grid), todo)):
                        _cancelled(cancel)
                        before = int(np.isfinite(arr).sum())
                        merge_fill(arr, part)
                        used.append(url.rsplit("/", 1)[-1].rsplit(".", 1)[0])
                        if cname in restricted and int(np.isfinite(arr).sum()) > before:
                            note = _srsp_note(cname, restricted[cname])
                            if note not in notes:
                                notes.append(note)
                if np.isfinite(dtm).all() and (not wants or np.isfinite(dsm).all()):
                    break
        return DemArrays(dtm, dsm if wants else None, ", ".join(used), licence_notes=notes)


# -------------------------------------------------------------------------------------------- Ireland

GSI_SERVER = "https://gsi.geodata.gov.ie/server/rest/services/Lidar"
# (coverage layer, ArcGIS layer id), best first: 1 m GSI / heritage surveys, then 2 m OPW / TII
GSI_LAYERS = (("IE_GSI_LiDAR_Coverage_GSI_DCHG_DP_IE26_ITM", 0),
              ("IE_GSI_LiDAR_Coverage_NYU_Dublin_IE26_ITM", 4),
              ("IE_GSI_LiDAR_Coverage_OPW_NASC_IE26_ITM", 3),
              ("IE_GSI_LiDAR_Coverage_OPW_Cork_IE26_ITM", 2),
              ("IE_GSI_LiDAR_Coverage_TII_IE26_ITM", 0),
              ("IE_GSI_LiDAR_Coverage_OPW_IE26_ITM", 3))
GSI_MAX_ZIP = 40_000_000   # skip the huge 25 cm Westmeath-style archives


def _gsi_tiles(layer: str, lid: int, bbox: tuple) -> list[dict]:
    w, s, e, n = bbox
    p = {"geometry": f"{w},{s},{e},{n}", "geometryType": "esriGeometryEnvelope", "inSR": "4326",
         "spatialRel": "esriSpatialRelIntersects", "outFields": "DATA_URL,RESOLUTION,DATECAPTUR",
         "returnGeometry": "false", "f": "json"}
    data = json.loads(service_get(f"{GSI_SERVER}/{layer}/MapServer/{lid}/query", p, timeout=60))
    return [f["attributes"] for f in data.get("features", []) if f.get("attributes", {}).get("DATA_URL")]


def _gsi_members(blob: bytes) -> dict[str, tuple[bytes, str]]:
    """{'dtm'|'dsm': (raster bytes, extension)} from a GSI zip (GeoTIFF or ESRI ASCII grid)."""
    out = {}
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        for info in z.infolist():
            base = info.filename.rsplit("/", 1)[-1].lower()
            ext = "." + base.rsplit(".", 1)[-1] if "." in base else ""
            if ext not in (".tif", ".tiff", ".asc"):
                continue
            kind = "dsm" if "dsm" in base else "dtm"   # TII zips hold a single (terrain) model
            if kind not in out:
                out[kind] = (z.read(info), ext)
    return out


class IrelandGsi(DemProvider):
    key = "gsi_lidar_ie"
    read_kind = "file"   # download estimates (maptomesh.estimate)
    label = "GSI / OPW open LiDAR 1-2 m (Ireland, partial)"
    res_m = 1.0
    has_dsm = True
    attribution = ("Contains Irish Public Sector Data (Geological Survey Ireland and partners: Office of Public "
                   "Works, Transport Infrastructure Ireland, Department of Culture, Heritage and the Gaeltacht, "
                   "Discovery Programme) licensed under a Creative Commons Attribution 4.0 International "
                   "(CC BY 4.0) licence.")
    licence = CC_BY
    regions = IRELAND_REGIONS
    max_area_km2 = 30.0
    sample = (53.4955, -9.0399, 3000)   # Galway 1 m block (GSI / Heritage)

    def fetch(self, aoi, grid, progress, cancel):
        from ..httpcache import RANGE_CACHE
        bbox = aoi.lonlat_bbox(margin=grid.step * 3)
        dtm = np.full(grid.vshape, np.nan)
        dsm = np.full(grid.vshape, np.nan)
        used = []

        def get(url):
            if RANGE_CACHE.size(url) > GSI_MAX_ZIP:
                return url, None
            return url, service_get(url, timeout=180)

        with ThreadPoolExecutor(WORKERS) as ex:
            for k, (layer, lid) in enumerate(GSI_LAYERS):
                _cancelled(cancel)
                tiles = _gsi_tiles(layer, lid, bbox)
                if not tiles:
                    continue
                progress(k / len(GSI_LAYERS), f"GSI LiDAR {layer.split('Coverage_')[-1]}: {len(tiles)} tiles")
                for url, blob in ex.map(get, [t["DATA_URL"] for t in tiles]):
                    _cancelled(cancel)
                    if blob is None or not blob.startswith(b"PK"):
                        continue
                    for kind, (data, ext) in _gsi_members(blob).items():
                        arr = dtm if kind == "dtm" else dsm
                        # TII tiles fill with -99 but tag 0 as nodata; nothing in Ireland is below -50 m
                        merge_fill(arr, _raster_to_grid(data, aoi, grid, src_crs=2157, ext=ext, min_valid=-50.0))
                    used.append(url.rsplit("/", 1)[-1])
                if np.isfinite(dtm).all():
                    break
        return DemArrays(dtm, dsm, ", ".join(used))


# ---------------------------------------------------------------------------------------------- Spain

class SpainIgnMdt05(DemProvider):
    key = "ign_es_mdt05"
    read_kind = "service"   # download estimates (maptomesh.estimate)
    label = "IGN MDT05 5 m (Spain)"
    res_m = 5.0
    attribution = "MDT05 CC BY 4.0 scne.es - © Instituto Geográfico Nacional de España"
    licence = "CC BY 4.0, Orden FOM/2807/2015 (https://www.ign.es/resources/licencia/Condiciones_licenciaUso_IGN.pdf)"
    regions = SPAIN_REGIONS
    max_area_km2 = 2500.0
    sample = (41.5932, 1.8376, 5000)   # Montserrat

    URL = "https://servicios.idee.es/wcs-inspire/mdt"

    def fetch(self, aoi, grid, progress, cancel):
        canary = aoi.lon < -13.0
        crs, cov = (4083, "Elevacion4083_5") if canary else (25830, "Elevacion25830_5")
        progress(0.0, "IGN MDT05 (WCS)")
        # int16 metres; 0 = no data (sea, outside Spain)
        dtm = _read_wcs(self.URL, cov, crs, 5.0, aoi, grid, cancel, axes=("x", "y"), scaling="mapserver",
                        zero_is_nodata=True)
        return DemArrays(dtm, None, cov)


# ---------------------------------------------------------------------------------------- South Tyrol

BZ_WCS = "https://geoservices9.civis.bz.it/geoserver/ows"


class _SouthTyrol(DemProvider):
    attribution = ("Autonome Provinz Bozen - Südtirol / Provincia autonoma di Bolzano - Alto Adige, "
                   "Amt für Landesplanung und Kartografie (CC0 1.0)")
    licence = CC0
    regions = SOUTH_TYROL_REGIONS
    has_dsm = True
    suffix = ""

    def fetch(self, aoi, grid, progress, cancel):
        out = []
        for k, model in enumerate(("DigitalTerrainModel", "DigitalElevationModel")[:2 if want_dsm() else 1]):   # DTM, DSM
            cov = f"p_bz-Elevation__{model}-{self.suffix}"
            progress(k / 2, f"South Tyrol {cov}")
            out.append(_read_wcs(BZ_WCS, cov, 25832, self.res_m, aoi, grid, cancel,
                                 extra={"geotiff:compression": "Deflate"}))
        return DemArrays(out[0], out[1] if len(out) > 1 else None, f"Bolzano WCS DTM/DSM {self.suffix}")


class SouthTyrol05(_SouthTyrol):
    key = "bz_dtm_05m"
    read_kind = "service"   # download estimates (maptomesh.estimate)
    label = "South Tyrol LiDAR DTM/DSM 0.5 m (Italy, valleys)"
    res_m = 0.5
    suffix = "0.5m"
    max_area_km2 = 150.0
    sample = (46.4844, 11.3256, 3000)   # Bolzano / Bozen


class SouthTyrol25(_SouthTyrol):
    key = "bz_dtm_25m"
    read_kind = "service"   # download estimates (maptomesh.estimate)
    label = "South Tyrol DTM/DSM 2.5 m (Italy)"
    res_m = 2.5
    suffix = "2.5m"
    max_area_km2 = 2500.0
    sample = (46.5300, 11.6000, 5000)   # Schlern / Seiser Alm


# -------------------------------------------------------------------------------------------- Tuscany

TOS_WMS = "https://www502.regione.toscana.it/wmsraster/com.rt.wms.RTmap/wms"
TOS_TERMS = "http://www502.regione.toscana.it/geoscopio/servizi/wms/MORFOLOGIA.htm"


class TuscanyIrs1m(DemProvider):
    key = "rt_tos_dtm1m_irs"
    read_kind = "file"   # download estimates (maptomesh.estimate)
    label = "Tuscany LiDAR DTM 1 m 2019-21 (Italy, partial, non-commercial)"
    res_m = 1.0
    attribution = ("DTM LiDAR 1 m 2019-2021 © Italian Remote Sensing S.r.l., distributed by Regione Toscana "
                   "(Geoscopio), CC BY-NC-SA")
    licence = f"CC BY-NC-SA (version not stated), data owned by Italian Remote Sensing S.r.l. ({TOS_TERMS})"
    licence_note = ("Tuscany 1 m LiDAR DTM 2019-21 (owned by Italian Remote Sensing S.r.l., distributed by Regione "
                    "Toscana for use by public bodies) is under CC BY-NC-SA: prints of this area must not be sold or "
                    "used commercially, and derived files you share must keep the same licence; terms: " + TOS_TERMS)
    regions = ((9.66, 42.20, 12.42, 44.48),)
    max_area_km2 = 2500.0
    sample = (42.8875, 11.6230, 3000)   # Monte Amiata
    prefer = 2

    def fetch(self, aoi, grid, progress, cancel):
        res = request_res(self.res_m, grid)
        bbox = _snap(src_bbox(aoi, grid, 25832), res)
        out = np.full(grid.vshape, np.nan)
        n = 0
        for (x0, y0, x1, y1), w, h in service_tiles(bbox, res, 4000):   # service limit 4096 px
            _cancelled(cancel)
            n += 1
            progress(0.1, f"Tuscany DTM 1 m request {n}")
            data = service_get(TOS_WMS, {
                "map": "wmsmorfologia", "SERVICE": "WMS", "VERSION": "1.3.0", "REQUEST": "GetMap",
                "LAYERS": "rt_morfologia.iddtm.1m.irs", "STYLES": "", "CRS": "EPSG:25832",
                "BBOX": f"{x0:.2f},{y0:.2f},{x1:.2f},{y1:.2f}", "WIDTH": w, "HEIGHT": h, "FORMAT": "image/tiff"},
                timeout=180)
            # float32 heights; exactly 0 outside the surveyed sheets (no nodata tag)
            merge_fill(out, _raster_to_grid(data, aoi, grid, src_crs=25832, zero_is_nodata=True))
        return DemArrays(out, None, f"WMS rt_morfologia.iddtm.1m.irs, {res:.1f} m, {n} request(s)")


register(ScotlandSrsp())
register(SouthTyrol05())
register(WalesLidar())
register(EnglandEaLidar())
register(IrelandGsi())
register(SouthTyrol25())
register(SpainIgnMdt05())
register(TuscanyIrs1m())
