"""France, Benelux and Switzerland: national LiDAR elevation models and classified point clouds.

DEMs (highest resolution first; ``best_dems`` fills gaps from the next one):

* IGN LiDAR HD MNT / MNS 0.5 m (France, ~80 % published, rest by end 2026) and IGN RGE ALTI 1 m
  (all of metropolitan France) - float GeoTIFFs for any bbox from the Geoplateforme WMS-R.
* swisstopo swissALTI3D 0.5 / 2 m DTM + swissSURFACE3D Raster 0.5 m DSM - 1 km COGs via STAC.
* AHN4 DTM / DSM 0.5 m (Netherlands) - 5 x 6.25 km COGs listed in the PDOK Atom feeds. The DTM
  has no data under buildings and water; interior holes are interpolated here.
* Luxembourg LiDAR 2024 DTM / DSM 0.5 m - one national COG each (no CRS tag: EPSG:2169).
* Flanders DHMV II DTM / DSM 1 m - WCS 1.0.0 of Digitaal Vlaanderen.

Point clouds (COPC, range-read):

* IGN LiDAR HD classified tiles (1 km) found through the ``IGNF_LIDAR-HD_METADONNEE:metadata`` WFS.
* swisstopo swissSURFACE3D tiles from STAC - only the 2024+ cycle is COPC (older tiles are LAS zips).
"""

from __future__ import annotations

import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests

from ..dem import DemArrays, DemProvider, register
from ..dem_readers import merge_fill, read_cog, read_geotiff_bytes, request_res, service_get, service_tiles, src_bbox
from ..lidar_index import Tile
from ..pointcloud import PointCloudProvider, register_pointcloud

WORKERS = 4   # parallel COG reads per provider (be gentle with the servers)

FR_REGIONS = ((-5.2, 42.3, 8.25, 51.1), (8.5, 41.3, 9.6, 43.05))   # mainland, Corsica
CH_REGIONS = ((5.95, 45.81, 10.5, 47.81),)
NL_REGIONS = ((3.3, 50.74, 7.23, 53.56),)
LU_REGIONS = ((5.73, 49.44, 6.53, 50.19),)
FL_REGIONS = ((2.53, 50.67, 5.92, 51.51),)

IGN_ATTRIB = ("Contains data from IGN (Institut national de l'information géographique et forestière), "
              "Licence Ouverte Etalab 2.0")
IGN_LICENCE = "Licence Ouverte / Open Licence 2.0 (https://www.etalab.gouv.fr/licence-ouverte-open-licence)"
SWISS_ATTRIB = "© swisstopo (Federal Office of Topography swisstopo)"
SWISS_LICENCE = ("swisstopo OGD terms: free use incl. commercial, source reference mandatory "
                 "(https://www.swisstopo.admin.ch/ogd-conditions)")


def _cancelled(cancel) -> None:
    if cancel is not None and cancel.is_set():
        raise RuntimeError("Cancelled")


def _json(url: str, params: dict | None = None) -> dict:
    return json.loads(service_get(url, params, timeout=90))


def _cached_xml(url: str, days: float = 30.0) -> str:
    """GET an XML document (index feed) with the same 30-day disk cache as ``service_get``.
    (``service_get`` rejects bodies that start with ``<?xml`` as service errors.)"""
    import hashlib

    from .. import config
    from ..dem_readers import SERVICE_CACHE
    path = SERVICE_CACHE / f"{hashlib.sha1(url.encode()).hexdigest()[:24]}.xml"
    if path.exists() and time.time() - path.stat().st_mtime < days * 86400:
        return path.read_text(encoding="utf-8")
    r = requests.get(url, headers={"User-Agent": config.USER_AGENT}, timeout=(20, 120))
    r.raise_for_status()
    SERVICE_CACHE.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    text = r.content.decode("utf-8", "replace")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    return text


def _fmt_bbox(b) -> str:
    return ",".join(f"{v:.2f}" for v in b)


def _datum_grid(aoi, grid, epsg: int):
    """Grid to hand to the generic readers for data in ``epsg``.

    The AOI's local CRS is a bare tmerc on GRS80 without a datum, so PROJ/GDAL transform national
    CRSs on other datums to it as a "ballpark" (no datum shift): CH1903+ / LV95, Amersfoort / RD,
    BD72 / Lambert 72 and LUREF land 60-155 m away from their true position. Shifting the grid by
    that (locally constant, < 1 m variation over 10 km) offset puts the data back in place. When the
    AOI CRS carries a datum (``+towgs84=0,0,0``) the offset is zero and the grid is returned as is."""
    import dataclasses

    from pyproj import Transformer
    cx, cy = grid.x0 + grid.nx * grid.step / 2, grid.y0 + grid.ny * grid.step / 2
    lon, lat = aoi.to_lonlat.transform(cx, cy)
    x, y = Transformer.from_crs(4258, epsg, always_xy=True).transform(lon, lat)   # true national position
    bx, by = Transformer.from_crs(epsg, aoi.crs, always_xy=True).transform(x, y)   # what the readers compute
    dx, dy = bx - cx, by - cy
    if not (np.isfinite(dx) and np.isfinite(dy)) or (abs(dx) < 0.05 and abs(dy) < 0.05) or max(abs(dx), abs(dy)) > 1000:
        return grid
    return dataclasses.replace(grid, x0=grid.x0 + dx, y0=grid.y0 + dy)


def _true_bbox(aoi, grid, epsg: int, margin_cells: float = 3.0) -> tuple[float, float, float, float]:
    """Like ``src_bbox`` but through lon/lat, so the datum shift of ``epsg`` is applied."""
    from pyproj import Transformer
    ring = aoi.outline_local(margin=grid.step * margin_cells, n=24)
    lon, lat = aoi.to_lonlat.transform(ring[:, 0], ring[:, 1])
    x, y = Transformer.from_crs(4258, epsg, always_xy=True).transform(lon, lat)
    return float(np.min(x)), float(np.min(y)), float(np.max(x)), float(np.max(y))


def _subgrids(grid, n: int = 2):
    """Split the vertex grid into n x n blocks (sharing their edge vertices) for parallel reads.
    Yields (row0, row1, col0, col1, sub_grid); vertex rows row0..row1 inclusive."""
    import dataclasses
    rows = np.linspace(0, grid.ny, n + 1).round().astype(int)
    cols = np.linspace(0, grid.nx, n + 1).round().astype(int)
    for i in range(n):
        for j in range(n):
            r0, r1, c0, c1 = rows[i], rows[i + 1], cols[j], cols[j + 1]
            if r1 > r0 and c1 > c0:
                yield r0, r1, c0, c1, dataclasses.replace(grid, nx=int(c1 - c0), ny=int(r1 - r0),
                                                          x0=grid.x0 + c0 * grid.step, y0=grid.y0 + r0 * grid.step)


def _read_cogs_split(urls: dict, aoi, grid, cancel, progress, src_crs=None, nodata=None) -> dict:
    """Read big single COGs (``{name: url}``) in 2 x 2 blocks, ``WORKERS`` requests at a time
    (one national file is otherwise read one block request after another)."""
    out = {k: np.full(grid.vshape, np.nan) for k in urls}
    jobs = [(k, u, blk) for k, u in urls.items() for blk in _subgrids(grid, 2 if max(grid.nx, grid.ny) >= 64 else 1)]

    def one(job):
        _cancelled(cancel)
        k, u, (r0, r1, c0, c1, g) = job
        return job, read_cog(u, aoi, g, src_crs=src_crs, nodata=nodata)

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(one, j) for j in jobs]
        try:
            for n, fut in enumerate(as_completed(futs)):
                (k, _, (r0, r1, c0, c1, _)), arr = fut.result()
                _cancelled(cancel)
                merge_fill(out[k][r0:r1 + 1, c0:c1 + 1], arr)
                progress((n + 1) / len(jobs), f"{k} block {n + 1}/{len(jobs)}")
        except BaseException:
            for f in futs:
                f.cancel()
            raise
    return out


def _fill_holes(a: np.ndarray, step: float, max_m2: float = 1e6) -> np.ndarray:
    """Interpolate NaN holes that are enclosed by data (buildings, ponds) and smaller than
    ``max_m2``. Holes touching the grid edge stay NaN (the next provider fills them)."""
    from scipy import ndimage

    from ..raster import fill_nan
    nan = np.isnan(a)
    if not nan.any() or nan.all():
        return a
    lab, n = ndimage.label(nan)
    ok = np.zeros(n + 1, bool)
    ok[1:] = np.asarray(ndimage.sum(nan, lab, index=np.arange(1, n + 1))) * step * step <= max_m2
    ok[np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))] = False
    hole = ok[lab]
    if hole.any():
        a = a.copy()
        a[hole] = fill_nan(a)[hole]
    return a


def _dsm_where_missing(dtm: np.ndarray, dsm: np.ndarray | None) -> np.ndarray | None:
    """No surface return (water, gaps) -> surface = ground."""
    if dsm is None:
        return None
    return np.where(np.isnan(dsm) & np.isfinite(dtm), dtm, dsm)


def _mosaic(urls: list[str], aoi, grid, cancel, progress, what: str, src_crs=None,
            f0: float = 0.0, f1: float = 1.0) -> tuple[np.ndarray, str]:
    """Read non-overlapping COG tiles (``WORKERS`` at a time) onto one vertex grid.
    Returns the mosaic and a "<n> tiles[, <k> failed]" note; raises if every tile failed."""
    out = np.full(grid.vshape, np.nan)
    if not urls:
        return out, "0 tiles"
    errors: list[str] = []
    done = 0

    def one(u):
        _cancelled(cancel)
        return read_cog(u, aoi, grid, src_crs=src_crs)

    with ThreadPoolExecutor(max_workers=min(WORKERS, len(urls))) as ex:
        futs = {ex.submit(one, u): u for u in urls}
        try:
            for fut in as_completed(futs):
                _cancelled(cancel)
                try:
                    merge_fill(out, fut.result())
                    done += 1
                except Exception as e:  # noqa: BLE001 - one bad tile: keep the others
                    if cancel is not None and cancel.is_set():
                        raise RuntimeError("Cancelled") from e
                    errors.append(f"{futs[fut].rsplit('/', 1)[-1]}: {type(e).__name__}: {e}")
                progress(f0 + (f1 - f0) * (done + len(errors)) / len(urls), f"{what} tile {done + len(errors)}/{len(urls)}")
        except BaseException:
            for f in futs:
                f.cancel()
            raise
    if done == 0:
        raise RuntimeError(f"{what}: no tile could be read ({errors[0][:200]})")
    return out, f"{done} tiles" + (f", {len(errors)} failed ({errors[0][:80]})" if errors else "")


# ====================================================================== France (IGN)

IGN_WMSR = "https://data.geopf.fr/wms-r"
IGN_WFS = "https://data.geopf.fr/wfs/ows"


def _ign_wms(layer: str, native: float, aoi, grid, cancel, max_px: int = 4000) -> np.ndarray:
    """IGN WMS-R GetMap (float32 GeoTIFF, metres, -9999 = no data) in Lambert-93 pieces."""
    bbox = src_bbox(aoi, grid, 2154)
    res = request_res(native, grid)
    out = np.full(grid.vshape, np.nan)
    for b, w, h in service_tiles(bbox, res, max_px):
        params = {"SERVICE": "WMS", "VERSION": "1.3.0", "REQUEST": "GetMap", "LAYERS": layer, "STYLES": "",
                  "CRS": "EPSG:2154", "BBOX": _fmt_bbox(b), "WIDTH": w, "HEIGHT": h,
                  "FORMAT": "image/geotiff", "EXCEPTIONS": "text/xml"}
        for attempt in range(3):   # the Geoplateforme answers the odd request with a transient 400
            _cancelled(cancel)
            try:
                data = service_get(IGN_WMSR, params)
                break
            except requests.HTTPError as e:
                if attempt == 2 or e.response is None or e.response.status_code not in (400, 429):
                    raise
                time.sleep(2 + 3 * attempt)
        merge_fill(out, read_geotiff_bytes(data, aoi, grid, src_crs=2154, nodata=-9999.0))
    return out


class IgnLidarHdMnt(DemProvider):
    key = "ign_lidarhd_mnt"
    label = "IGN LiDAR HD 0.5 m (France)"
    res_m = 0.5
    has_dsm = True
    attribution = IGN_ATTRIB + " - LiDAR HD MNT/MNS"
    licence = IGN_LICENCE
    regions = FR_REGIONS
    max_area_km2 = 2500.0     # a service: the request size follows the grid, not the area
    sample = (48.8867, 2.3431, 2000)   # Montmartre, Paris

    def fetch(self, aoi, grid, progress, cancel):
        progress(0.0, "IGN LiDAR HD MNT")
        dtm = _ign_wms("IGNF_LIDAR-HD_MNT_ELEVATION.ELEVATIONGRIDCOVERAGE.LAMB93", self.res_m, aoi, grid, cancel)
        dsm = None
        if np.isfinite(dtm).any():
            progress(0.5, "IGN LiDAR HD MNS")
            dsm = _ign_wms("IGNF_LIDAR-HD_MNS_ELEVATION.ELEVATIONGRIDCOVERAGE.LAMB93", self.res_m, aoi, grid, cancel)
            dsm = _dsm_where_missing(dtm, dsm)
        return DemArrays(dtm, dsm, "WMS-R IGNF_LIDAR-HD_MNT/MNS (EPSG:2154)")


class IgnRgeAlti(DemProvider):
    key = "ign_rgealti"
    label = "IGN RGE ALTI 1 m (France)"
    res_m = 1.0
    attribution = IGN_ATTRIB + " - RGE ALTI"
    licence = IGN_LICENCE
    regions = FR_REGIONS
    max_area_km2 = 10_000.0
    sample = (45.9237, 6.8694, 4000)   # Chamonix

    def fetch(self, aoi, grid, progress, cancel):
        progress(0.0, "IGN RGE ALTI")
        dtm = _ign_wms("ELEVATION.ELEVATIONGRIDCOVERAGE.HIGHRES", self.res_m, aoi, grid, cancel)
        return DemArrays(dtm, None, "WMS-R ELEVATION.ELEVATIONGRIDCOVERAGE.HIGHRES (EPSG:2154)")


def _year(*dates) -> int:
    for d in dates:
        if d and str(d)[:4].isdigit():
            return int(str(d)[:4])
    return 0


class IgnLidarHdCopc(PointCloudProvider):
    key = "ign_lidarhd_copc"
    label = "IGN LiDAR HD (France)"
    attribution = IGN_ATTRIB + " - LiDAR HD"
    licence = IGN_LICENCE
    regions = FR_REGIONS
    sample = (48.8600, 2.3400, 1000)

    def tiles(self, bbox, max_items):
        w, s, e, n = bbox
        out, seen, start = [], set(), 0
        page = 500
        while len(out) < max_items:
            d = _json(IGN_WFS, {
                "SERVICE": "WFS", "VERSION": "2.0.0", "REQUEST": "GetFeature",
                "TYPENAMES": "IGNF_LIDAR-HD_METADONNEE:metadata", "OUTPUTFORMAT": "application/json",
                "COUNT": page, "STARTINDEX": start,
                "BBOX": f"{s:.6f},{w:.6f},{n:.6f},{e:.6f},urn:ogc:def:crs:EPSG::4326"})
            feats = d.get("features") or []
            for f in feats:
                p = f.get("properties") or {}
                url = p.get("url_npl") or ""
                if not url.lower().endswith(".copc.laz") or url in seen:
                    continue
                seen.add(url)
                pts = np.array([c for poly in f["geometry"]["coordinates"] for ring in poly for c in ring])
                tb = (float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max()))
                out.append(Tile(url, f"IGN LiDAR HD {p.get('code_mission') or ''}".strip(),
                                _year(p.get("date_fin_acquisition"), p.get("date_debut_acquisition"),
                                      p.get("date_edition")), tb))
            if len(feats) < page:
                break
            start += page
        return out[:max_items]


# ====================================================================== Switzerland (swisstopo)

SWISS_STAC = "https://data.geo.admin.ch/api/stac/v1"


def _swiss_items(collection: str, bbox, max_items: int = 3000) -> list[dict]:
    url = f"{SWISS_STAC}/collections/{collection}/items"
    params = {"bbox": ",".join(f"{v:.5f}" for v in bbox), "limit": 100}
    feats: list[dict] = []
    while url and len(feats) < max_items:
        d = _json(url, params)
        feats += d.get("features") or []
        url = next((lk["href"] for lk in d.get("links", []) if lk.get("rel") == "next"), None)
        params = None
    return feats


def _latest_per_tile(items: list[dict]) -> list[dict]:
    """swisstopo item ids are <product>_<year>_<tile>: keep the newest year of each tile."""
    best: dict[str, dict] = {}
    for it in items:
        parts = it["id"].rsplit("_", 2)
        if len(parts) != 3 or not parts[1].isdigit():
            continue
        cur = best.get(parts[2])
        if cur is None or int(parts[1]) > int(cur["id"].rsplit("_", 2)[1]):
            best[parts[2]] = it
    return list(best.values())


def _swiss_tif(item: dict, gsd: float) -> str | None:
    for a in item.get("assets", {}).values():
        if a.get("href", "").endswith(".tif") and abs(float(a.get("gsd") or a.get("eo:gsd") or -1) - gsd) < 1e-6:
            return a["href"]
    return None


class SwissAlti3d(DemProvider):
    key = "swisstopo_alti3d"
    label = "swisstopo swissALTI3D 0.5 m (Switzerland)"
    res_m = 0.5
    has_dsm = True
    attribution = SWISS_ATTRIB + " - swissALTI3D, swissSURFACE3D Raster"
    licence = SWISS_LICENCE
    regions = CH_REGIONS
    max_area_km2 = 400.0      # 1 km tiles: ~400 DTM files (+ DSM) at the limit
    sample = (46.5935, 7.9077, 3000)   # Lauterbrunnen valley
    dsm_budget_mb = 100.0     # skip the DSM (0.5 m tiles, overviews only to 2 m) beyond this estimate

    def fetch(self, aoi, grid, progress, cancel):
        bbox = aoi.lonlat_bbox(margin=grid.step * 2)
        gsd = 2.0 if grid.step >= 4 else 0.5
        items = _latest_per_tile(_swiss_items("ch.swisstopo.swissalti3d", bbox))
        urls = [u for u in (_swiss_tif(it, gsd) or _swiss_tif(it, 0.5) for it in items) if u]
        g = _datum_grid(aoi, grid, 2056)
        dtm, n_dtm = _mosaic(urls, aoi, g, cancel, progress, "swissALTI3D", f0=0.0, f1=0.6)
        detail = f"swissALTI3D {gsd:g} m {n_dtm} ({', '.join(sorted({it['id'].split('_')[1] for it in items}))})"
        dsm = None
        _cancelled(cancel)
        ditems = _latest_per_tile(_swiss_items("ch.swisstopo.swisssurface3d-raster", bbox))
        durls = [u for u in (_swiss_tif(it, 0.5) for it in ditems) if u]
        f = max(f for f in (1, 2, 4) if 0.5 * f <= max(grid.step / 2, 0.5))   # overview read_cog will use
        est_mb = len(durls) * (2000 / f) ** 2 * 4 / 1e6   # LZW float32: ~4 bytes/px (measured)
        if durls and est_mb <= self.dsm_budget_mb:
            dsm, n_dsm = _mosaic(durls, aoi, g, cancel, progress, "swissSURFACE3D Raster", f0=0.6, f1=1.0)
            dsm = _dsm_where_missing(dtm, dsm)
            detail += f"; swissSURFACE3D Raster 0.5 m {n_dsm}"
        elif durls:
            detail += f"; DSM skipped (~{est_mb:.0f} MB)"
        return DemArrays(dtm, dsm, detail)


class SwissSurface3dCopc(PointCloudProvider):
    key = "swisstopo_surface3d_copc"
    label = "swisstopo swissSURFACE3D (Switzerland)"
    attribution = SWISS_ATTRIB + " - swissSURFACE3D"
    licence = SWISS_LICENCE
    regions = CH_REGIONS
    sample = (47.3720, 8.5420, 1000)   # Zurich (2024 COPC)

    def tiles(self, bbox, max_items):
        out = []
        for it in _swiss_items("ch.swisstopo.swisssurface3d", bbox, max_items=max(max_items * 3, 100)):
            for a in it.get("assets", {}).values():
                href = a.get("href", "")
                if href.lower().endswith(".copc.laz") or "copc" in (a.get("type") or ""):
                    year = _year(it.get("properties", {}).get("datetime"), it["id"].split("_")[1])
                    if year < 2024:
                        continue
                    b = it.get("bbox") or bbox
                    out.append(Tile(href, f"swissSURFACE3D {year}", year, tuple(float(v) for v in b[:4])))
        return out[:max_items]


# ====================================================================== Netherlands (AHN4, PDOK)

AHN_FEEDS = {"dtm": "https://service.pdok.nl/rws/actueel-hoogtebestand-nederland/atom/dtm_05m.xml",
             "dsm": "https://service.pdok.nl/rws/actueel-hoogtebestand-nederland/atom/dsm_05m.xml"}
_AHN_INDEX: dict[str, list] = {}
_AHN_LOCK = threading.Lock()


def _ahn_index(kind: str) -> list[tuple[str, tuple[float, float, float, float]]]:
    """(COG URL, lon/lat bbox) of every AHN sheet in the PDOK Atom feed."""
    with _AHN_LOCK:
        if kind not in _AHN_INDEX:
            xml = _cached_xml(AHN_FEEDS[kind])
            out = []
            for tag in re.findall(r"<link\b[^>]*>", xml):
                href = re.search(r'href="([^"]+\.tif)"', tag)
                bb = re.search(r'bbox="([-\d. ]+)"', tag)
                if href and bb:
                    v = [float(x) for x in bb.group(1).split()]
                    if len(v) == 4:
                        out.append((href.group(1), (v[0], v[1], v[2], v[3])))
            if not out:
                raise RuntimeError(f"no tiles in {AHN_FEEDS[kind]}")
            _AHN_INDEX[kind] = out
        return _AHN_INDEX[kind]


def _hits(index, bbox) -> list[str]:
    w, s, e, n = bbox
    return [u for u, (x0, y0, x1, y1) in index if x0 <= e and x1 >= w and y0 <= n and y1 >= s]


class Ahn4(DemProvider):
    key = "ahn4"
    label = "AHN4 0.5 m (Netherlands)"
    res_m = 0.5
    has_dsm = True
    attribution = "AHN - Actueel Hoogtebestand Nederland (www.ahn.nl), AHN4 via PDOK"
    licence = "CC0 1.0 (https://creativecommons.org/publicdomain/zero/1.0/)"
    regions = NL_REGIONS
    max_area_km2 = 1000.0     # ~30 sheets; overviews down to 16 m keep it small
    sample = (52.0300, 6.0150, 2000)   # Posbank, Veluwezoom

    def fetch(self, aoi, grid, progress, cancel):
        bbox = aoi.lonlat_bbox(margin=grid.step * 2)
        urls = _hits(_ahn_index("dtm"), bbox)
        if not urls:
            return DemArrays(np.full(grid.vshape, np.nan), None, "no AHN sheet")
        g = _datum_grid(aoi, grid, 28992)
        dtm, n = _mosaic(urls, aoi, g, cancel, progress, "AHN4 DTM", f0=0.0, f1=0.5)
        dtm = _fill_holes(dtm, grid.step)   # the DTM is empty under buildings and water
        _cancelled(cancel)
        durls = _hits(_ahn_index("dsm"), bbox)
        dsm = None
        if durls:
            dsm, _ = _mosaic(durls, aoi, g, cancel, progress, "AHN4 DSM", f0=0.5, f1=1.0)
            dsm = _dsm_where_missing(dtm, dsm)
        names = ", ".join(u.rsplit("/", 1)[-1] for u in urls)
        return DemArrays(dtm, dsm, f"AHN4 {n}: {names}")


# ====================================================================== Luxembourg (LiDAR 2024)

LU_BASE = "https://download.data.public.lu/resources/bd-l-lidar2024-releve-3d-du-territoire-luxembourgeois/20241223-093912/"


class Luxembourg2024(DemProvider):
    key = "lu_lidar2024"
    label = "Luxembourg LiDAR 2024 0.5 m"
    res_m = 0.5
    has_dsm = True
    attribution = "Administration du cadastre et de la topographie (ACT) Luxembourg - LiDAR 2024"
    licence = "CC0 1.0 (https://data.public.lu/fr/datasets/lidar-2024-releve-3d-du-territoire-luxembourgeois/)"
    regions = LU_REGIONS
    max_area_km2 = 3000.0     # one national COG with overviews down to ~1 km
    sample = (49.9340, 6.2020, 3000)   # Vianden

    def fetch(self, aoi, grid, progress, cancel):
        progress(0.0, "Luxembourg LiDAR 2024 DTM + DSM")
        g = _datum_grid(aoi, grid, 2169)
        got = _read_cogs_split({"DTM": LU_BASE + "MNT_Lidar2024.tif", "DSM": LU_BASE + "MNS_Lidar2024.tif"},
                               aoi, g, cancel, progress, src_crs=2169, nodata=-9999.0)
        dtm = _fill_holes(got["DTM"], grid.step, max_m2=2e5)
        dsm = _dsm_where_missing(dtm, got["DSM"]) if np.isfinite(dtm).any() else None
        return DemArrays(dtm, dsm, "MNT/MNS_Lidar2024.tif (EPSG:2169)")


# ====================================================================== Flanders (DHMV II)

FL_WCS = "https://geo.api.vlaanderen.be/DHMV/wcs"


def _fl_wcs(coverage: str, aoi, grid, cancel, max_px: int = 2500) -> np.ndarray:
    bbox = _true_bbox(aoi, grid, 31370)
    grid = _datum_grid(aoi, grid, 31370)   # same shape, shifted for the BD72 datum
    res = request_res(1.0, grid)
    out = np.full(grid.vshape, np.nan)
    for b, w, h in service_tiles(bbox, res, max_px):
        _cancelled(cancel)
        data = service_get(FL_WCS, {
            "SERVICE": "WCS", "VERSION": "1.0.0", "REQUEST": "GetCoverage", "COVERAGE": coverage,
            "FORMAT": "GeoTIFF", "CRS": "EPSG:31370", "BBOX": _fmt_bbox(b), "WIDTH": w, "HEIGHT": h})
        merge_fill(out, read_geotiff_bytes(data, aoi, grid, src_crs=31370, nodata=-9999.0))
    return out


class FlandersDhmv2(DemProvider):
    key = "vl_dhmv2"
    label = "Digitaal Hoogtemodel Vlaanderen II 1 m (Flanders)"
    res_m = 1.0
    has_dsm = True
    attribution = "Bron: Digitaal Vlaanderen - Digitaal Hoogtemodel Vlaanderen II (2013-2015)"
    licence = "Modellicentie Gratis Hergebruik v1.0 (https://data.vlaanderen.be/id/licentie/modellicentie-gratis-hergebruik/v1.0)"
    regions = FL_REGIONS
    max_area_km2 = 2500.0
    sample = (50.7830, 2.8150, 2000)   # Kemmelberg

    def fetch(self, aoi, grid, progress, cancel):
        progress(0.0, "DHMV II DTM")
        dtm = _fl_wcs("DHMVII_DTM_1m", aoi, grid, cancel)
        dtm = _fill_holes(dtm, grid.step, max_m2=2e5)
        dsm = None
        if np.isfinite(dtm).any():
            progress(0.5, "DHMV II DSM")
            dsm = _dsm_where_missing(dtm, _fl_wcs("DHMVII_DSM_1m", aoi, grid, cancel))
        return DemArrays(dtm, dsm, "WCS DHMVII_DTM_1m / DHMVII_DSM_1m (EPSG:31370)")


register(IgnLidarHdMnt())
register(IgnRgeAlti())
register(SwissAlti3d())
register(Ahn4())
register(Luxembourg2024())
register(FlandersDhmv2())
register_pointcloud(IgnLidarHdCopc())
register_pointcloud(SwissSurface3dCopc())
