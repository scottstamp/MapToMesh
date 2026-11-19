"""Central Europe: Austria, German states, and COPC point clouds for Slovenia, Poland, Saxony and Wallonia.

Elevation models (highest resolution first; ``best_dems`` fills gaps from the next one):

* Austria - BEV ALS DTM + DSM 1 m: national Cloud-Optimized GeoTIFFs in 50 km EPSG:3035 tiles.
  Each "Stichtag" folder (``20250915`` ...) holds a complete national mosaic; the newest folder is
  found once by HEAD-probing and remembered for a week.
* Lower Saxony - LGLN DGM1 1 m COGs (STAC API) + bDOM20 image-matching surface model (STAC API).
* North Rhine-Westphalia - DGM1 + DOM1 1 m GeoTIFFs, 1 km tiles (file names from the directory index).
* Bavaria - DGM1 1 m GeoTIFFs, 1 km tiles with predictable names.
* Rhineland-Palatinate - DGM1 + DOM1 1 m GeoTIFFs, 1 km tiles (file names from the directory index).
* Brandenburg and Berlin - DGM1 1 m GeoTIFFs zipped per 1 km tile.
* Mecklenburg-Western Pomerania - DGM1 + DOM1 through the state WCS (any extent / resolution).
* Saxony - DGM1 1 m GeoTIFFs zipped per 2 km tile.

The German tile files are not cloud-optimized: whole files (1-15 MB) are downloaded once and kept
in the 30-day service cache, so ``max_area_km2`` is modest and larger areas fall back to the next
model. Missing tiles (404) are remembered for 30 days.

Point clouds (COPC, ``PointCloudProvider``): the Flai ``open-lidar-data`` mirror on AWS S3
(a third-party copy of the official open data). Each dataset publishes a shapefile of file
bounding boxes (``shp/files_bbox_<name>.shp``); its .shp/.shx are downloaded once and kept as a
compact ``.npz``, and only the .dbf records of intersecting files are range-read (persistent range
cache). The files carry no CRS, so ``Tile.crs`` is set per dataset. Saxony's GeoSN point clouds use
class 20 for non-ground points, which ``pointcloud`` reads as ground, so that provider is only
registered once ``Tile`` has a ``class_map`` field.
"""

from __future__ import annotations

import dataclasses
import datetime
import hashlib
import io
import json
import math
import re
import struct
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests

from .. import config
from ..dem import DemArrays, DemProvider, register
from ..dem_readers import merge_fill, read_cog, read_service, reproject_array, request_res, service_get, src_bbox
from ..lidar_index import Tile
from ..pointcloud import Cancelled, PointCloudProvider, register_pointcloud

CACHE = config.CACHE_DIR / "central_eu"
WORKERS = 4                      # parallel downloads per provider (be gentle with state servers)
INDEX_DAYS = 14                  # directory listings / STAC searches
MISSING_DAYS = 30                # remembered 404s
YEAR = datetime.date.today().year

_LOCK = threading.Lock()


# ---------------------------------------------------------------- small HTTP helpers

def _session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = config.USER_AGENT
    s.mount("https://", requests.adapters.HTTPAdapter(max_retries=requests.adapters.Retry(
        total=3, backoff_factor=1.0, status_forcelist=(429, 500, 502, 503, 504))))
    return s


def _disk_cached(name: str, days: float, make):
    """JSON value cached on disk under CACHE (``make()`` computes it)."""
    path = CACHE / f"{name}.json"
    if path.exists() and time.time() - path.stat().st_mtime < days * 86400:
        try:
            return json.loads(path.read_text())
        except ValueError:
            pass
    val = make()
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(val))
    tmp.replace(path)
    return val


class _Missing:
    """URLs that answered 404, remembered on disk for MISSING_DAYS."""

    def __init__(self):
        self.path = CACHE / "missing.json"
        self._d: dict[str, float] | None = None

    def _load(self) -> dict[str, float]:
        if self._d is None:
            try:
                self._d = json.loads(self.path.read_text())
            except (OSError, ValueError):
                self._d = {}
            now = time.time()
            self._d = {u: t for u, t in self._d.items() if now - t < MISSING_DAYS * 86400}
        return self._d

    def has(self, url: str) -> bool:
        with _LOCK:
            return url in self._load()

    def add(self, url: str) -> None:
        with _LOCK:
            d = self._load()
            d[url] = time.time()
            CACHE.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(f".{threading.get_ident()}.tmp")
            tmp.write_text(json.dumps(d))
            tmp.replace(self.path)


MISSING = _Missing()


def _get_file(url: str, timeout: float = 180.0) -> bytes | None:
    """A whole (small) file, cached 30 days by ``service_get``; None when the server has no such
    file (404 / 410, remembered)."""
    if MISSING.has(url):
        return None
    try:
        return service_get(url, timeout=timeout)
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code in (404, 410):
            MISSING.add(url)
            return None
        raise


def _exists(url: str) -> bool:
    """HEAD through the range cache (size is remembered); False for 404 (remembered)."""
    from ..httpcache import RANGE_CACHE
    if MISSING.has(url):
        return False
    try:
        RANGE_CACHE.size(url)
        return True
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code in (404, 410):
            MISSING.add(url)
            return False
        raise


def _listing(url: str, pattern: str, name: str) -> dict[str, str]:
    """``"E_N" -> file name`` from a directory listing (HTML / XML), cached INDEX_DAYS.
    ``pattern`` has groups (file, E, N)."""
    def make():
        r = _session().get(url, timeout=(20, 300))
        r.raise_for_status()
        out: dict[str, str] = {}
        for f, e, n in re.findall(pattern, r.text):
            out[f"{int(e)}_{int(n)}"] = f   # several vintages: the listing is sorted, keep the last
        if not out:
            raise RuntimeError(f"no tiles listed at {url}")
        return out
    return _disk_cached(name, INDEX_DAYS, make)


def _unzip(data: bytes, ext: str = ".tif") -> bytes:
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for n in z.namelist():
            if n.lower().endswith(ext):
                return z.read(n)
    raise RuntimeError(f"no {ext} in zip")


def _tif_to_grid(data: bytes, aoi, grid, src_crs=None) -> np.ndarray:
    """GeoTIFF bytes -> vertex grid; block-averaged ('average') when the grid is much coarser than
    the data (point sampling 1 m heights at 10 m steps would alias), bilinear otherwise."""
    from rasterio.io import MemoryFile
    with MemoryFile(data) as mf, mf.open() as src:
        arr = src.read(1).astype(np.float32)
        nd = src.nodata
        crs = src.crs or src_crs
        if crs is None:
            raise ValueError("GeoTIFF without CRS")
        res = abs(src.res[0])
        if nd is not None and np.isfinite(nd):
            arr[arr == nd] = np.nan
        arr[(arr < -1000) | (arr > 9000)] = np.nan    # stray fill values (-9999, -32767, 3.4e38)
        rs = "average" if grid.step > 4 * res else "bilinear"
        return reproject_array(arr, src.transform, crs, aoi, grid, nodata=None, resampling=rs)


def _km_tiles(aoi, grid, epsg: int, size_m: int) -> list[tuple[int, int]]:
    """Lower-left corners (metres) of the ``size_m`` tiles of an aligned grid covering the AOI."""
    x0, y0, x1, y1 = src_bbox(aoi, grid, epsg)
    xs = range(int(math.floor(x0 / size_m)) * size_m, int(math.ceil(x1 / size_m)) * size_m, size_m)
    ys = range(int(math.floor(y0 / size_m)) * size_m, int(math.ceil(y1 / size_m)) * size_m, size_m)
    return [(x, y) for y in ys for x in xs]


def _parallel(jobs: list, fn, progress, cancel, label: str) -> list:
    """Run ``fn(job)`` for every job with WORKERS threads; results in job order. Raises Cancelled."""
    out = [None] * len(jobs)
    if not jobs:
        return out
    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(lambda j: None if cancel.is_set() else fn(j), j): i for i, j in enumerate(jobs)}
        try:
            for f in list(futs):
                out[futs[f]] = f.result()
                done += 1
                progress(done / len(jobs), f"{label}: {done}/{len(jobs)} files")
                if cancel.is_set():
                    raise Cancelled()
        except BaseException:
            for f in futs:
                f.cancel()
            raise
    return out


# ---------------------------------------------------------------- Austria (BEV)

BEV_BASE = "https://data.bev.gv.at/download/ALS"
BEV_PROBE_TILE = "N2800000E4600000"     # a 50 km tile inside Austria (exists in every series)


def _bev_folder() -> dict:
    """Newest BEV series folder and its file-name prefix, e.g. {"folder": "20250915", "prefix": "ALS_DTM_"}.
    Series are published as 15 September of a year (the 2021 series sits in 20210401; 2019 has no
    ``ALS_DTM_`` prefix)."""
    def make():
        cands = [f"{y}0915" for y in range(YEAR + 1, 2018, -1)]
        cands.insert(cands.index("20210915") + 1, "20210401")
        s = _session()
        for folder in cands:
            for prefix in ("ALS_DTM_", ""):
                url = f"{BEV_BASE}/DTM/{folder}/{prefix}CRS3035RES50000m{BEV_PROBE_TILE}.tif"
                r = s.head(url, timeout=30)
                if r.status_code == 200:
                    return {"folder": folder, "prefix": "ALS_{}_" if prefix else ""}
        raise RuntimeError("no BEV ALS series folder found")
    return _disk_cached("bev_series", 7, make)


class BevAustria(DemProvider):
    key = "bev_als_at"
    label = "BEV ALS 1 m (Austria)"
    res_m = 1.0
    has_dsm = True
    attribution = "Bundesamt für Eich- und Vermessungswesen (BEV), ALS-Höhenraster 1 m (DGM/DOM), CC BY 4.0"
    licence = "CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/; https://www.bev.gv.at)"
    regions = ((9.52, 46.37, 17.17, 49.03),)
    max_area_km2 = 2500.0          # COGs with overviews: cost grows with grid cells, not area
    sample = (47.3075, 11.3850, 3000)   # Innsbruck Nordkette

    def fetch(self, aoi, grid, progress, cancel):
        ser = _bev_folder()
        x0, y0, x1, y1 = src_bbox(aoi, grid, 3035)
        T = 50_000
        cells = [(e, n) for n in range(int(y0 // T) * T, int(math.ceil(y1 / T)) * T, T)
                 for e in range(int(x0 // T) * T, int(math.ceil(x1 / T)) * T, T)]
        dtm = np.full(grid.vshape, np.nan)
        dsm = np.full(grid.vshape, np.nan)
        used = []
        steps = len(cells) * 2
        for k, (e, n) in enumerate(cells):
            for j, (kind, arr) in enumerate((("DTM", dtm), ("DSM", dsm))):
                if cancel.is_set():
                    raise Cancelled()
                name = f"{ser['prefix'].format(kind)}CRS3035RES50000mN{n}E{e}.tif"
                url = f"{BEV_BASE}/{kind}/{ser['folder']}/{name}"
                if not _exists(url):
                    continue
                progress((2 * k + j) / steps, f"BEV ALS {kind} {ser['folder']} N{n // 1000}E{e // 1000}")
                merge_fill(arr, read_cog(url, aoi, grid))
                if kind == "DTM":
                    used.append(f"N{n // 1000}E{e // 1000}")
        return DemArrays(dtm, dsm if np.isfinite(dsm).any() else None,
                         f"series {ser['folder']}: " + ", ".join(used))


# ---------------------------------------------------------------- Lower Saxony (LGLN STAC)

def _stac_search(api: str, collection: str, bbox, max_items: int = 1000) -> list[dict]:
    """STAC item search (all pages), cached on disk. Uses normal TLS verification: Python's
    certifi-based check does not consult revocation lists (Windows schannel reports the
    dgm.stac.lgln.niedersachsen.de certificate as revoked; requests accepts it)."""
    bb = ",".join(f"{v:.5f}" for v in bbox)
    key = hashlib.sha1(f"{api}|{collection}|{bb}".encode()).hexdigest()[:16]

    def make():
        s = _session()
        url, params, feats = f"{api}/search", {"collections": collection, "bbox": bb, "limit": 100}, []
        while url and len(feats) < max_items:
            r = s.get(url, params=params, timeout=(20, 120))
            r.raise_for_status()
            js = r.json()
            feats += js.get("features", [])
            nxt = [ln for ln in js.get("links", []) if ln.get("rel") == "next"]
            url = nxt[0]["href"] if nxt and js.get("features") else None
            params = None
        return [{"id": f["id"], "datetime": f["properties"].get("datetime", ""),
                 "assets": {k: v["href"] for k, v in f["assets"].items()}} for f in feats]
    return _disk_cached(f"stac_{key}", 1, make)


def _newest_per_tile(items: list[dict], asset: str) -> list[tuple[str, str]]:
    """(tile name, href) keeping the newest item of each tile (names end in a year or date)."""
    best: dict[str, dict] = {}
    for it in items:
        if asset not in it["assets"]:
            continue
        m = re.match(r"(.*?_\d+_\d+_\d+_[a-z]{2})", it["id"])
        tile = m.group(1) if m else it["id"]
        if tile not in best or it["datetime"] > best[tile]["datetime"]:
            best[tile] = it
    return [(t, it["assets"][asset]) for t, it in sorted(best.items())]


class LglnNiedersachsen(DemProvider):
    key = "lgln_dgm1_ni"
    label = "LGLN DGM1 1 m (Lower Saxony)"
    res_m = 1.0
    has_dsm = True
    attribution = "© GeoBasis-DE / LGLN, DGM1 and bDOM20, CC BY 4.0"
    licence = "CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/; https://ni-lgln-opengeodata.hub.arcgis.com)"
    regions = ((6.60, 51.29, 11.60, 53.90),)
    max_area_km2 = 120.0           # one 1 km COG (+ one bDOM COG) per km2
    sample = (51.8880, 10.4260, 3000)   # Goslar / Harz edge

    def fetch(self, aoi, grid, progress, cancel):
        bbox = aoi.lonlat_bbox(margin=grid.step * 3)
        dtm_items = _newest_per_tile(_stac_search("https://dgm.stac.lgln.niedersachsen.de", "dgm1", bbox), "dgm1-tif")
        dsm_items = _newest_per_tile(_stac_search("https://bdom.stac.lgln.niedersachsen.de", "BDOM", bbox), "bdom20")
        jobs = [("dtm", h) for _, h in dtm_items] + [("dsm", h) for _, h in dsm_items]
        res = _parallel(jobs, lambda j: read_cog(j[1], aoi, grid), progress, cancel, "LGLN DGM1/bDOM")
        dtm = np.full(grid.vshape, np.nan)
        dsm = np.full(grid.vshape, np.nan)
        for (kind, _), arr in zip(jobs, res):
            merge_fill(dtm if kind == "dtm" else dsm, arr)
        return DemArrays(dtm, dsm if np.isfinite(dsm).any() else None,
                         f"{len(dtm_items)} DGM1 + {len(dsm_items)} bDOM20 tiles")


# ---------------------------------------------------------------- German 1-2 km GeoTIFF tiles

class _TileDem(DemProvider):
    """Whole-file GeoTIFF tiles on an aligned km grid. Subclasses give ``epsg``, ``tile_m`` and
    ``url(kind, e_km, n_km)`` (None = no such tile); zipped tiles set ``zipped``."""
    epsg: int = 25832
    tile_m: int = 1000
    zipped: bool = False
    res_m = 1.0

    def url(self, kind: str, e: int, n: int) -> str | None:
        raise NotImplementedError

    def fetch(self, aoi, grid, progress, cancel):
        kinds = ("dtm", "dsm") if self.has_dsm else ("dtm",)
        cells = _km_tiles(aoi, grid, self.epsg, self.tile_m)
        jobs = []
        for kind in kinds:
            for x, y in cells:
                u = self.url(kind, x // 1000, y // 1000)
                if u:
                    jobs.append((kind, u))

        def one(job):
            data = _get_file(job[1])
            if data is None:
                return None
            if self.zipped:
                data = _unzip(data)
            return _tif_to_grid(data, aoi, grid, src_crs=self.epsg)

        res = _parallel(jobs, one, progress, cancel, self.label)
        out = {k: np.full(grid.vshape, np.nan) for k in kinds}
        n_ok = 0
        for (kind, _), arr in zip(jobs, res):
            if arr is not None:
                merge_fill(out[kind], arr)
                n_ok += kind == "dtm"
        dsm = out.get("dsm")
        return DemArrays(out["dtm"], dsm if dsm is not None and np.isfinite(dsm).any() else None,
                         f"{n_ok} of {len(cells)} tiles")


NRW_BASE = "https://www.opengeodata.nrw.de/produkte/geobasis/hm"


class NrwDgm1(_TileDem):
    key = "nrw_dgm1"
    label = "Geobasis NRW DGM1/DOM1 1 m (North Rhine-Westphalia)"
    has_dsm = True
    attribution = "Geobasis NRW, DGM1 and DOM1 (dl-de/zero-2-0)"
    licence = "Datenlizenz Deutschland – Zero – Version 2.0 (https://www.govdata.de/dl-de/zero-2-0)"
    regions = ((5.85, 50.32, 9.47, 52.54),)
    max_area_km2 = 40.0            # ~2.2 MB DGM + ~3 MB DOM per km2
    sample = (50.6650, 7.2100, 3000)   # Siebengebirge

    def url(self, kind, e, n):
        prod = "dgm1" if kind == "dtm" else "dom1"
        idx = _listing(f"{NRW_BASE}/{prod}_tiff/{prod}_tiff/",
                       rf'name="({prod}_32_(\d+)_(\d+)_1_nw_\d{{4}}\.tif)"', f"nrw_{prod}")
        f = idx.get(f"{e}_{n}")
        return f"{NRW_BASE}/{prod}_tiff/{prod}_tiff/{f}" if f else None


class BavariaDgm1(_TileDem):
    key = "bvv_dgm1_by"
    label = "Bayerische Vermessungsverwaltung DGM1 1 m (Bavaria)"
    has_dsm = False
    attribution = "Datenquelle: Bayerische Vermessungsverwaltung – www.geodaten.bayern.de (CC BY 4.0)"
    licence = "CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/; https://geodaten.bayern.de/opengeodata/)"
    regions = ((8.97, 47.27, 13.84, 50.57),)
    max_area_km2 = 60.0            # ~2.7 MB per km2
    sample = (47.4700, 11.0700, 3000)   # Garmisch-Partenkirchen

    def url(self, kind, e, n):
        return f"https://download1.bayernwolke.de/a/dgm/dgm1/{e}_{n}.tif"


RLP_BASE = "https://geobasis-rlp.de/data"


class RlpDgm1(_TileDem):
    key = "lvermgeo_dgm1_rp"
    label = "LVermGeo DGM1/DOM1 1 m (Rhineland-Palatinate)"
    has_dsm = True
    attribution = f"©GeoBasis-DE / LVermGeoRP {YEAR}, dl-de/by-2-0, www.lvermgeo.rlp.de (data modified)"
    licence = "Datenlizenz Deutschland – Namensnennung – Version 2.0 (https://www.govdata.de/dl-de/by-2-0)"
    regions = ((6.10, 48.96, 8.51, 50.95),)
    max_area_km2 = 40.0
    sample = (50.1450, 7.1700, 3000)   # Cochem (Moselle valley)

    def url(self, kind, e, n):
        prod = "dgm1" if kind == "dtm" else "dom1"
        idx = _listing(f"{RLP_BASE}/{prod}/current/tif/",
                       rf'href="({prod}_32_(\d+)_(\d+)_1_rp_\d{{4}}\.tif)"', f"rlp_{prod}")
        f = idx.get(f"{e}_{n}")
        return f"{RLP_BASE}/{prod}/current/tif/{f}" if f else None


class BrandenburgDgm1(_TileDem):
    key = "lgb_dgm1_bb"
    label = "LGB DGM1 1 m (Brandenburg, Berlin)"
    has_dsm = False
    epsg = 25833
    zipped = True
    attribution = f"© GeoBasis-DE/LGB {YEAR}, dl-de/by-2-0 (data modified)"
    licence = "Datenlizenz Deutschland – Namensnennung – Version 2.0 (https://www.govdata.de/dl-de/by-2-0)"
    regions = ((11.26, 51.36, 14.77, 53.56),)
    max_area_km2 = 80.0            # ~1.3 MB per km2
    sample = (52.4030, 13.0380, 3000)   # Potsdam

    def url(self, kind, e, n):
        return f"https://data.geobasis-bb.de/geobasis/daten/dgm/tif/dgm_33{e:03d}-{n}.zip"


class SaxonyDgm1(_TileDem):
    key = "geosn_dgm1_sn"
    label = "GeoSN DGM1 1 m (Saxony)"
    has_dsm = False
    epsg = 25833
    tile_m = 2000
    zipped = True
    attribution = f"© GeoSN {YEAR}, dl-de/by-2-0 (data modified)"
    licence = "Datenlizenz Deutschland – Namensnennung – Version 2.0 (https://www.govdata.de/dl-de/by-2-0)"
    regions = ((11.87, 50.17, 15.04, 51.69),)
    max_area_km2 = 20.0            # ~14.5 MB per 2 km tile (the Flai COPC mirror also covers Saxony)
    sample = (50.9620, 14.0730, 3000)   # Bastei, Saxon Switzerland

    def url(self, kind, e, n):
        return ("https://geocloud.landesvermessung.sachsen.de/public.php/dav/files/JCcXyifaNdLDnxZ/"
                f"dgm1_33{e}_{n}_2_sn_tiff.zip")


# ---------------------------------------------------------------- Mecklenburg-Western Pomerania (WCS)

class MvWcs(DemProvider):
    key = "laiv_dgm1_mv"
    label = "LAiV DGM1/DOM1 1 m (Mecklenburg-Western Pomerania)"
    res_m = 1.0
    has_dsm = True
    attribution = f"© GeoBasis-DE/M-V {YEAR}, CC BY 4.0"
    licence = "CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/; https://www.laiv-mv.de)"
    regions = ((10.59, 53.11, 14.42, 54.69),)
    max_area_km2 = 1500.0          # the WCS resamples server-side: cost follows grid cells
    sample = (53.6200, 11.4300, 3000)   # Schwerin
    WCS = {"dtm": ("https://www.geodaten-mv.de/dienste/dgm_wcs", "mv_dgm"),
           "dsm": ("https://www.geodaten-mv.de/dienste/dom_wcs", "mv_dom1")}

    def fetch(self, aoi, grid, progress, cancel):
        bbox = src_bbox(aoi, grid, 25833)
        res = request_res(self.res_m, grid)
        out = {}
        for k, (kind, (base, cov)) in enumerate(self.WCS.items()):
            progress(k / 2, f"MV WCS {cov}")

            def tile(b, w, h, base=base, cov=cov):
                q = (f"{base}?SERVICE=WCS&REQUEST=GetCoverage&VERSION=2.0.1&COVERAGEID={cov}"
                     f"&SUBSET=x({b[0]:.2f},{b[2]:.2f})&SUBSET=y({b[1]:.2f},{b[3]:.2f})"
                     f"&SCALESIZE=x({w}),y({h})&FORMAT=image/tiff")
                return service_get(q)
            out[kind] = read_service(tile, bbox, res, 2000, aoi, grid, src_crs=25833, cancel=cancel)
        dsm = out["dsm"] if np.isfinite(out["dsm"]).any() else None
        return DemArrays(out["dtm"], dsm, f"WCS at {res:.1f} m")


# ---------------------------------------------------------------- Flai open-lidar-data (COPC)

FLAI = "https://open-lidar-data.s3.eu-central-1.amazonaws.com/data/"


def _shp_base(prefix: str) -> str:
    """URL of a dataset's bounding-box shapefile without extension (the file name does not always
    match the folder, e.g. CLSS_2023_extension -> files_bbox_CLSS_2023_aneks), via an S3 listing."""
    def make():
        r = _session().get(FLAI.rsplit("/data/", 1)[0] + "/",
                           params={"list-type": "2", "prefix": f"data/{prefix}/shp/"}, timeout=(20, 60))
        r.raise_for_status()
        keys = re.findall(r"<Key>([^<]+\.shp)</Key>", r.text)
        if not keys:
            raise RuntimeError(f"no shapefile index under {prefix}/shp/")
        return keys[0]
    key = _disk_cached("flai_shp_" + prefix.replace("/", "_"), 90, make)
    return FLAI.rsplit("/data/", 1)[0] + "/" + key[:-4]


def _shp_index(prefix: str) -> dict:
    """Bounding boxes of the files of a Flai dataset (``shp/files_bbox_*.shp``), cached as .npz,
    plus the .dbf layout for range reads of single records."""
    path = CACHE / "flai" / f"{prefix.replace('/', '_')}.npz"
    if path.exists() and time.time() - path.stat().st_mtime < 90 * 86400:
        z = np.load(path)
        return {"bbox": z["bbox"], "hlen": int(z["hlen"]), "rlen": int(z["rlen"]),
                "fpos": int(z["fpos"]), "flen": int(z["flen"]), "base": str(z["base"])}
    from ..httpcache import RANGE_CACHE
    base = _shp_base(prefix)
    s = _session()
    shx = s.get(base + ".shx", timeout=(20, 300))
    shx.raise_for_status()
    shp = s.get(base + ".shp", timeout=(20, 600))
    shp.raise_for_status()
    n = (len(shx.content) - 100) // 8
    idx = np.frombuffer(shx.content, dtype=">i4", offset=100, count=n * 2).reshape(n, 2)
    off = idx[:, 0].astype(np.int64) * 2
    raw = np.frombuffer(shp.content, dtype=np.uint8)
    stype = raw[(off + 8)[:, None] + np.arange(4)].copy().view("<i4").ravel()
    bbox = raw[(off + 12)[:, None] + np.arange(32)].copy().view("<f8").reshape(n, 4)
    bbox[stype == 0] = np.nan
    head = RANGE_CACHE.read(base + ".dbf", 0, 4096)
    nrec, hlen, rlen = struct.unpack("<IHH", head[4:12])
    if nrec != n:
        raise RuntimeError(f"{prefix}: {nrec} dbf records vs {n} shapes")
    pos, fpos, flen = 1, None, None
    for i in range(32, hlen - 1, 32):
        d = head[i:i + 32]
        if d[0] == 0x0D:
            break
        if d[:11].split(b"\x00")[0] == b"fname":
            fpos, flen = pos, d[16]
        pos += d[16]
    if fpos is None:
        raise RuntimeError(f"{prefix}: no fname field")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, bbox=bbox, hlen=hlen, rlen=rlen, fpos=fpos, flen=flen, base=base)
    return {"bbox": bbox, "hlen": hlen, "rlen": rlen, "fpos": fpos, "flen": flen, "base": base}


def _shp_extent(prefix: str) -> tuple[float, float, float, float]:
    """Overall bounding box of a dataset from the 100-byte .shp header (range-cached)."""
    from ..httpcache import RANGE_CACHE
    h = RANGE_CACHE.read(_shp_base(prefix) + ".shp", 0, 100)
    return struct.unpack("<4d", h[36:68])


# Tile.class_map (source class -> ASPRS class, applied by pointcloud) is a requested framework
# feature; providers whose class codes clash with the ASPRS reading are only registered once it exists.
_TILE_CLASS_MAP = "class_map" in {f.name for f in dataclasses.fields(Tile)}


class _FlaiCopc(PointCloudProvider):
    """Datasets, newest first: (S3 prefix under data/, label, year, CRS)."""
    datasets: tuple = ()
    class_map: dict | None = None    # non-ASPRS class codes -> ASPRS (needs Tile.class_map)

    def __init__(self):
        self._idx: dict[str, dict] = {}
        self._lock = threading.Lock()

    def _index(self, prefix):
        with self._lock:
            if prefix not in self._idx:
                self._idx[prefix] = _shp_index(prefix)
            return self._idx[prefix]

    def tiles(self, bbox, max_items):
        from pyproj import Transformer
        from ..httpcache import RANGE_CACHE
        w, s, e, n = bbox
        lon = np.linspace(w, e, 9)
        lat = np.linspace(s, n, 9)
        LO, LA = np.meshgrid(lon, lat)
        out: list[Tile] = []
        taken: list[tuple] = []          # lon/lat boxes of newer tiles already chosen
        for prefix, label, year, crs in self.datasets:   # newest first
            tf = Transformer.from_crs("EPSG:4326", crs.split("+")[0], always_xy=True)
            X, Y = tf.transform(LO.ravel(), LA.ravel())
            qx0, qy0, qx1, qy1 = np.min(X), np.min(Y), np.max(X), np.max(Y)
            ex = _shp_extent(prefix)
            if qx0 > ex[2] or qx1 < ex[0] or qy0 > ex[3] or qy1 < ex[1]:
                continue
            ix = self._index(prefix)
            b = ix["bbox"]
            hit = np.nonzero((b[:, 0] <= qx1) & (b[:, 2] >= qx0) & (b[:, 1] <= qy1) & (b[:, 3] >= qy0))[0]
            back = Transformer.from_crs(crs.split("+")[0], "EPSG:4326", always_xy=True)
            dbf = ix["base"] + ".dbf"
            for i in hit:
                x0, y0, x1, y1 = b[i]
                cx, cy = back.transform([x0, x1, x0, x1], [y0, y0, y1, y1])
                tb = (min(cx), min(cy), max(cx), max(cy))
                mx, my = (tb[0] + tb[2]) / 2, (tb[1] + tb[3]) / 2
                if any(t[0] <= mx <= t[2] and t[1] <= my <= t[3] for t in taken):
                    continue     # a newer dataset already covers this spot
                rec = RANGE_CACHE.read(dbf, ix["hlen"] + int(i) * ix["rlen"] + ix["fpos"], ix["flen"])
                fname = rec.decode("utf-8", "replace").strip()
                if not fname.endswith(".copc.laz"):
                    continue
                extra = {"class_map": self.class_map} if self.class_map and _TILE_CLASS_MAP else {}
                out.append(Tile(f"{FLAI}{prefix}/copc/{fname}", label, year, tb, crs=crs, provider=self.key,
                                **extra))
                if len(out) >= max_items:
                    return out
            taken += [t.bbox for t in out if t.project == label]
        return out


MIRROR = "via the Flai open-lidar-data mirror (third party, https://github.com/flai-ai/open-lidar-data)"


class FlaiSlovenia(_FlaiCopc):
    key = "flai_si"
    label = "Slovenia LiDAR (GURS CLSS / ARSO LSS, COPC mirror)"
    attribution = ("Geodetska uprava Republike Slovenije, CLSS 2023-2024 LiDAR; ARSO LiDAR 2011-2015; CC BY 4.0; "
                   + MIRROR)
    licence = "CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/deed.sl)"
    regions = ((13.36, 45.42, 16.61, 46.88),)
    sample = (46.0440, 14.7222, 600)
    datasets = (
        ("SI/GURS/CLSS_2024", "GURS CLSS 2024 (Flai mirror)", 2024, "EPSG:3794"),
        ("SI/GURS/CLSS_2023", "GURS CLSS 2023 (Flai mirror)", 2023, "EPSG:3794"),
        ("SI/GURS/CLSS_2023_extension", "GURS CLSS 2023 extension (Flai mirror)",
         2023, "EPSG:3794"),
        ("SI/GURS/LSS_2011-2015", "ARSO LSS 2011-2015 (Flai mirror)", 2014, "EPSG:3794"),
    )


class FlaiPoland(_FlaiCopc):
    key = "flai_pl"
    label = "Poland LiDAR (GUGiK, COPC mirror)"
    attribution = "Główny Urząd Geodezji i Kartografii (GUGiK), LiDAR point clouds 2018-2023; " + MIRROR
    licence = ("Polish state geodetic data, free re-use without restrictions "
               "(https://www.geoportal.gov.pl/pl/dane/dane-pomiarowe-lidar-lidar/)")
    regions = ((14.12, 49.00, 24.15, 54.84),)
    sample = (50.4328, 20.7888, 600)
    datasets = tuple(
        (f"PL/GUGiK/Skaning_laserowy_{y}", f"GUGiK LiDAR {y} (Flai mirror)", y,
         "EPSG:2177" if y == 2018 else "EPSG:2180")
        for y in range(2023, 2017, -1))


class FlaiSaxony(_FlaiCopc):
    """GeoSN classes: 2 ground, 20 NON-ground (buildings + vegetation), 8 / 30 interpolated ground
    (water / under buildings). ``pointcloud.GROUND`` reads 20 as ground (3DEP meaning), so this
    provider is registered only when Tile.class_map exists to remap 20 -> 1."""
    class_map = {20: 1, 30: 2, 62: 1}
    key = "flai_sn"
    label = "Saxony LiDAR (GeoSN, COPC mirror)"
    attribution = "© GeoSN, Laserscandaten 2015-2023, dl-de/by-2-0; " + MIRROR
    licence = "Datenlizenz Deutschland – Namensnennung – Version 2.0 (https://www.govdata.de/dl-de/by-2-0)"
    regions = ((11.87, 50.17, 15.04, 51.69),)
    sample = (51.0500, 13.7400, 600)    # Dresden
    datasets = (("DE/GeoSN/lidar_2015-2023", "GeoSN LiDAR 2015-2023 (Flai mirror)", 2020,
                 "EPSG:25833"),)


class FlaiWallonia(_FlaiCopc):
    key = "flai_be_wal"
    label = "Wallonia LiDAR 2021-2022 (SPW, COPC mirror)"
    attribution = "© SPW – Service public de Wallonie, LIDAR 2021-2022, CC BY 4.0; " + MIRROR
    licence = "CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/deed.fr)"
    regions = ((2.84, 49.49, 6.41, 50.82),)
    sample = (50.0421, 4.6792, 600)     # southern Namur province (the mirror covers parts only)
    datasets = tuple((f"BE/SPW/{p}_2021-2022", f"SPW LIDAR 2021-2022 {p.title()} (Flai mirror)",
                      2022, "EPSG:3812")
                     for p in ("NAMUR", "LIEGE", "BRABANT", "HAINAUT", "LUXEMBOURG"))


for _p in (BevAustria(), LglnNiedersachsen(), NrwDgm1(), BavariaDgm1(), RlpDgm1(), BrandenburgDgm1(),
           SaxonyDgm1(), MvWcs()):
    register(_p)
for _p in (FlaiSlovenia(), FlaiPoland(), FlaiWallonia()):
    register_pointcloud(_p)
if _TILE_CLASS_MAP:
    register_pointcloud(FlaiSaxony())
