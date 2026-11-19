"""Nordic, Baltic and Arctic elevation models (no accounts or API keys needed).

* Finland: NLS Korkeusmalli 2 m (LiDAR DTM). 6 x 6 km tiled GeoTIFFs on the Funet mirror; the tile
  index is NLS's own national VRT (``dem2m_direct.vrt``), parsed once and cached on disk.
* Estonia: Maa- ja Ruumiamet 1 m DTM, one COG per 1:10 000 map sheet (5 x 5 km, L-EST97). The sheet
  number follows from the coordinates, so the download URL is built directly (no HTML scraping).
* ArcticDEM v4.1 2 m mosaic (Polar Geospatial Center, PGC STAC API): satellite-stereo surface model
  for everything north of ~59 N plus Alaska and Kamchatka (Norway, Sweden, Finland, Iceland,
  Greenland, Faroes, Svalbard, northern Canada...). Heights are WGS84-ellipsoidal; they are converted
  to EGM2008 heights with NGA's geoid grid (PROJ CDN COG) so seams to national models stay small.
* Iceland: ÍslandsDEM v1.0 10 m (national model built from ArcticDEM, gap-filled, ISN2016 heights).

Not here (see the research notes): Norway (Kartverket services unreachable from outside Europe at the
time of writing), Sweden / Denmark (account or token required), Latvia (uncompressed 160 MB LAS
tiles only), Lithuania (order-based).
"""

from __future__ import annotations

import json
import re
import time

import numpy as np

from .. import config
from ..dem import DemArrays, DemProvider, register
from ..dem_readers import merge_fill, read_cog, service_get, src_bbox

INDEX_DIR = config.CACHE_DIR / "dem_index"
INDEX_DAYS = 30


def _check(cancel) -> None:
    if cancel.is_set():
        raise RuntimeError("Cancelled")


def _load_json(name: str, days: float = INDEX_DAYS):
    path = INDEX_DIR / name
    if path.exists() and time.time() - path.stat().st_mtime < days * 86400:
        try:
            return json.loads(path.read_text())
        except ValueError:
            return None
    return None


def _save_json(name: str, obj) -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    path = INDEX_DIR / name
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj))
    tmp.replace(path)


# ---------------------------------------------------------------- Finland

FI_BASE = "https://www.nic.funet.fi/index/geodata/mml/dem2m/"
FI_VRT = FI_BASE + "dem2m_direct.vrt"
_FI_INDEX: dict | None = None


def _fi_index() -> dict:
    """{'paths': [...], 'bounds': [[x0, y0, x1, y1], ...]} in EPSG:3067 from NLS's national VRT."""
    global _FI_INDEX
    if _FI_INDEX is not None:
        return _FI_INDEX
    idx = _load_json("fi_dem2m_index.json")
    if idx is None:
        text = service_get(FI_VRT, timeout=180).decode("utf-8", "replace")
        gt = [float(v) for v in re.search(r"<GeoTransform>([^<]+)</GeoTransform>", text).group(1).split(",")]
        ox, px, _, oy, _, py = gt
        paths, bounds = [], []
        for m in re.finditer(r"<SourceFilename[^>]*>([^<]+)</SourceFilename>.*?"
                             r'<DstRect xOff="([\d.]+)" yOff="([\d.]+)" xSize="([\d.]+)" ySize="([\d.]+)"',
                             text, re.S):
            path, xo, yo, xs, ys = m.group(1), *(float(v) for v in m.groups()[1:])
            x0, y1 = ox + xo * px, oy + yo * py
            x1, y0 = x0 + xs * px, y1 + ys * py
            paths.append(path)
            bounds.append([x0, y0, x1, y1])
        if not paths:
            raise RuntimeError("NLS 2 m index (VRT) has no tiles")
        idx = {"paths": paths, "bounds": bounds}
        _save_json("fi_dem2m_index.json", idx)
    _FI_INDEX = idx
    return idx


class NlsFinland2m(DemProvider):
    key = "nls_fi_dem2m"
    label = "NLS 2 m (Finland)"
    res_m = 2.0
    attribution = "Contains data from the National Land Survey of Finland Elevation model 2 m (NLS, CC BY 4.0)."
    licence = "CC BY 4.0 (https://www.maanmittauslaitos.fi/en/opendata-licence-cc40)"
    regions = ((19.0, 59.7, 31.6, 70.1),)
    max_area_km2 = 400.0      # tiles have no overviews: everything is read at 2 m
    sample = (63.0960, 29.8060, 3000)   # Koli

    def fetch(self, aoi, grid, progress, cancel):
        idx = _fi_index()
        x0, y0, x1, y1 = src_bbox(aoi, grid, 3067)
        b = np.asarray(idx["bounds"])
        hit = np.where((b[:, 0] < x1) & (b[:, 2] > x0) & (b[:, 1] < y1) & (b[:, 3] > y0))[0]
        out = np.full(grid.vshape, np.nan)
        used = []
        for k, i in enumerate(hit):
            _check(cancel)
            path = idx["paths"][i]
            name = path.rsplit("/", 1)[-1].removesuffix(".tif")
            progress(k / max(len(hit), 1), f"NLS 2 m tile {name}")
            merge_fill(out, read_cog(FI_BASE + path, aoi, grid))
            used.append(name)
        return DemArrays(out, None, ", ".join(used))


# ---------------------------------------------------------------- Estonia

EE_URL = ("https://geoportaal.maaruum.ee/index.php?lang_id=1&plugin_act=otsing&kaardiruut={s}"
          "&andmetyyp=dem_1m_geotiff&dl=1&f={s}_dtm_1m.tif&page_id=614")
EE_X = (365_000, 740_000)       # extent of the 1:10 000 sheet grid (L-EST97, EPSG:3301)
EE_Y = (6_370_000, 6_635_000)


def ee_sheet(x: float, y: float) -> str:
    """1:10 000 map sheet containing L-EST97 point (x = easting, y = northing).

    The 1:20 000 sheet (10 x 10 km) is 4 digits from the 10 km row n = y // 10 km and column
    e = x // 10 km: (n // 10 - 59)(e // 10 - 2)(n % 10)(e % 10); the 5th digit is the 5 km quadrant
    (1 SW, 2 SE, 3 NW, 4 NE). Checked against the official sheet grid (epk10T): 2354 of 2357 sheets
    match; the 3 others are sheets without data (labelling quirks in the grid file)."""
    n, e = int(y // 10_000), int(x // 10_000)
    q = 1 + (1 if x % 10_000 >= 5_000 else 0) + (2 if y % 10_000 >= 5_000 else 0)
    return f"{n // 10 - 59}{e // 10 - 2}{n % 10}{e % 10}{q}"


class MaruEstonia1m(DemProvider):
    key = "maru_ee_dtm1m"
    label = "Maa- ja Ruumiamet 1 m (Estonia)"
    res_m = 1.0
    attribution = "Maa- ja Ruumiamet, maapinna kõrgusmudel 1 m (digital terrain model), open data."
    licence = ("Maa- ja Ruumiameti avaandmete litsents (CC BY 4.0 equivalent, commercial use allowed) "
               "(https://geoportaal.maaruum.ee/opendata-licence)")
    regions = ((21.6, 57.45, 28.3, 59.8),)
    max_area_km2 = 2500.0     # COGs with overviews
    sample = (57.7144, 27.0592, 3000)   # Suur Munamägi

    _missing: set[str] = set()

    def fetch(self, aoi, grid, progress, cancel):
        if not self._missing:
            self._missing.update(_load_json("ee_dtm1m_missing.json") or [])
        x0, y0, x1, y1 = src_bbox(aoi, grid, 3301)
        sheets = []
        for y in range(int(max(y0, EE_Y[0]) // 5000) * 5000, int(min(y1, EE_Y[1] - 1)) + 1, 5000):
            for x in range(int(max(x0, EE_X[0]) // 5000) * 5000, int(min(x1, EE_X[1] - 1)) + 1, 5000):
                sheets.append(ee_sheet(x + 1, y + 1))
        if len(sheets) > 150:
            raise RuntimeError(f"{len(sheets)} map sheets (limit 150)")
        out = np.full(grid.vshape, np.nan)
        used, failed, new_missing = [], [], False
        for k, s in enumerate(sheets):
            _check(cancel)
            if s in self._missing:
                continue
            progress(k / max(len(sheets), 1), f"Estonia 1 m DTM sheet {s}")
            url = EE_URL.format(s=s)
            try:
                part = read_cog(url, aoi, grid)
            except Exception:  # noqa: BLE001
                _check(cancel)
                if _ee_no_file(url):   # no DTM for this sheet (sea, border): remember it
                    self._missing.add(s)
                    new_missing = True
                else:                  # transient failure: leave the hole to the fallbacks
                    failed.append(s)
                continue
            merge_fill(out, part)
            used.append(s)
        if new_missing:
            _save_json("ee_dtm1m_missing.json", sorted(self._missing))
        return DemArrays(out, None, ", ".join(used) + (f" (failed: {', '.join(failed)})" if failed else ""))


def _ee_no_file(url: str) -> bool:
    """The download script answers a missing sheet with an HTML page (HTTP 200), not a 404."""
    import requests
    try:
        r = requests.head(url, headers={"User-Agent": config.USER_AGENT}, timeout=30, allow_redirects=True)
    except requests.RequestException:
        return False
    return r.status_code == 404 or (r.ok and r.headers.get("Content-Type", "").startswith("text/html"))


# ---------------------------------------------------------------- ArcticDEM

PGC_STAC = "https://stac.pgc.umn.edu/api/v1/search"
ADEM_COLLECTION = "arcticdem-mosaics-v4.1-2m"
EGM2008 = "https://cdn.proj.org/us_nga_egm08_25.tif"   # NGA EGM2008 geoid undulation, public domain


def geoid_egm2008(aoi, grid) -> np.ndarray:
    """EGM2008 geoid height (ellipsoidal minus orthometric) on the vertex grid."""
    n = read_cog(EGM2008, aoi, grid)
    if not np.isfinite(n).all():
        raise RuntimeError("EGM2008 geoid grid not available for this area")
    return n


class ArcticDem2m(DemProvider):
    key = "pgc_arcticdem_2m"
    label = "ArcticDEM 2 m (Arctic, surface model)"
    res_m = 2.0
    dsm_only = True
    attribution = ("ArcticDEM v4.1 (Porter et al. 2023, doi:10.7910/DVN/3VDC4W): DEMs provided by the Polar "
                   "Geospatial Center under NSF-OPP awards 1043681, 1559691, 1542736, 1810976, and 2129685.")
    licence = ("CC BY 4.0 (PGC STAC catalogue; https://www.pgc.umn.edu/guides/user-services/acknowledgement-policy/)")
    # real mosaic extent (100 km supertiles): everything north of ~59 N, Alaska / Aleutians down to
    # ~51 N, Kamchatka and the Kurils
    regions = ((-180.0, 59.0, 180.0, 84.0), (-180.0, 51.0, -129.0, 59.0), (150.0, 50.7, 180.0, 59.0))
    max_area_km2 = 20_000.0   # COGs with overviews
    sample = (69.6460, 18.9550, 3000)   # Tromsø, Norway

    def fetch(self, aoi, grid, progress, cancel):
        w, s, e, n = aoi.lonlat_bbox(margin=grid.step * 3)
        body = service_get(PGC_STAC, {"collections": ADEM_COLLECTION, "limit": 100,
                                      "bbox": ",".join(f"{v:.5f}" for v in (w, s, e, n))})
        items = json.loads(body).get("features", [])
        out = np.full(grid.vshape, np.nan)
        used = []
        for k, it in enumerate(items):
            _check(cancel)
            href = it.get("assets", {}).get("dem", {}).get("href")
            if not href:
                continue
            progress(k / max(len(items), 1), f"ArcticDEM tile {it['id']}")
            merge_fill(out, read_cog(href, aoi, grid))
            used.append(it["id"])
        if np.isfinite(out).any():
            _check(cancel)
            out -= geoid_egm2008(aoi, grid)   # WGS84 ellipsoid -> EGM2008 heights
        return DemArrays(out, None, ", ".join(used))


# ---------------------------------------------------------------- Iceland

IS_DEM10 = "https://ftp.lmi.is/gisdata/raster/IslandsDEMv1.0_10x10m_isn2016_zmasl.tif"


class IslandsDem10m(DemProvider):
    key = "natt_islandsdem_10m"
    label = "ÍslandsDEM 10 m (Iceland)"
    res_m = 10.0
    dsm_only = True   # derived from the ArcticDEM surface model (edited, but trees / buildings remain)
    attribution = "ÍslandsDEM v1.0 © Náttúrufræðistofnun (formerly Landmælingar Íslands), CC BY 4.0."
    licence = "CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)"
    regions = ((-24.6, 63.2, -13.4, 66.6),)
    max_area_km2 = 1500.0     # one national file without overviews: read at 10 m
    sample = (63.9920, -19.6660, 6000)   # Hekla

    def fetch(self, aoi, grid, progress, cancel):
        _check(cancel)
        progress(0.0, "ÍslandsDEM 10 m")
        out = read_cog(IS_DEM10, aoi, grid)
        out[out < -1000] = np.nan   # nodata is -3.4e38 (float min); guard against resampling edges
        return DemArrays(out, None, "IslandsDEMv1.0_10x10m_isn2016")


register(NlsFinland2m())
register(MaruEstonia1m())
register(ArcticDem2m())
register(IslandsDem10m())
