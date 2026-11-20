"""Global models: Copernicus DEM GLO-30 (GLO-90 where a 30 m tile is not public), FABDEM and
ALOS AW3D30.

* Copernicus GLO-30: COGs on AWS Open Data, one 1x1 degree tile per file, named after the tile's
  south-west corner. Surface model (EGM2008 heights): trees and buildings are part of the heights.
* FABDEM V1-2 (University of Bristol / Fathom): Copernicus GLO-30 with forests and buildings
  removed, i.e. a 30 m bare-earth model, 60S-80N. Non-commercial licence (warning with the model).
  Read from the Fondazione LINKS COG copy on Hugging Face (anonymous HTTPS, 1x1 degree tiles with
  overviews); its STAC collection lists which tiles exist (ocean has none). The sea is stored as
  0 m and treated as no data, so a coastal model takes its sea from Copernicus as before. Being bare
  earth, FABDEM ranks before the Copernicus surface model at equal effective resolution.
* ALOS World 3D-30m (AW3D30, JAXA): 30 m surface model, COGs on Microsoft Planetary Computer
  (anonymous SAS token per container). JAXA asks commercial users to notify it (warning with the
  model). Ranked after Copernicus: a fallback when Copernicus can't be read.
"""

from __future__ import annotations

import json
import math
import threading
import time

import numpy as np
import requests

from .. import config
from ..dem import DemArrays, DemProvider, register
from ..dem_readers import merge_fill, read_cog, service_get
from ..httpcache import register_signer

ATTRIBUTION = ("produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH "
               "2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved")


def _ns(lat: int, digits: int = 2) -> str:
    return f"N{lat:0{digits}d}" if lat >= 0 else f"S{-lat:0{digits}d}"


def _ew(lon: int) -> str:
    return f"E{lon:03d}" if lon >= 0 else f"W{-lon:03d}"


def _tile_name(lat0: int, lon0: int, res_code: str) -> str:
    return f"Copernicus_DSM_COG_{res_code}_{_ns(lat0)}_00_{_ew(lon0)}_00_DEM"


def _cells(aoi, grid) -> list[tuple[int, int]]:
    """1x1 degree cells (south-west corners) under the AOI."""
    w, s, e, n = aoi.lonlat_bbox(margin=grid.step * 3)
    return [(la, lo) for la in range(math.floor(s), math.floor(n) + 1)
            for lo in range(math.floor(w), math.floor(e) + 1)]


class CopernicusGlo30(DemProvider):
    key = "copernicus_glo30"
    label = "Copernicus DEM 30 m (global)"
    res_m = 30.0
    dsm_only = True
    attribution = ATTRIBUTION
    licence = "Copernicus DEM licence, free for any use with the notice above (https://registry.opendata.aws/copernicus-dem/)"
    max_area_km2 = 50_000.0
    sample = (27.9881, 86.9250, 20000)   # Everest

    def fetch(self, aoi, grid, progress, cancel):
        out = np.full(grid.vshape, np.nan)
        cells = _cells(aoi, grid)
        used = []
        for k, (la, lo) in enumerate(cells):
            if cancel.is_set():
                raise RuntimeError("Cancelled")
            for bucket, code in (("copernicus-dem-30m", "10"), ("copernicus-dem-90m", "30")):
                name = _tile_name(la, lo, code)
                url = f"https://{bucket}.s3.amazonaws.com/{name}/{name}.tif"
                progress(k / len(cells), f"Copernicus tile {name}")
                try:
                    merge_fill(out, read_cog(url, aoi, grid))
                    used.append(name)
                    break
                except Exception:  # noqa: BLE001 - ocean tiles don't exist; try GLO-90, then give up
                    continue
        return DemArrays(out, None, ", ".join(used))

    def cog_urls(self, aoi, grid, dsm):
        return [f"https://copernicus-dem-30m.s3.amazonaws.com/{n}/{n}.tif"
                for n in (_tile_name(la, lo, "10") for la, lo in _cells(aoi, grid))]


# ---------------------------------------------------------------- FABDEM (non-commercial)

FABDEM_BASE = "https://huggingface.co/buckets/links-ads/fabdem/resolve/"
FABDEM_INDEX = config.CACHE_DIR / "global" / "fabdem_tiles.json"
FABDEM_INDEX_DAYS = 90
FABDEM_TERMS = "https://data.bris.ac.uk/data/dataset/s5hqmjcdj8yo2ibzi9b4ew3sn"
FABDEM_NOTE = ("FABDEM V1-2 (University of Bristol / Fathom) is licensed for non-commercial use only "
               "(CC BY-NC-SA 4.0): don't sell prints that use it without a commercial licence from Fathom "
               f"(fabdem@fathom.global), and share derived files under the same licence; terms: {FABDEM_TERMS}")
_fabdem_lock = threading.Lock()


def _fabdem_tiles() -> set[str]:
    """Names (``N47W053``) of the FABDEM tiles that exist, from the mirror's STAC collection
    (2 MB, kept for 90 days; a stale copy is used if the mirror can't be reached)."""
    with _fabdem_lock:
        stale = None
        if FABDEM_INDEX.exists():
            try:
                stale = set(json.loads(FABDEM_INDEX.read_text()))
            except ValueError:
                stale = None
            if stale and time.time() - FABDEM_INDEX.stat().st_mtime < FABDEM_INDEX_DAYS * 86400:
                return stale
        try:
            coll = json.loads(service_get(FABDEM_BASE + "collection.json", timeout=120, cache=False, allow_xml=True))
        except Exception:  # noqa: BLE001
            if stale:
                return stale
            raise
        names = {lk["href"].rsplit("/", 1)[-1].split("_")[0] for lk in coll.get("links", [])
                 if lk.get("rel") == "item"}
        if len(names) < 1000:
            raise RuntimeError(f"FABDEM index has only {len(names)} tiles")
        config.atomic_write(FABDEM_INDEX, json.dumps(sorted(names)))
        return names


def fabdem_url(lat0: int, lon0: int) -> str:
    """COG of the 1x1 degree tile with south-west corner (lat0, lon0), in its 10x10 degree folder,
    e.g. ``tiles/N40W060-N50W050_FABDEM_V1-2/N47W053_FABDEM_V1-2.tif``."""
    la, lo = (lat0 // 10) * 10, (lon0 // 10) * 10
    folder = f"{_ns(la)}{_ew(lo)}-{_ns(la + 10)}{_ew(lo + 10)}_FABDEM_V1-2"
    return f"{FABDEM_BASE}tiles/{folder}/{_ns(lat0)}{_ew(lon0)}_FABDEM_V1-2.tif"


class Fabdem(DemProvider):
    key = "fabdem"
    label = "FABDEM 30 m bare earth (global, non-commercial)"
    res_m = 30.0
    attribution = ("FABDEM V1-2 © University of Bristol and Fathom (Hawker et al. 2022, "
                   "https://doi.org/10.5523/bris.s5hqmjcdj8yo2ibzi9b4ew3sn), CC BY-NC-SA 4.0; " + ATTRIBUTION)
    licence = "CC BY-NC-SA 4.0, non-commercial (https://creativecommons.org/licenses/by-nc-sa/4.0/)"
    licence_note = FABDEM_NOTE
    regions = ((-180.0, -60.0, 180.0, 80.0),)
    max_area_km2 = 50_000.0
    sample = (-3.0700, 37.3500, 20000)   # Kilimanjaro (forested slopes)
    prefer = 2

    def fetch(self, aoi, grid, progress, cancel):
        have = _fabdem_tiles()
        out = np.full(grid.vshape, np.nan)
        cells = [(la, lo) for la, lo in _cells(aoi, grid) if f"{_ns(la)}{_ew(lo)}" in have]
        used = []
        for k, (la, lo) in enumerate(cells):
            if cancel.is_set():
                raise RuntimeError("Cancelled")
            name = f"{_ns(la)}{_ew(lo)}"
            progress(k / max(len(cells), 1), f"FABDEM tile {name}")
            # nodata -9999; the sea is stored as exactly 0 m: no data here, so Copernicus keeps it
            merge_fill(out, read_cog(fabdem_url(la, lo), aoi, grid, nodata=-9999.0))
            used.append(name)
        out[out == 0.0] = np.nan
        return DemArrays(out, None, ", ".join(used))

    def cog_urls(self, aoi, grid, dsm):
        have = _fabdem_tiles()
        return [fabdem_url(la, lo) for la, lo in _cells(aoi, grid) if f"{_ns(la)}{_ew(lo)}" in have]


# ---------------------------------------------------------------- ALOS AW3D30 (JAXA)

PC_BLOB = "https://ai4edataeuwest.blob.core.windows.net/alos-dem/"
AW3D30_COLLECTION = "alos-dem"
AW3D30_TERMS = "https://www.eorc.jaxa.jp/ALOS/en/dataset/aw3d30/aw3d30_e.htm"
AW3D30_NOTE = ("ALOS World 3D-30m (AW3D30, © JAXA) is free for commercial use, but JAXA's terms ask commercial "
               "users to notify JAXA in advance and to show '©JAXA' with the product: notify JAXA before selling "
               f"prints that use it; terms: {AW3D30_TERMS}")
_pc_lock = threading.Lock()
_pc_token: dict = {"token": None, "expiry": 0.0}


def _pc_sign(url: str, refresh: bool = False) -> str:
    """``url`` with a current anonymous read-only SAS token for Planetary Computer's alos-dem container."""
    from ..usgs3dep import _parse_expiry
    with _pc_lock:
        if refresh or _pc_token["token"] is None or _pc_token["expiry"] - time.time() < 120:
            r = requests.get(f"{config.PC_SAS_API}/{AW3D30_COLLECTION}", timeout=(20, 60),
                             headers={"User-Agent": config.USER_AGENT})
            r.raise_for_status()
            d = r.json()
            _pc_token["token"] = d["token"]
            _pc_token["expiry"] = _parse_expiry(d.get("msft:expiry"))
        tok = _pc_token["token"]
    return f"{url.split('?', 1)[0]}?{tok}"


register_signer(PC_BLOB, _pc_sign)


class Aw3d30(DemProvider):
    key = "jaxa_aw3d30"
    label = "ALOS World 3D 30 m (JAXA, global)"
    res_m = 30.0
    dsm_only = True
    attribution = "ALOS World 3D-30m (AW3D30) ©JAXA, via Microsoft Planetary Computer"
    licence = f"JAXA AW3D30 terms of use: free, commercial use with prior notice to JAXA ({AW3D30_TERMS})"
    licence_note = AW3D30_NOTE
    regions = ((-180.0, -84.0, 180.0, 84.0),)
    max_area_km2 = 50_000.0
    sample = (35.3606, 138.7274, 20000)   # Mt Fuji
    prefer = 2

    def fetch(self, aoi, grid, progress, cancel):
        out = np.full(grid.vshape, np.nan)
        cells = _cells(aoi, grid)
        used, errors = [], []
        for k, (la, lo) in enumerate(cells):
            if cancel.is_set():
                raise RuntimeError("Cancelled")
            name = f"ALPSMLC30_{_ns(la, 3)}{_ew(lo)}_DSM"
            progress(k / len(cells), f"AW3D30 tile {name}")
            try:
                merge_fill(out, read_cog(f"{PC_BLOB}AW3D30_global/{name}.tif", aoi, grid))
                used.append(name)
            except Exception as e:  # noqa: BLE001
                if "404" not in str(e):   # 404: an ocean tile, which doesn't exist
                    errors.append(f"{name}: {type(e).__name__}: {str(e)[:120]}")
        if not used and errors:
            raise RuntimeError(errors[0])
        return DemArrays(out, None, ", ".join(used))


register(CopernicusGlo30())
register(Fabdem())
register(Aw3d30())
