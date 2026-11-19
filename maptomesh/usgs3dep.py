"""USGS 3D Elevation Program (3DEP) LiDAR point clouds via Microsoft Planetary Computer.

Planetary Computer serves the 3DEP point clouds as COPC tiles (~1 km, each in its project's
own CRS) behind a STAC API, so they go through the same remote COPC reader and range cache as
NRCan's CanElevation tiles. Access is anonymous; blob URLs need a SAS token, which is fetched
here and added per request by the range cache (cache keys stay token-free).

Classification follows the 3DEP base specification: ground (2), water (9), bridge decks (17),
noise (7/18) and "ignored ground" near breaklines (20) are classified; buildings and trees
usually stay in class 1, which the surface builder routes with OpenStreetMap footprints.
The collection is a snapshot (projects up to ~2022).
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from datetime import datetime, timezone

import requests

from . import config
from .httpcache import register_signer
from .lidar_index import Tile

BLOB_PREFIX = "https://usgslidareuwest.blob.core.windows.net/"

# lon/lat boxes of 3DEP coverage: contiguous US, Alaska, Hawaii, Puerto Rico + USVI
US_REGIONS = (
    (-125.0, 24.0, -66.5, 49.5),
    (-180.0, 51.0, -129.5, 71.6),
    (-160.5, 18.8, -154.6, 22.4),
    (-67.5, 17.5, -64.4, 18.6),
)

_FIELDS = {"include": ["id", "bbox", "properties.3dep:usgs_id", "properties.start_datetime",
                       "assets.data.href"],
           "exclude": ["links", "geometry"]}


def in_us(bbox: tuple[float, float, float, float]) -> bool:
    w, s, e, n = bbox
    return any(w <= rx1 and e >= rx0 and s <= ry1 and n >= ry0 for rx0, ry0, rx1, ry1 in US_REGIONS)


def _session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = config.USER_AGENT
    s.mount("https://", requests.adapters.HTTPAdapter(max_retries=requests.adapters.Retry(
        total=4, connect=4, backoff_factor=1.0, status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=None)))  # token GET and the read-only search POST are safe to retry
    return s


# ---------------------------------------------------------------- SAS token

_TOKEN_LOCK = threading.Lock()
_token: dict = {"token": None, "expiry": 0.0}


def _parse_expiry(s: str | None) -> float:
    if not s:
        return time.time() + 1800
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc).timestamp()
    except ValueError:
        return time.time() + 1800


def sign(url: str, refresh: bool = False) -> str:
    """``url`` with a current read-only SAS token for the 3DEP COPC container."""
    with _TOKEN_LOCK:
        if refresh or _token["token"] is None or _token["expiry"] - time.time() < 120:
            r = _session().get(f"{config.PC_SAS_API}/{config.USGS_COPC_COLLECTION}", timeout=(20, 60))
            r.raise_for_status()
            d = r.json()
            _token["token"] = d["token"]
            _token["expiry"] = _parse_expiry(d.get("msft:expiry"))
        tok = _token["token"]
    return f"{url.split('?', 1)[0]}?{tok}"


register_signer(BLOB_PREFIX, sign)


# ---------------------------------------------------------------- STAC search

def _search(bbox: tuple[float, float, float, float], max_items: int) -> list[dict]:
    body = {"collections": [config.USGS_COPC_COLLECTION], "bbox": list(bbox), "limit": 1000,
            "fields": _FIELDS}
    url, method = f"{config.PC_STAC_API}/search", "POST"
    s = _session()
    feats: list[dict] = []
    while url and len(feats) < max_items:
        kw = {"timeout": (20, 90)}
        r = s.post(url, json=body, **kw) if method == "POST" else s.get(url, **kw)
        r.raise_for_status()
        d = r.json()
        feats += d.get("features", [])
        nxt = next((ln for ln in d.get("links", []) if ln.get("rel") == "next"), None)
        if nxt is None:
            break
        url, method = nxt["href"], nxt.get("method", "GET").upper()
        if method == "POST":
            body = {**body, **nxt.get("body", {})} if nxt.get("merge") else nxt.get("body", body)
    return feats


def _cached_search(bbox: tuple[float, float, float, float], max_items: int) -> list[dict]:
    """Search a slightly enlarged, rounded box (cache hits while the area is nudged around),
    cached on disk for a day."""
    q = 0.01
    rb = (math.floor(bbox[0] / q) * q, math.floor(bbox[1] / q) * q,
          math.ceil(bbox[2] / q) * q, math.ceil(bbox[3] / q) * q)
    rb = tuple(round(v, 4) for v in rb)
    key = hashlib.sha1(f"{config.USGS_COPC_COLLECTION}|{rb}|{max_items}".encode()).hexdigest()[:16]
    cache = config.CACHE_DIR / "stac" / f"pc_{key}.json"
    if cache.exists() and time.time() - cache.stat().st_mtime < 86400:
        return json.loads(cache.read_text())
    feats = _search(rb, max_items)
    slim = [{"id": f["id"], "bbox": f["bbox"],
             "usgs_id": f.get("properties", {}).get("3dep:usgs_id"),
             "start": f.get("properties", {}).get("start_datetime"),
             "href": f.get("assets", {}).get("data", {}).get("href")} for f in feats]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(slim))
    return slim


def _year(start: str | None, name: str) -> int:
    if start and start[:4].isdigit():
        return int(start[:4])
    digits = [int(p) for p in name.replace("-", "_").split("_") if p.isdigit() and len(p) == 4]
    return digits[-1] if digits else 0


def tiles(bbox: tuple[float, float, float, float], max_items: int = 5000) -> list[Tile]:
    """3DEP COPC tiles whose bounding box intersects ``bbox`` (lon/lat). URLs are unsigned."""
    if not in_us(bbox):
        return []
    w, s, e, n = bbox
    out = []
    for f in _cached_search(bbox, max_items):
        b = f["bbox"]
        if not f.get("href") or b[0] > e or b[2] < w or b[1] > n or b[3] < s:
            continue
        name = f.get("usgs_id") or f["id"]
        label = name[len("USGS_LPC_"):] if name.startswith("USGS_LPC_") else name
        out.append(Tile(f["href"].split("?", 1)[0], f"USGS {label}", _year(f.get("start"), name),
                        tuple(float(v) for v in b[:4]), provider="usgs"))
    return out


def footprints(bbox: tuple[float, float, float, float], max_items: int = 4000) -> dict:
    """GeoJSON of tile footprints (bounding boxes) for the coverage map layer."""
    feats = []
    for t in tiles(bbox, max_items):
        w, s, e, n = t.bbox
        feats.append({"type": "Feature",
                      "properties": {"project": t.project, "year": t.year},
                      "geometry": {"type": "Polygon",
                                   "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]]}})
    return {"type": "FeatureCollection", "features": feats}
