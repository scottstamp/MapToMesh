"""HRDEM (1 m / 2 m LiDAR-derived DTM + DSM mosaics) STAC search (CCMEO STAC API).

Read through ``maptomesh.providers.north_america.Hrdem`` like every other elevation model.
"""

from __future__ import annotations

import hashlib
import json
import time

import requests

from . import config

COLLECTIONS = ("hrdem-mosaic-1m", "hrdem-mosaic-2m")


def stac_items(bbox: tuple[float, float, float, float], collection: str) -> list[dict]:
    key = hashlib.sha1(f"{collection}|{bbox}".encode()).hexdigest()[:16]
    cache = config.CACHE_DIR / "stac" / f"{key}.json"
    if cache.exists() and time.time() - cache.stat().st_mtime < 86400:
        return json.loads(cache.read_text())
    r = requests.get(
        f"{config.STAC_API}/search",
        params={"collections": collection, "bbox": ",".join(f"{v:.6f}" for v in bbox), "limit": 50},
        headers={"User-Agent": config.USER_AGENT}, timeout=60,
    )
    r.raise_for_status()
    feats = r.json().get("features", [])
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(feats))
    return feats
