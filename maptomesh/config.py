"""Paths and remote endpoints."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = Path(os.environ.get("MAPTOMESH_CACHE", ROOT / "cache"))
WEB_DIR = ROOT / "web"
OUTPUT_DIR = CACHE_DIR / "output"
ELEV_CACHE_DIR = CACHE_DIR / "elevation"
OSM_CACHE_DIR = CACHE_DIR / "osm"
# persistent app data (user reviews, incl. optional contact details): not a cache, never committed
DATA_DIR = Path(os.environ.get("MAPTOMESH_DATA_DIR", ROOT / "data"))

for _d in (CACHE_DIR, OUTPUT_DIR, ELEV_CACHE_DIR, OSM_CACHE_DIR, DATA_DIR):
    _d.mkdir(parents=True, exist_ok=True)

LIDAR_BASE = "https://canelevation-lidar-point-clouds.s3.ca-central-1.amazonaws.com/pointclouds_nuagespoints/"
PROJECTS_ZIP_URL = LIDAR_BASE + "Index_LiDARprojects_projetslidar.zip"
TILES_ZIP_URL = LIDAR_BASE + "Index_LiDARtiles_tuileslidar.zip"

STAC_API = "https://datacube.services.geo.ca/stac/api"

# USGS 3DEP LiDAR (COPC) on Microsoft Planetary Computer: anonymous, URL signing via SAS tokens
PC_STAC_API = "https://planetarycomputer.microsoft.com/api/stac/v1"
PC_SAS_API = "https://planetarycomputer.microsoft.com/api/sas/v1/token"
USGS_COPC_COLLECTION = "3dep-lidar-copc"

USER_AGENT = "MapToMesh/0.1 (topographic 3MF generator)"

# public base URL for absolute links in the page metadata (link previews); set it to an empty
# value to use the URL each request came in on instead (local testing)
PUBLIC_URL = os.environ.get("MAPTOMESH_PUBLIC_URL", "https://maptomesh.com").rstrip("/")

# Hard safety limits
MAX_GRID_CELLS = 2400  # cells along the longest side
MAX_TILES = 2500       # LiDAR tiles per request
