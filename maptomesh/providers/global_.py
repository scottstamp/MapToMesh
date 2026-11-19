"""Global fallback: Copernicus DEM GLO-30 (GLO-90 where a 30 m tile is not public).

COGs on AWS Open Data, one 1x1 degree tile per file, named after the tile's south-west corner.
Surface model (EGM2008 heights): trees and buildings are part of the heights.
"""

from __future__ import annotations

import math

import numpy as np

from ..dem import DemArrays, DemProvider, register
from ..dem_readers import merge_fill, read_cog

ATTRIBUTION = ("produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH "
               "2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved")


def _tile_name(lat0: int, lon0: int, res_code: str) -> str:
    ns = f"N{lat0:02d}" if lat0 >= 0 else f"S{-lat0:02d}"
    ew = f"E{lon0:03d}" if lon0 >= 0 else f"W{-lon0:03d}"
    return f"Copernicus_DSM_COG_{res_code}_{ns}_00_{ew}_00_DEM"


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
        w, s, e, n = aoi.lonlat_bbox(margin=grid.step * 3)
        out = np.full(grid.vshape, np.nan)
        cells = [(la, lo) for la in range(math.floor(s), math.floor(n) + 1)
                 for lo in range(math.floor(w), math.floor(e) + 1)]
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


register(CopernicusGlo30())
