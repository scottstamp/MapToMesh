"""Canada and US elevation models (the point clouds themselves are in ``pointcloud`` / ``usgs_ept``).

* NRCan HRDEM 1 m / 2 m mosaics (DTM + DSM COGs via the CCMEO STAC API).
* USGS 3DEP 1/3 arc-second (~10 m) seamless DTM, public COGs in USGS's own ``prd-tnm`` bucket.
"""

from __future__ import annotations

import math

import numpy as np

from ..dem import DemArrays, DemProvider, register, want_dsm
from ..dem_readers import merge_fill, read_cog
from ..usgs3dep import US_REGIONS


class Hrdem(DemProvider):
    key = "nrcan_hrdem"
    label = "NRCan HRDEM 1-2 m (Canada)"
    res_m = 1.0
    has_dsm = True
    attribution = "Contains information licensed under the Open Government Licence - Canada (NRCan HRDEM)."
    licence = "Open Government Licence - Canada (https://open.canada.ca/en/open-government-licence-canada)"
    regions = ((-141.1, 41.6, -52.5, 83.2),)
    max_area_km2 = 40_000.0
    sample = (47.5615, -52.7126, 3000)

    def fetch(self, aoi, grid, progress, cancel):
        from concurrent.futures import ThreadPoolExecutor
        from ..hrdem import COLLECTIONS, stac_items
        bbox = aoi.lonlat_bbox(margin=grid.step * 2)
        dtm = np.full(grid.vshape, np.nan)
        dsm = np.full(grid.vshape, np.nan)
        used = []
        with ThreadPoolExecutor(max_workers=2) as ex:   # an item's DTM and DSM side by side
            for coll in COLLECTIONS:
                items = stac_items(bbox, coll)
                for k, it in enumerate(items):
                    if cancel.is_set():
                        raise RuntimeError("Cancelled")
                    progress(k / max(len(items), 1), f"HRDEM {it['id']}")
                    futs = [(arr, ex.submit(read_cog, it["assets"][name]["href"], aoi, grid))
                            for name, arr in (("dtm", dtm), ("dsm", dsm))[:2 if want_dsm() else 1]
                            if it["assets"].get(name, {}).get("href")]
                    for arr, fut in futs:
                        merge_fill(arr, fut.result())
                    used.append(it["id"])
                if np.isfinite(dtm).mean() > 0.98:
                    break
        return DemArrays(dtm, dsm if np.isfinite(dsm).any() else None, ", ".join(used))


class Usgs3depSeamless(DemProvider):
    key = "usgs_3dep_13"
    label = "USGS 3DEP 10 m (US)"
    res_m = 10.0
    attribution = "USGS 3D Elevation Program (public domain)."
    licence = "Public domain (https://www.usgs.gov/information-policies-and-instructions/copyrights-and-credits)"
    regions = US_REGIONS
    max_area_km2 = 50_000.0
    sample = (36.0570, -112.1180, 8000)   # Grand Canyon

    def fetch(self, aoi, grid, progress, cancel):
        w, s, e, n = aoi.lonlat_bbox(margin=grid.step * 3)
        out = np.full(grid.vshape, np.nan)
        # 1x1 degree tiles named after the NW corner: n38w122 covers 37-38 N, 122-121 W
        cells = [(la, lo) for la in range(math.floor(s) + 1, math.floor(n) + 2)
                 for lo in range(math.floor(w), math.floor(e) + 1)]
        used = []
        for k, (la, lo) in enumerate(cells):
            if cancel.is_set():
                raise RuntimeError("Cancelled")
            name = f"{'n' if la >= 0 else 's'}{abs(la):02d}{'w' if lo < 0 else 'e'}{abs(lo):03d}"
            url = f"https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/current/{name}/USGS_13_{name}.tif"
            progress(k / len(cells), f"USGS 3DEP tile {name}")
            try:
                merge_fill(out, read_cog(url, aoi, grid))
                used.append(name)
            except Exception:  # noqa: BLE001 - no tile there (ocean, outside the US)
                continue
        return DemArrays(out, None, ", ".join(used))


register(Hrdem())
register(Usgs3depSeamless())
