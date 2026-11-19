"""Smoke-test elevation providers on their sample areas (needs internet).

    python tests/probe_sources.py                 # every DEM and point-cloud provider
    python tests/probe_sources.py ign_rgealti ...  # selected provider keys
    python tests/probe_sources.py --list

DEM providers: fetch the sample area at 200 cells and check the share of valid heights and that the
values are plausible. Point-cloud providers: find tiles for the sample area, open the first one
remotely and read its root node.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from maptomesh import dem, providers  # noqa: E402,F401  (registers providers)
from maptomesh.geo import AOI, Grid  # noqa: E402
from maptomesh.pointcloud import POINTCLOUD_PROVIDERS, _file_crs  # noqa: E402


def probe_dem(p, cells: int) -> tuple[bool, str]:
    lat, lon, size = p.sample
    aoi = AOI(lat, lon, size, size)
    grid = Grid.for_aoi(aoi, cells)
    t0 = time.time()
    arr = p.fetch(aoi, grid, lambda f, m: None, threading.Event())
    dt = time.time() - t0
    d = arr.dtm
    if d.shape != grid.vshape:
        return False, f"wrong shape {d.shape} != {grid.vshape}"
    fin = np.isfinite(d)
    share = fin.mean()
    msg = f"{share:6.1%} valid"
    if fin.any():
        msg += f", {np.nanmin(d):8.1f} .. {np.nanmax(d):8.1f} m"
    if arr.dsm is not None:
        ds = np.isfinite(arr.dsm)
        msg += f", DSM {ds.mean():.0%}"
        if ds.any() and fin.any():
            both = ds & fin
            if both.any():
                msg += f" (DSM-DTM p50 {np.median(arr.dsm[both] - d[both]):+.2f} m, p99 {np.percentile(arr.dsm[both] - d[both], 99):.1f} m)"
    msg += f", {dt:.1f}s"
    ok = share > 0.8 and fin.any() and -500 < np.nanmin(d) and np.nanmax(d) < 9000 and np.nanstd(d) > 0.05
    if arr.dsm is not None and np.isfinite(arr.dsm).any() and fin.any():
        both = np.isfinite(arr.dsm) & fin
        ok &= bool(np.median(arr.dsm[both] - d[both]) > -1.0)   # DSM must not sit below the DTM
    return ok, msg + (f" [{arr.detail[:80]}]" if arr.detail else "")


def probe_pc(p) -> tuple[bool, str]:
    from laspy import CopcReader
    lat, lon, size = p.sample
    aoi = AOI(lat, lon, size, size)
    t0 = time.time()
    tiles = p.tiles(aoi.lonlat_bbox(), 100)
    if not tiles:
        return False, "no tiles"
    t = tiles[0]
    with CopcReader.open(t.url) as r:
        crs, zf = _file_crs(r, t.crs)
        pts = r.query(resolution=50.0)
        cls = sorted(set(np.unique(np.asarray(pts.classification)).tolist()))
    dt = time.time() - t0
    ok = crs is not None and len(pts) > 0
    return ok, (f"{len(tiles)} tiles ({t.project}, {t.year}), CRS {crs.name if crs else None}, z x{zf:g}, "
                f"root query {len(pts)} pts classes {cls}, {dt:.1f}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("keys", nargs="*")
    ap.add_argument("--cells", type=int, default=200)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if providers.IMPORT_ERRORS:
        print("import errors:", providers.IMPORT_ERRORS)
    allp = [("dem", p) for p in dem.ordered()] + [("pc", p) for p in POINTCLOUD_PROVIDERS]
    if a.list:
        for kind, p in allp:
            print(f"{kind:3} {p.key:24} {getattr(p, 'res_m', '-')!s:>5} m  {p.label}")
        return
    sel = [(k, p) for k, p in allp if not a.keys or p.key in a.keys]
    results = []
    for kind, p in sel:
        if not p.sample:
            print(f"SKIP {p.key}: no sample area")
            continue
        try:
            ok, msg = probe_dem(p, a.cells) if kind == "dem" else probe_pc(p)
        except Exception as e:  # noqa: BLE001
            ok, msg = False, f"{type(e).__name__}: {str(e)[:300]}"
        results.append(ok)
        print(f"{'PASS' if ok else 'FAIL'} {kind:3} {p.key:24} {msg}", flush=True)
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
