"""End-to-end integration tests against a running MapToMesh server.

    python run.py --no-browser --port 8765        # in another terminal
    python tests/integration.py [--base http://127.0.0.1:8765] [case ...]

Each case submits a job, waits for it, re-validates the downloaded 3MF with
tests/validate_3mf.py and stores the thumbnail under cache/integration/.
Network access to NRCan S3, the CCMEO STAC API and OpenFreeMap is required.
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_3mf import validate  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "cache" / "integration"

# St. John's (St_Johns_2020_2) focus + regressions for other classification styles
CASES: dict[str, dict] = {
    # buildings classified, trees in class 1, harbour + open ocean beyond tile coverage
    "stj_harbour_landcover": dict(lat=47.5660, lon=-52.6950, width_m=3200, height_m=2400, grid_cells=800,
                                  frame=True, osm_roads="medium", color_mode="landcover", name="St Johns Harbour"),
    # 121 tiles, terrain-only elevation bands, circle split into 4 printable tiles
    "stj_10km_bands": dict(lat=47.5550, lon=-52.7300, width_m=10000, height_m=10000, shape="circle", grid_cells=700,
                           surface="terrain", color_mode="elevation", z_exaggeration=2.5, tiles_x=2, tiles_y=2,
                           colors={"band_mode": "quantile"}, name="St Johns 10km"),
    # easternmost point: AOI runs off the eastern edge of the LiDAR project into the Atlantic
    "stj_cape_spear_hex": dict(lat=47.5236, lon=-52.6290, width_m=2600, height_m=2600 * 0.8660254, shape="hexagon",
                               grid_cells=600, color_mode="landcover", osm_roads="major", tiles_x=2,
                               layout="objects", name="Cape Spear"),
    # fine detail full surface (trees), single colour, rounded rectangle, high grid
    "stj_signal_hill_full": dict(lat=47.5700, lon=-52.6810, width_m=1400, height_m=1000, shape="rounded",
                                 corner_radius_mm=15, grid_cells=1000, surface="full", color_mode="single",
                                 size_mm=180, name="Signal Hill"),
    # Flatrock / Torbay sea cliffs: steep-face smoothing, land-under-OSM-water check, shore walls
    "stj_flatrock_cliffs": dict(lat=47.7113, lon=-52.7099, width_m=3000, height_m=3000, grid_cells=600,
                                osm_roads="medium", color_mode="landcover", cliff_smoothing=1.0,
                                name="Flatrock cliffs"),
    # thin streams (OSM centrelines) must stay continuous: checked via the Rivers part
    "stj_flatrock_streams": dict(lat=47.7113, lon=-52.7090, width_m=2273, height_m=3000, grid_cells=600,
                                 surface="terrain", osm_roads="medium", color_mode="landcover",
                                 base_separate=True, base_mm=1.0, name="Flatrock streams"),
    # same harbour area from the HRDEM COG mosaic (DSM-DTM split by OSM footprints)
    "stj_hrdem": dict(lat=47.5660, lon=-52.6950, width_m=3200, height_m=2400, grid_cells=600, source="hrdem",
                      color_mode="landcover", osm_roads="major", name="St Johns HRDEM"),
    # base plate printed separately: model must start at z=0, base colour not in its filament list
    "stj_base_separate": dict(lat=47.5660, lon=-52.6950, width_m=3200, height_m=2400, grid_cells=500,
                              color_mode="landcover", osm_roads="major", frame=True, tiles_x=2,
                              base_separate=True, name="St Johns base separate"),
    "stj_bands_base_separate": dict(lat=47.5550, lon=-52.7300, width_m=6000, height_m=6000, shape="circle",
                                    grid_cells=500, surface="terrain", color_mode="elevation",
                                    z_exaggeration=2.5, base_separate=True, name="St Johns bands base"),
    # frame label + separate base with locating lip, 2 tiles (staggered base seams)
    "stj_label_lip_tiles": dict(lat=47.5660, lon=-52.6950, width_m=3200, height_m=2400, grid_cells=600,
                                color_mode="landcover", osm_roads="medium", frame=True, frame_width_mm=8,
                                label=True, label_text="St. John's Harbour", base_separate=True, tiles_x=2,
                                name="St Johns label"),
    # regressions: buildings unclassified (Ottawa 2020), trees unclassified (Hamilton 2021)
    "ottawa_landcover": dict(lat=45.4236, lon=-75.7009, width_m=2000, height_m=1500, shape="rounded", grid_cells=700,
                             frame=True, osm_roads="medium", color_mode="landcover", name="Ottawa"),
    "hamilton_bands": dict(lat=43.2385, lon=-79.8950, width_m=3000, height_m=2598, shape="hexagon", grid_cells=500,
                           surface="terrain", color_mode="elevation", z_exaggeration=2.0, name="Hamilton"),
}


def water_levels(path: Path) -> dict | None:
    """Distinct top heights of the 'Water' part (one per flat water body expected)."""
    import xml.etree.ElementTree as ET
    import zipfile

    import numpy as np

    ns = {"m": "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"}
    root = ET.fromstring(zipfile.ZipFile(path).read("3D/3dmodel.model"))
    tops, all_area = [], 0.0
    water_area = 0.0
    wet_v, land_v = [], []
    for obj in root.find("m:resources", ns).findall("m:object", ns):
        mesh = obj.find("m:mesh", ns)
        if mesh is None:
            continue
        v = np.array([[float(e.get(k)) for k in "xyz"] for e in mesh.find("m:vertices", ns)])
        t = np.array([[int(e.get(k)) for k in ("v1", "v2", "v3")] for e in mesh.find("m:triangles", ns)])
        p = v[t]
        n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
        up = n[:, 2] > 1e-12
        a = np.abs(n[up, 2]) / 2
        level = (p[up][:, 0, 2] == p[up][:, 1, 2]) & (p[up][:, 1, 2] == p[up][:, 2, 2])
        zl = p[up][:, 0, 2]
        name = obj.get("name") or ""
        part = name.split(" - ")[-1]
        if part in ("Water", "Rivers"):
            wet_v.append(v)
        elif part == "Ground" or part.startswith("Band "):
            land_v.append(v)
        if part in ("Base",) or "(fill)" in name:
            continue
        all_area += a.sum()
        if name.split(" - ")[-1] == "Water":
            water_area += a.sum()
            # flat water tops must sit on the 0.2 mm layer grid (default layer settings)
            on_grid = np.abs(zl / 0.2 - np.round(zl / 0.2)) < 1e-3
            tops.append((a, level, level & on_grid))
    if not tops:
        return None
    a = np.concatenate([t[0] for t in tops])
    lv = np.concatenate([t[1] for t in tops])
    og = np.concatenate([t[2] for t in tops])
    out = {"flat_share": float(a[lv].sum() / max(a.sum(), 1e-12)), "area_share": water_area / max(all_area, 1e-9),
           "on_layer_grid": float(a[og].sum() / max(a[lv].sum(), 1e-12))}
    if wet_v and land_v:
        # shore edge: land top minus water top at the outline points both parts share
        def top_by_xy(vs):
            v = np.concatenate(vs)
            key = np.round(v[:, 0] * 1e4).astype(np.int64) * 10_000_000 + np.round(v[:, 1] * 1e4).astype(np.int64)
            order = np.lexsort((-v[:, 2], key))
            first = np.r_[True, key[order][1:] != key[order][:-1]]
            return key[order][first], v[order][first, 2]
        kw, zw = top_by_xy(wet_v)
        kl, zl = top_by_xy(land_v)
        common, iw, il = np.intersect1d(kw, kl, return_indices=True)
        if len(common):
            h = zl[il] - zw[iw]
            out["shore_mm"] = [round(float(x), 2) for x in np.percentile(h, [50, 95, 100])]
    return out


def _min_z(path: Path) -> float:
    import xml.etree.ElementTree as ET
    import zipfile
    ns = {"m": "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"}
    root = ET.fromstring(zipfile.ZipFile(path).read("3D/3dmodel.model"))
    return min(float(v.get("z")) for v in root.iter(f"{{{ns['m']}}}vertex"))


def run_case(base: str, name: str, body: dict, timeout: float = 1800) -> dict:
    t0 = time.time()
    r = requests.post(f"{base}/api/jobs", json=body, timeout=30)
    r.raise_for_status()
    jid = r.json()["id"]
    while True:
        j = requests.get(f"{base}/api/jobs/{jid}", timeout=30).json()
        if j["state"] in ("done", "error", "cancelled"):
            break
        if time.time() - t0 > timeout:
            requests.post(f"{base}/api/jobs/{jid}/cancel", timeout=30)
            return {"case": name, "ok": False, "error": "timeout"}
        time.sleep(1.5)
    elapsed = time.time() - t0
    if j["state"] != "done":
        return {"case": name, "ok": False, "error": j.get("error"), "log": j.get("log", [])[-5:]}
    OUT.mkdir(parents=True, exist_ok=True)
    data = requests.get(f"{base}/api/jobs/{jid}/model.3mf", timeout=300).content
    path = OUT / f"{name}.3mf"
    path.write_bytes(data)
    (OUT / f"{name}.png").write_bytes(requests.get(f"{base}/api/jobs/{jid}/thumbnail.png", timeout=60).content)
    stl = requests.get(f"{base}/api/jobs/{jid}/stl.zip", timeout=300)
    prev = requests.get(f"{base}/api/jobs/{jid}/preview.bin", timeout=300)
    print(f"--- {name}")
    ok = validate(str(path), verbose=False)
    res = j["result"]
    if res_base := j["result"].get("base_file"):
        bpath = OUT / f"{name}_base.3mf"
        bpath.write_bytes(requests.get(f"{base}/api/jobs/{jid}/base.3mf", timeout=300).content)
        ok &= validate(str(bpath), verbose=False)
        zmin = _min_z(path)
        base_col = body["colors"]["base"] if "colors" in body and "base" in body["colors"] else "#2B2B2B"
        in_main = any(f["color"].upper() == base_col.upper() for f in j["result"]["filaments"])
        print(f"  separate base {res_base}: model min z = {zmin:.3f} mm, base colour in model filaments: {in_main}")
        if abs(zmin) > 1e-3 or in_main:
            ok = False
    water = water_levels(path)
    if water is not None and (water["flat_share"] < 0.95 or water["on_layer_grid"] < 0.99):
        print(f"  water surface not flat: {water}")
        ok = False
    # land meets water at one uniform edge (shore ramp): no tall, jagged walls at cliff feet
    wall = max(body.get("layer_height_mm", 0.2), body.get("water_recess_mm", 0.6))
    if water is not None and "shore_mm" in water and body.get("shore_slope_deg", 70) < 89.5             and water["shore_mm"][1] > wall + 0.25:
        print(f"  shore edge too tall: {water['shore_mm']} mm (expected about {wall} mm)")
        ok = False
    return {
        "case": name, "ok": ok and stl.ok and prev.ok, "seconds": round(elapsed, 1),
        "triangles": res["stats"]["triangles"], "parts": res["stats"]["parts"], "mb": round(len(data) / 1e6, 1),
        "size_mm": res["stats"]["size_mm"], "relief_mm": res["stats"]["relief_mm"],
        "lidar": f"{res['data'].get('points', 0) / 1e6:.1f}M pts/{res['data'].get('tiles', '-')} tiles"
        if res["data"].get("points") else ",".join(res["data"].get("items", [])),
        "notes": res["notes"],
        "water": water,
        "swaps": res["stats"].get("colour_changes_est"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8765")
    ap.add_argument("cases", nargs="*")
    a = ap.parse_args()
    names = a.cases or list(CASES)
    results = []
    for n in names:
        try:
            results.append(run_case(a.base, n, CASES[n]))
        except Exception as e:  # noqa: BLE001
            results.append({"case": n, "ok": False, "error": f"{type(e).__name__}: {e}"})
    print("\n=== summary")
    for r in results:
        if r.get("seconds") is not None:
            print(f"{'PASS' if r['ok'] else 'FAIL'} {r['case']:24} {r['seconds']:6.1f}s {r['triangles']:>10,} tris "
                  f"{r['parts']:>3} parts {r['mb']:6.1f} MB  {r['size_mm']} relief {r['relief_mm']} mm  {r['lidar']}"
                  f"  ~{r['swaps']} colour changes")
            if r.get("water"):
                w = r["water"]
                print(f"     water: {w['flat_share']:.1%} of water surface perfectly level, "
                      f"{w['on_layer_grid']:.1%} of it on the layer grid, water = {w['area_share']:.0%} of top area"
                      + (f", shore edge p50/p95/max {w['shore_mm']} mm" if "shore_mm" in w else ""))
            for note in r["notes"]:
                print(f"     note: {note}")
        else:
            print(f"FAIL {r['case']:24} {r.get('error')}  {r.get('log', '')}")
    sys.exit(0 if all(r["ok"] for r in results) else 1)


if __name__ == "__main__":
    main()
