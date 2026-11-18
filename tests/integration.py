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
    # same harbour area from the HRDEM COG mosaic (DSM-DTM split by OSM footprints)
    "stj_hrdem": dict(lat=47.5660, lon=-52.6950, width_m=3200, height_m=2400, grid_cells=600, source="hrdem",
                      color_mode="landcover", osm_roads="major", name="St Johns HRDEM"),
    # regressions: buildings unclassified (Ottawa 2020), trees unclassified (Hamilton 2021)
    "ottawa_landcover": dict(lat=45.4236, lon=-75.7009, width_m=2000, height_m=1500, shape="rounded", grid_cells=700,
                             frame=True, osm_roads="medium", color_mode="landcover", name="Ottawa"),
    "hamilton_bands": dict(lat=43.2385, lon=-79.8950, width_m=3000, height_m=2598, shape="hexagon", grid_cells=500,
                           surface="terrain", color_mode="elevation", z_exaggeration=2.0, name="Hamilton"),
}


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
    return {
        "case": name, "ok": ok and stl.ok and prev.ok, "seconds": round(elapsed, 1),
        "triangles": res["stats"]["triangles"], "parts": res["stats"]["parts"], "mb": round(len(data) / 1e6, 1),
        "size_mm": res["stats"]["size_mm"], "relief_mm": res["stats"]["relief_mm"],
        "lidar": f"{res['data'].get('points', 0) / 1e6:.1f}M pts/{res['data'].get('tiles', '-')} tiles"
        if res["data"].get("points") else ",".join(res["data"].get("items", [])),
        "notes": res["notes"],
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
                  f"{r['parts']:>3} parts {r['mb']:6.1f} MB  {r['size_mm']} relief {r['relief_mm']} mm  {r['lidar']}")
            for note in r["notes"]:
                print(f"     note: {note}")
        else:
            print(f"FAIL {r['case']:24} {r.get('error')}  {r.get('log', '')}")
    sys.exit(0 if all(r["ok"] for r in results) else 1)


if __name__ == "__main__":
    main()
