"""Estimate filament colour changes of a 3MF: per layer, the colours whose solids cross it.

    python tests/layer_colors.py model.3mf [--layer 0.2]
"""

import argparse
import xml.etree.ElementTree as ET
import zipfile

import numpy as np

NS = {"m": "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"}


def layer_stats(path: str, layer: float = 0.2) -> dict:
    root = ET.fromstring(zipfile.ZipFile(path).read("3D/3dmodel.model"))
    res = root.find("m:resources", NS)
    mats = {bm.get("id"): [b.get("displaycolor") for b in bm.findall("m:base", NS)]
            for bm in res.findall("m:basematerials", NS)}
    ranges: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    zmax = 0.0
    for obj in res.findall("m:object", NS):
        mesh = obj.find("m:mesh", NS)
        if mesh is None:
            continue
        color = mats[obj.get("pid")][int(obj.get("pindex"))]
        v = np.array([float(e.get("z")) for e in mesh.find("m:vertices", NS)])
        t = np.array([[int(e.get(k)) for k in ("v1", "v2", "v3")] for e in mesh.find("m:triangles", NS)])
        z = v[t]
        ranges.setdefault(color, []).append((z.min(axis=1), z.max(axis=1)))
        zmax = max(zmax, float(v.max()))
    nl = int(np.ceil(zmax / layer)) + 1
    counts = np.zeros(nl, np.int64)
    for color, rr in ranges.items():
        diff = np.zeros(nl + 2, np.int64)
        for zmin, zmx in rr:
            lo = np.floor(zmin / layer - 0.5).astype(np.int64) + 1
            hi = np.ceil(zmx / layer - 0.5).astype(np.int64) - 1
            ok = hi >= lo
            np.add.at(diff, np.clip(lo[ok], 0, nl), 1)
            np.add.at(diff, np.clip(hi[ok] + 1, 0, nl + 1), -1)
        counts += (np.cumsum(diff)[:nl] > 0)
    return {"layers": nl, "colour_changes": int(np.maximum(counts - 1, 0).sum()),
            "max_colours_per_layer": int(counts.max()),
            "layers_with_2plus": int((counts >= 2).sum()), "per_layer": counts.tolist()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--layer", type=float, default=0.2)
    a = ap.parse_args()
    st = layer_stats(a.path, a.layer)
    print({k: v for k, v in st.items() if k != "per_layer"})
    print("colours per layer:", "".join(str(min(c, 9)) for c in st["per_layer"]))
