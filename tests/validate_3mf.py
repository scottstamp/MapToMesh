"""Independent 3MF checker: parses the package and verifies every mesh object
is a closed, consistently oriented 2-manifold with positive volume, and that
components / build items / material references resolve.

    python tests/validate_3mf.py model.3mf [...]
"""

from __future__ import annotations

import re
import sys
import zipfile
import xml.etree.ElementTree as ET

import numpy as np

NS = {"m": "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"}


def _mesh_arrays(obj: ET.Element):
    verts = obj.find("m:mesh/m:vertices", NS)
    tris = obj.find("m:mesh/m:triangles", NS)
    v = np.array([[float(e.get("x")), float(e.get("y")), float(e.get("z"))] for e in verts], dtype=np.float64)
    t = np.array([[int(e.get("v1")), int(e.get("v2")), int(e.get("v3"))] for e in tris], dtype=np.int64)
    return v, t


def check_mesh(v: np.ndarray, t: np.ndarray) -> dict:
    n = len(v) + 1
    e = np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])
    d = e[:, 0] * n + e[:, 1]
    rev = e[:, 1] * n + e[:, 0]
    u, cnt = np.unique(d, return_counts=True)
    p = v[t]
    cross = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    vol = float(np.einsum("ij,ij->i", p[:, 0], np.cross(p[:, 1], p[:, 2])).sum() / 6)
    return {
        "vertices": len(v), "triangles": len(t),
        "bad_index": int(((t < 0) | (t >= len(v))).sum()),
        "dup_directed_edges": int((cnt > 1).sum()),
        "open_edges": int((~np.isin(rev, u)).sum()),
        "degenerate": int((np.linalg.norm(cross, axis=1) < 1e-9).sum()),
        "volume_mm3": round(vol, 2),
        "bbox": [v.min(axis=0).round(3).tolist(), v.max(axis=0).round(3).tolist()],
    }


def validate(path: str, verbose: bool = True) -> bool:
    ok = True
    with zipfile.ZipFile(path) as zf:
        names = set(zf.namelist())
        for req in ("[Content_Types].xml", "_rels/.rels", "3D/3dmodel.model"):
            if req not in names:
                print(f"  MISSING {req}"); ok = False
        rels = zf.read("_rels/.rels").decode()
        for target in re.findall(r'Target="/?([^"]+)"', rels):
            if target not in names:
                print(f"  rels target missing: {target}"); ok = False
        root = ET.fromstring(zf.read("3D/3dmodel.model"))
    if root.get("unit") != "millimeter":
        print("  unit is not millimeter"); ok = False
    res = root.find("m:resources", NS)
    mats = {}
    for bm in res.findall("m:basematerials", NS):
        mats[bm.get("id")] = [b.get("displaycolor") for b in bm.findall("m:base", NS)]
    objects = {}
    total = 0
    for obj in res.findall("m:object", NS):
        oid = obj.get("id")
        objects[oid] = obj
        if obj.find("m:mesh", NS) is not None:
            pid, pidx = obj.get("pid"), obj.get("pindex")
            if pid is not None and (pid not in mats or int(pidx) >= len(mats[pid])):
                print(f"  object {oid}: bad material ref {pid}/{pidx}"); ok = False
            v, t = _mesh_arrays(obj)
            st = check_mesh(v, t)
            total += st["triangles"]
            bad = st["bad_index"] or st["dup_directed_edges"] or st["open_edges"] or st["degenerate"] or st["volume_mm3"] <= 0
            ok &= not bad
            if verbose or bad:
                color = mats.get(pid, [None] * 99)[int(pidx or 0)] if pid else None
                print(f"  {'FAIL' if bad else 'ok  '} #{oid} {obj.get('name')!r:34} {color} tris={st['triangles']:>8,} "
                      f"open={st['open_edges']} dup={st['dup_directed_edges']} degen={st['degenerate']} vol={st['volume_mm3']:,.0f}")
        comps = obj.find("m:components", NS)
        if comps is not None:
            for c in comps.findall("m:component", NS):
                if c.get("objectid") not in objects:  # must be defined earlier
                    print(f"  object {oid}: component {c.get('objectid')} undefined/forward ref"); ok = False
    for item in root.find("m:build", NS).findall("m:item", NS):
        if item.get("objectid") not in objects:
            print(f"  build item -> missing object {item.get('objectid')}"); ok = False
    print(f"{'PASS' if ok else 'FAIL'} {path}: {len(objects)} objects, {total:,} triangles, {len(next(iter(mats.values()), []))} materials")
    return ok


if __name__ == "__main__":
    results = [validate(p) for p in sys.argv[1:]]
    sys.exit(0 if all(results) else 1)
