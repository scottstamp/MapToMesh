"""Partition / part-builder checks (offline): python tests/test_partition.py"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402

from maptomesh.mesh import check_manifold  # noqa: E402
from maptomesh.partition import DROPPED, FieldSpec, Partition, shape_field  # noqa: E402


def check(mesh, name):
    st = check_manifold(mesh)
    ok = st["open_edges"] == 0 and st["dup_directed_edges"] == 0 and st["degenerate"] == 0 and st["volume"] > 0
    if not ok:
        print("FAIL", name, st)
    return ok, st


def run(seed: int, R: int, C: int, shape: str, frame: bool) -> bool:
    rng = np.random.default_rng(seed)
    z = 3 + 1.5 + 8 * ndimage.gaussian_filter(rng.random((R + 1, C + 1)), 4)
    z = (z - z.min()) / (z.max() - z.min()) * 10 + 4.5
    flat_lake = ndimage.gaussian_filter(rng.random((R + 1, C + 1)), 3) > 0.62
    z[flat_lake] = 4.5  # exactly flat water (strip-decimated, touches band threshold)
    a00, a01, a10, a11 = z[:-1, :-1], z[:-1, 1:], z[1:, :-1], z[1:, 1:]
    ok_all = True
    pad = 4 if frame else 0
    fm = shape_field(shape, R, C, C - 2 * pad, R - 2 * pad, corner_r=5)
    for mode in ("landcover", "bands"):
        p = Partition(R, C, np.abs(a00 - a11) <= np.abs(a01 - a10))
        if frame:
            p.step(shape_field(shape, R, C, C - 2 * pad, R - 2 * pad, corner_r=5, grow=3), 0, below=DROPPED)
            p.step(fm, 0, below=0)
        elif shape != "rectangle":
            p.step(fm, 0, below=DROPPED)
        import copy
        shape_only = copy.deepcopy(p)  # outline cuts only: used for the base plate
        shape_only.finish(1)
        defs = []
        base_mesh = shape_only.build_part({1}, FieldSpec(3.0), FieldSpec(0.0), (0, R, 0, C), 0.3, 0, 0)
        ok_all &= check(base_mesh, f"{seed} {shape} base")[0]
        if mode == "landcover":
            for lab in (5, 4, 3, 2):
                m = ndimage.gaussian_filter(rng.random((R + 1, C + 1)), 1.5) > 0.56
                p.step(ndimage.gaussian_filter(m.astype(float), 0.7), 0.5, above=lab)
            p.finish(1)
            roof = (z + 3).ravel()
            water = np.full(z.size, 4.2)
            defs += [({1}, FieldSpec(z.ravel()), FieldSpec(3.0)),
                     ({5}, FieldSpec(roof), FieldSpec(3.0)),
                     ({3}, FieldSpec(water), FieldSpec(water, -1.0, lo=3.05)),
                     ({3}, FieldSpec(water, -1.0, lo=3.05), FieldSpec(3.0)),
                     ({2}, FieldSpec(z.ravel()), FieldSpec(z.ravel(), -1.0, lo=3.05))]
        else:
            th = [4.5, 7.0, 9.5, 12.0]  # 4.5 == flat lake level (knife edges)
            for k in range(len(th), 0, -1):
                p.step(z.ravel(), th[k - 1], above=10 + k)
            p.finish(10)
            lo = [3.0] + th
            for k in range(len(lo)):
                hi = lo[k + 1] if k + 1 < len(lo) else np.inf
                defs.append(({10 + j for j in range(k, len(lo))}, FieldSpec(z.ravel(), hi=hi), FieldSpec(lo[k])))
        if frame:
            defs.append(({0}, FieldSpec(5.0), FieldSpec(0.0)))
        vols = []
        for labels, top, bot in defs:
            for win in ((0, R, 0, C), (0, R // 2, 0, C // 3)):
                mesh = p.build_part(labels, top, bot, win, 0.3, 0.0, 0.0)
                if mesh is None:
                    continue
                ok, st = check(mesh, f"{seed} {shape} {mode} {sorted(labels)} {win}")
                ok_all &= ok
                if win == (0, R, 0, C):
                    vols.append(st["volume"])
        if mode == "bands" and not frame and shape == "rectangle":
            # slabs + base must add up to the volume under the terrain surface
            ref = Partition(R, C, np.abs(a00 - a11) <= np.abs(a01 - a10))
            ref.finish(1)
            whole = check_manifold(ref.build_part({1}, FieldSpec(z.ravel()), FieldSpec(3.0), (0, R, 0, C), 0.3, 0, 0))
            rel = abs(sum(vols) - whole["volume"]) / whole["volume"]
            if rel > 1e-4:  # thin outline walls are lifted to 10 um
                print("FAIL volume mismatch", sum(vols), whole["volume"])
                ok_all = False
    return ok_all


if __name__ == "__main__":
    t = time.time()
    ok = True
    for i, (shape, frame) in enumerate([("rectangle", False), ("circle", False), ("hexagon", True),
                                        ("rounded", True), ("rectangle", True), ("circle", True)]):
        ok &= run(i, 60 + 7 * i, 80 + 5 * i, shape, frame)
    print("partition tests", "PASS" if ok else "FAIL", f"{time.time() - t:.1f}s")
    sys.exit(0 if ok else 1)
