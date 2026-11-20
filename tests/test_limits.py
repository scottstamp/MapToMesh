"""Offline checks for the job limits (maptomesh/limits.py), as on a 4 GB server.

    python tests/test_limits.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["MAPTOMESH_MEMORY_MB"] = "3277"      # 80% of 4 GB, before the module reads it
os.environ.pop("MAPTOMESH_PIPELINE", None)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from maptomesh import limits  # noqa: E402
from maptomesh.options import Options  # noqa: E402


def opts(**kw) -> Options:
    base = dict(lat=47.566, lon=-52.695, width_m=3200, height_m=2400, grid_cells=600)
    base.update(kw)
    return Options(**base)


def test_budget() -> None:
    assert limits.PIPELINE == 1, "a 4 GB server runs one job at a time"
    assert limits.PART_WORKERS == 1, "and meshes with one thread (lower peak)"
    assert 2500 < limits.JOB_MEMORY_MB < 3000, limits.JOB_MEMORY_MB


def test_estimates() -> None:
    # measured peaks (MB) with one part worker on St. John's jobs: the model must be within -5% /
    # +15% (it may overestimate a little, never underestimate much)
    H = dict(width_m=3200, height_m=2400)
    B = dict(width_m=10000, height_m=10000, shape="circle", surface="terrain")
    for kw, measured in ((dict(H, grid_cells=1400, color_mode="landcover"), 1741),
                         (dict(H, grid_cells=2000, color_mode="landcover"), 3247),      # LiDAR download peak 3192
                         (dict(B, grid_cells=1400, color_mode="elevation"), 1534),
                         (dict(H, grid_cells=2400, color_mode="single", surface="terrain"), 3103)):
        est = limits.estimate(opts(**kw))["memory_mb"]
        assert 0.95 * measured <= est <= 1.15 * measured, (kw, est, measured)


def test_caps() -> None:
    cases = [dict(color_mode="landcover"), dict(color_mode="elevation", surface="terrain"),
             dict(color_mode="single", surface="terrain"), dict(color_mode="single", source="dem")]
    caps = []
    for kw in cases:
        cap = limits.max_cells(opts(**kw))
        caps.append(cap)
        assert limits.estimate(opts(grid_cells=cap, **kw))["memory_mb"] <= limits.JOB_MEMORY_MB
        limits.check(opts(grid_cells=cap, **kw))                      # at the cap: accepted
        if cap < 2400:
            assert limits.estimate(opts(grid_cells=min(2400, cap + 200), **kw))["memory_mb"] >                 limits.JOB_MEMORY_MB * 0.9
            try:
                limits.check(opts(grid_cells=min(2400, cap + 100), **kw))
            except limits.LimitError as e:
                assert e.kind == "too_large" and e.suggest == {"grid_cells": max(150, cap // 50 * 50)} and e.advice
            else:
                raise AssertionError(f"{kw}: {cap + 100} cells accepted")
    assert caps[0] < caps[1], caps                 # land cover (+ classified point clouds) is the heaviest
    # longer, thinner areas allow more cells on the long side (same vertex count)
    assert limits.max_cells(opts(width_m=4000, height_m=1000)) > limits.max_cells(opts(width_m=2000, height_m=2000))


def test_advice() -> None:
    o = opts(grid_cells=1400, color_mode="landcover")
    for kind in ("timeout", "memory"):
        tips, sug = limits.advice_for(kind, o)
        assert tips and sug["grid_cells"] < 1400 and sug["grid_cells"] % 50 == 0
    tips, sug = limits.advice_for("busy", o)
    assert tips and not sug


if __name__ == "__main__":
    for fn in (test_budget, test_estimates, test_caps, test_advice):
        fn()
        print(f"{fn.__name__}: ok")
    print("all passed")
