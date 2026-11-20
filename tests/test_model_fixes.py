"""Offline checks for the surface / mesh pipeline helpers (model.py, partition.py).

    python tests/test_model_fixes.py
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from maptomesh import model  # noqa: E402
from maptomesh.partition import DROPPED, Partition, shape_field  # noqa: E402


def _field(rng, shape, sigma=3.0):
    return ndimage.gaussian_filter(rng.random(shape), sigma)


def test_label_reduce():
    """Per-label max / min match ndimage's (which sorts the whole grid per call)."""
    rng = np.random.default_rng(3)
    m = _field(rng, (90, 70), 2.0) > 0.52
    lab, n = ndimage.label(m, structure=np.ones((3, 3)))
    v = rng.random(m.shape) * 100
    idx = np.arange(1, n + 1)
    assert np.array_equal(model._label_reduce(np.maximum, v, lab, n), np.asarray(ndimage.maximum(v, lab, idx)))
    assert np.array_equal(model._label_reduce(np.minimum, v, lab, n), np.asarray(ndimage.minimum(v, lab, idx)))
    iv = rng.integers(-50, 50, m.shape)
    assert np.array_equal(model._label_reduce(np.maximum, iv, lab, n), np.asarray(ndimage.maximum(iv, lab, idx)))


def test_partition_copy_and_cells():
    """Copies share the source triangles but nothing they write; the incrementally tracked cell
    labels equal classifying the cell centres with every step's field afterwards."""
    rng = np.random.default_rng(5)
    R, C = 40, 50
    p = Partition(R, C, _field(rng, (R, C)) > 0.5)
    fields = [shape_field("circle", R, C, C - 6, R - 6).ravel()]
    p.step(fields[0], 0.0, below=DROPPED)
    q = copy.deepcopy(p)
    r = p.copy()
    assert q.tris is p.tris and r.tris is p.tris
    for lab in (5, 4, 3):
        f = _field(rng, (R + 1, C + 1), 1.5).ravel()
        fields.append(f)
        p.step(f, float(np.median(f)), above=lab)
    f = _field(rng, (R + 1, C + 1), 1.5).ravel()
    fields.append(f)
    p.step(f, float(np.median(f)), above=7, only=4)
    p.finish(1)
    assert (q.tri_label != p.tri_label).any() and len(q.steps) == 1 and len(q.p_cnt) < len(p.p_cnt)
    q.finish(2)
    assert not (r.tri_label == 2).any()

    def centre(a):
        a = a.reshape(R + 1, C + 1)
        return (a[:-1, :-1] + a[:-1, 1:] + a[1:, :-1] + a[1:, 1:]) / 4

    assert np.array_equal(p.cell_labels(1), p.classify([centre(f) for f in fields], 1))
    assert {5, 4, 3, 7, 1, DROPPED} >= set(np.unique(p.cell_labels(1)).tolist())


def test_partition_step_sides():
    """The grid-based side test (most triangles active) and the gather (few active) agree."""
    rng = np.random.default_rng(9)
    R, C = 30, 30
    diag = _field(rng, (R, C)) > 0.5
    f = _field(rng, (R + 1, C + 1), 2.0).ravel()
    f[::17] = np.nan
    th = float(np.nanmedian(f))
    p = Partition(R, C, diag)
    le, ge = p._tri_sides(f, th)
    fv = f[p.tris]
    with np.errstate(invalid="ignore"):
        assert np.array_equal(le, fv.max(axis=1) <= th)
        assert np.array_equal(ge, fv.min(axis=1) >= th)


def test_shore_cap_identical():
    """Querying only the vertices the cap can bind gives the same capped surface as capping with
    the distance of every vertex within reach."""
    rng = np.random.default_rng(11)
    R, C = 120, 140
    wet = _field(rng, (R + 1, C + 1), 6.0) > 0.5
    ind = ndimage.gaussian_filter(wet.astype(float), 1.5)
    diag = _field(rng, (R, C)) > 0.5
    top = 5 + 20 * _field(rng, (R + 1, C + 1), 4.0) + 30 * (rng.random((R + 1, C + 1)) > 0.995)
    wtop = np.full(top.shape, 4.0)
    wall, k = 0.4, 0.7
    got = model._shore_cap(diag, ind, wtop, wall, k, top)
    reach = max(2.0, (float(top.max()) - 4.0 - wall) / k + 2.0)
    d = model._iso_distance(diag, ind, 0.5, reach)
    want = np.minimum(top, np.where(np.isfinite(d), wtop + wall + k * np.nan_to_num(d), np.inf))
    assert np.array_equal(got, want)
    assert (got < top).any()


if __name__ == "__main__":
    for fn in (test_label_reduce, test_partition_copy_and_cells, test_partition_step_sides,
               test_shore_cap_identical):
        fn()
        print(f"{fn.__name__}: ok")
    print("all passed")
