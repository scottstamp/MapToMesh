"""Offline checks for maptomesh.raster and the OSM water fill.

    python tests/test_raster.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from maptomesh.osm import _fill_box  # noqa: E402
from maptomesh.raster import fill_polygon, masked_median, rasterize_lines  # noqa: E402


def _old_median(a, mask, where, size, q=50.0):
    out = np.array(a, dtype=np.float64)
    r = size // 2
    rows, cols = np.nonzero(where)
    vals = np.where(mask, out, np.nan)
    win = np.lib.stride_tricks.sliding_window_view(np.pad(vals, r, constant_values=np.nan), (size, size))
    w = win[rows, cols].reshape(len(rows), -1)
    ok = np.isfinite(w).any(axis=1)
    out[rows[ok], cols[ok]] = np.nanmedian(w[ok], axis=1) if q == 50.0 else np.nanpercentile(w[ok], q, axis=1)
    return out


def test_masked_median_matches_nan_functions():
    rng = np.random.default_rng(3)
    for size in (3, 5, 9):
        for q in (50.0, 20.0, 0.0, 100.0, 87.5):
            a = rng.normal(100, 20, (60, 70))
            for dens in (0.02, 0.3, 0.8):
                mask = rng.random(a.shape) < dens     # sparse: lone finite values, empty windows
                where = rng.random(a.shape) < 0.5
                got = masked_median(a, mask, where, size, q)
                want = _old_median(a, mask, where, size, q)
                assert np.abs(got - want).max() < 1e-9, (size, q, dens)
    # empty windows stay unchanged
    a = rng.normal(size=(20, 20))
    got = masked_median(a, np.zeros(a.shape, bool), np.ones(a.shape, bool), 3)
    assert np.array_equal(got, a)


def test_water_fill_box_matches_full():
    rng = np.random.default_rng(5)
    shape = (70, 90)
    full = np.zeros(shape, bool)
    boxed = np.zeros(shape, bool)
    for _ in range(300):
        c = rng.uniform(-30, 120, 2)
        n = rng.integers(3, 10)
        ang = np.sort(rng.uniform(0, 2 * np.pi, n))
        rad = rng.uniform(1, 25, n)
        outer = np.c_[c[0] + rad * np.cos(ang), c[1] + rad * np.sin(ang)]
        parts = [outer]
        if rng.random() < 0.5:                      # a hole
            parts.append(c + (outer - c) * 0.4)
        full |= fill_polygon(parts, shape)
        box = _fill_box(parts, shape)
        if box is not None:
            boxed[box[0]] |= box[1]
    assert full.any() and np.array_equal(full, boxed)


def test_rasterize_lines_crop_identical():
    rng = np.random.default_rng(7)
    shape = (120, 150)
    lines = []
    for _ in range(40):
        pts = rng.uniform([-10, -10], [160, 130], (rng.integers(2, 6), 2))
        lines.append((pts, float(rng.choice([0.3, 1.0, 1.5, 2.2, 3.0, 5.0]))))
    got = rasterize_lines(lines, shape)
    # reference: full-grid distance transform per width group
    want = np.zeros(shape, bool)
    groups = {}
    for pts, hw in lines:
        groups.setdefault(max(round(hw * 4) / 4, 1.0), []).append(pts)
    for hw, plist in groups.items():
        centre = np.zeros(shape, bool)
        for p in plist:
            c = rasterize_centre(p, shape)
            centre |= c
        if centre.any():
            want |= ndimage.distance_transform_edt(~centre) <= hw
    # the centre pixels are recovered by a tiny line; groups differ only by crop, so compare per group too
    assert np.array_equal(got, want)


def rasterize_centre(p, shape):
    """Centre-line pixels of one polyline, as rasterize_lines marks them (same sampling)."""
    H, W = shape
    center = np.zeros(shape, bool)
    p = np.asarray(p, dtype=np.float64)
    a, b = p[:-1], p[1:]
    seg = np.hypot(*(b - a).T)
    n = np.maximum(np.ceil(seg / 0.5).astype(np.intp), 1) + 1
    e = np.repeat(np.arange(len(a)), n)
    t = (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)) / np.repeat(n - 1, n)
    q = a[e] + (b[e] - a[e]) * t[:, None]
    c = np.rint(q[:, 0]).astype(np.int64)
    r = np.rint(q[:, 1]).astype(np.int64)
    ok = (c >= 0) & (c < W) & (r >= 0) & (r < H)
    center[r[ok], c[ok]] = True
    return center


def test_thin_lines_survive_colour_cut():
    shape = (80, 80)
    for deg in (0, 10, 30, 45):
        d = np.deg2rad(deg)
        x0, y0 = 10.0, 20.0
        x1, y1 = x0 + 55 * np.cos(d), y0 + 55 * np.sin(d)
        for hw in (0.3, 0.8):
            m = rasterize_lines([(np.array([[x0, y0], [x1, y1]]), hw)], shape)
            f = ndimage.gaussian_filter(m.astype(float), 0.7)
            ts = np.linspace(0.1, 0.9, 60)
            xs, ys = x0 + ts * (x1 - x0), y0 + ts * (y1 - y0)
            v = f[np.rint(ys).astype(int), np.rint(xs).astype(int)]
            assert v.min() > 0.6, (deg, hw, v.min())


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
