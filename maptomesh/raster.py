"""Grid utilities: hole filling, rasterisation and label cleanup."""

from __future__ import annotations

import numpy as np
from scipy import ndimage


# ---------------------------------------------------------------- hole filling

def fill_nan(a: np.ndarray) -> np.ndarray:
    """Fill NaNs with a smooth push-pull (pyramid) interpolation."""
    a = np.asarray(a, dtype=np.float64)
    valid = np.isfinite(a)
    if valid.all():
        return a.copy()
    if not valid.any():
        return np.zeros_like(a)
    levels = []
    val = np.where(valid, a, 0.0)
    w = valid.astype(np.float64)
    while True:
        levels.append((val, w))
        h, wd = val.shape
        if h <= 1 and wd <= 1:
            break
        ph, pw = h + (h & 1), wd + (wd & 1)
        vv = np.zeros((ph, pw)); ww = np.zeros((ph, pw))
        vv[:h, :wd] = val * w
        ww[:h, :wd] = w
        vs = vv.reshape(ph // 2, 2, pw // 2, 2).sum(axis=(1, 3))
        ws = ww.reshape(ph // 2, 2, pw // 2, 2).sum(axis=(1, 3))
        val = np.divide(vs, ws, out=np.zeros_like(vs), where=ws > 0)
        w = np.minimum(ws, 1.0)
    # pull: upsample coarse estimates into gaps
    val, w = levels[-1]
    filled = val
    for v, wt in reversed(levels[:-1]):
        h, wd = v.shape
        up = np.repeat(np.repeat(filled, 2, axis=0), 2, axis=1)[:h, :wd]
        up = ndimage.uniform_filter(up, size=3, mode="nearest")
        filled = wt * v + (1 - wt) * up
    out = a.copy()
    out[~valid] = filled[~valid]
    return out


def smooth(a: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return a
    return ndimage.gaussian_filter(a, sigma, mode="nearest")


def despike(a: np.ndarray, mask: np.ndarray, thresh: float, size: int = 3) -> np.ndarray:
    """Clamp isolated spikes (birds, wires) to local median + thresh."""
    med = ndimage.median_filter(a, size=size, mode="nearest")
    out = a.copy()
    spikes = mask & (a > med + thresh)
    out[spikes] = med[spikes]
    return out


def masked_median(a: np.ndarray, mask: np.ndarray, where: np.ndarray, size: int, q: float = 50.0) -> np.ndarray:
    """At the ``where`` cells, the median (or the ``q``-th percentile) of ``a`` over the ``mask``
    cells in a size x size window (others unchanged). Only the bounding box of ``where`` is
    processed."""
    out = np.array(a, dtype=np.float64)
    if not where.any():
        return out
    r = size // 2
    rows, cols = np.nonzero(where)
    r0, r1 = max(rows.min() - r, 0), min(rows.max() + r + 1, a.shape[0])
    c0, c1 = max(cols.min() - r, 0), min(cols.max() + r + 1, a.shape[1])
    vals = np.where(mask[r0:r1, c0:c1], out[r0:r1, c0:c1], np.nan)
    win = np.lib.stride_tricks.sliding_window_view(np.pad(vals, r, constant_values=np.nan), (size, size))
    for k in range(0, len(rows), 20000):    # bounded temporaries (WebAssembly memory)
        i, j = rows[k:k + 20000], cols[k:k + 20000]
        w = win[i - r0, j - c0].reshape(len(i), -1)
        ok = np.isfinite(w).any(axis=1)
        out[i[ok], j[ok]] = np.nanmedian(w[ok], axis=1) if q == 50.0 else np.nanpercentile(w[ok], q, axis=1)
    return out


# ---------------------------------------------------------------- rasterisation

def fill_polygon(rings: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    """Even-odd fill of one (multi)polygon given as rings in *vertex* grid
    coordinates (col, row). Returns a bool mask over vertex positions."""
    H, W = shape
    rr_all, xx_all = [], []
    for ring in rings:
        if len(ring) < 3:
            continue
        p = np.asarray(ring, dtype=np.float64)
        x1, y1 = p[:, 0], p[:, 1]
        x2, y2 = np.roll(x1, -1), np.roll(y1, -1)
        lo, hi = np.minimum(y1, y2), np.maximum(y1, y2)
        r0 = np.maximum(np.ceil(lo), 0).astype(np.int64)
        r1 = np.minimum(np.ceil(hi) - 1, H - 1).astype(np.int64)  # rows with lo <= r < hi
        n = np.maximum(r1 - r0 + 1, 0).astype(np.intp)   # repeat counts: 32-bit in WebAssembly
        if n.sum() == 0:
            continue
        e = np.repeat(np.arange(len(p)), n)
        rows = r0[e] + (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n))
        t = (rows - y1[e]) / (y2[e] - y1[e])
        xs = x1[e] + t * (x2[e] - x1[e])
        rr_all.append(rows)
        xx_all.append(xs)
    mask = np.zeros(shape, dtype=bool)
    if not rr_all:
        return mask
    rows = np.concatenate(rr_all)
    xs = np.concatenate(xx_all)
    order = np.lexsort((xs, rows))
    rows, xs = rows[order], xs[order]
    # pair consecutive crossings within each row
    starts = np.ones(len(rows), dtype=bool)
    starts[1:] = rows[1:] != rows[:-1]
    rank = np.arange(len(rows)) - np.maximum.accumulate(np.where(starts, np.arange(len(rows)), 0))
    a = rank % 2 == 0
    ia = np.nonzero(a)[0]
    ia = ia[ia + 1 < len(rows)]
    ia = ia[rows[ia + 1] == rows[ia]]
    r = rows[ia]
    c0 = np.clip(np.ceil(xs[ia]), 0, W).astype(np.int64)
    c1 = np.clip(np.floor(xs[ia + 1]) + 1, 0, W).astype(np.int64)
    ok = c1 > c0
    diff = np.zeros((H, W + 1), dtype=np.int32)
    np.add.at(diff, (r[ok], c0[ok]), 1)
    np.add.at(diff, (r[ok], c1[ok]), -1)
    return np.cumsum(diff, axis=1)[:, :W] > 0


def rasterize_lines(lines: list[tuple[np.ndarray, float]], shape: tuple[int, int]) -> np.ndarray:
    """Rasterise polylines (vertex-grid coords) with per-line half-width (cells)."""
    H, W = shape
    out = np.zeros(shape, dtype=bool)
    if not lines:
        return out
    # group by half-width so one distance transform serves many lines
    by_w: dict[float, list[np.ndarray]] = {}
    for pts, hw in lines:
        by_w.setdefault(round(hw * 4) / 4, []).append(np.asarray(pts, dtype=np.float64))
    for hw, plist in by_w.items():
        center = np.zeros(shape, dtype=bool)
        for p in plist:
            if len(p) < 2:
                continue
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
        if not center.any():
            continue
        if hw <= 0.5:
            out |= center
        else:
            d = ndimage.distance_transform_edt(~center)
            out |= d <= hw
    return out


# ---------------------------------------------------------------- labels

def vertex_to_cell_frac(v: np.ndarray) -> np.ndarray:
    """Fraction of a cell's 4 corners that are True."""
    v = v.astype(np.float32)
    return (v[:-1, :-1] + v[:-1, 1:] + v[1:, :-1] + v[1:, 1:]) * 0.25


def remove_small(mask: np.ndarray, min_size: int) -> np.ndarray:
    if min_size <= 1 or not mask.any():
        return mask
    lab, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    keep = sizes >= min_size
    keep[0] = False
    return keep[lab]


def fix_pinches(labels: np.ndarray, protect: int | None = -1, max_iter: int = 40) -> np.ndarray:
    """Remove diagonal-only contacts between same-label cells.

    Two cells of label L touching only at a corner produce a non-manifold
    vertex/edge in the extruded mesh. We reassign one off-diagonal cell to L
    (never touching ``protect`` cells unless unavoidable).
    """
    L = labels.copy()
    H, W = L.shape
    for _ in range(max_iter):
        changed = False
        for pr in (0, 1):
            for pc in (0, 1):
                # non-overlapping 2x2 windows of this parity (views into L)
                h, w = (H - pr) // 2, (W - pc) // 2
                if h == 0 or w == 0:
                    continue
                A = L[pr:pr + 2 * h:2, pc:pc + 2 * w:2]
                B = L[pr:pr + 2 * h:2, pc + 1:pc + 1 + 2 * w:2]
                C = L[pr + 1:pr + 1 + 2 * h:2, pc:pc + 2 * w:2]
                D = L[pr + 1:pr + 1 + 2 * h:2, pc + 1:pc + 1 + 2 * w:2]
                p1 = (A == D) & (B != A) & (C != A)  # diagonal A-D
                p2 = (B == C) & (A != B) & (D != B) & ~p1  # diagonal B-C
                if protect is not None:
                    p1 &= A != protect
                    p2 &= B != protect
                if not (p1.any() or p2.any()):
                    continue
                changed = True
                # p1: give B (or C if B protected) label A
                if protect is not None:
                    useB = p1 & (B != protect)
                    useC = p1 & ~useB & (C != protect)
                    drop = p1 & ~useB & ~useC
                else:
                    useB, useC, drop = p1, np.zeros_like(p1), np.zeros_like(p1)
                B[useB] = A[useB]
                C[useC] = A[useC]
                if protect is not None:
                    D[drop] = protect
                if protect is not None:
                    useA = p2 & (A != protect)
                    useD = p2 & ~useA & (D != protect)
                    drop2 = p2 & ~useA & ~useD
                else:
                    useA, useD, drop2 = p2, np.zeros_like(p2), np.zeros_like(p2)
                A[useA] = B[useA]
                D[useD] = B[useD]
                if protect is not None:
                    C[drop2] = protect
        if not changed:
            break
    return L


def fix_mask_pinches(mask: np.ndarray, allowed: np.ndarray | None = None, max_iter: int = 40) -> np.ndarray:
    """Binary version: grow the mask (inside ``allowed``) to remove pinches."""
    M = mask.copy()
    for _ in range(max_iter):
        A, B, C, D = M[:-1, :-1], M[:-1, 1:], M[1:, :-1], M[1:, 1:]
        p1 = A & D & ~B & ~C
        p2 = B & C & ~A & ~D
        if not (p1.any() or p2.any()):
            break
        r, c = np.nonzero(p1)
        for rr, cc in zip(r, c):
            if allowed is None or allowed[rr, cc + 1]:
                M[rr, cc + 1] = True
            elif allowed[rr + 1, cc]:
                M[rr + 1, cc] = True
            else:
                M[rr + 1, cc + 1] = False
        r, c = np.nonzero(p2)
        for rr, cc in zip(r, c):
            if allowed is None or allowed[rr, cc]:
                M[rr, cc] = True
            elif allowed[rr + 1, cc + 1]:
                M[rr + 1, cc + 1] = True
            else:
                M[rr + 1, cc] = False
    return M
