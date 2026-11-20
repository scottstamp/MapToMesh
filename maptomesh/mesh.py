"""Watertight solids from masked heightfields.

``heightfield_solid`` turns a cell mask plus a top surface (scalar or
per-vertex heights) and a flat bottom into a closed, consistently oriented,
2-manifold triangle mesh:

* top   : two triangles per cell (diagonal picked to follow ridges), or
          row-strips when the top is flat
* bottom: row-strips "zippered" so every vertex on a shared line appears in
          both neighbouring strips (no T-junctions)
* walls : one quad per boundary cell edge

Masks must be free of diagonal-only contacts (see raster.fix_pinches).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Mesh:
    vertices: np.ndarray  # (n, 3) float32, millimetres
    triangles: np.ndarray  # (m, 3) int32

    def translate(self, dx: float, dy: float, dz: float = 0.0) -> "Mesh":
        v = self.vertices.copy()
        v[:, 0] += dx; v[:, 1] += dy; v[:, 2] += dz
        return Mesh(v, self.triangles)

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.vertices.min(axis=0), self.vertices.max(axis=0)


def _segments(start: np.ndarray, count: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For strips with ``count`` segments starting at global index ``start``,
    return (strip id, first-vertex global index) for every segment."""
    count = np.maximum(count, 0).astype(np.intp)   # repeat counts: intp is 32-bit in WebAssembly
    total = int(count.sum())
    strip = np.repeat(np.arange(len(start)), count)
    g = np.repeat(start, count) + (np.arange(total) - np.repeat(np.cumsum(count) - count, count))
    return strip, g


def _strips(m: np.ndarray):
    """Zipper-triangulate a cell mask as row strips (+z winding).

    Returns (need, (rows, cols), tris) where ``need`` marks the vertices used,
    (rows, cols) lists them in row-major order and ``tris`` indexes that list.
    Every vertex on a strip edge that a neighbouring strip or a boundary wall
    uses is included, so there are no T-junctions.
    """
    R, C = m.shape
    mp = np.zeros((R, C + 2), dtype=np.int8)
    mp[:, 1:-1] = m
    d = np.diff(mp, axis=1)
    sr, sc = np.nonzero(d == 1)
    er, ec = np.nonzero(d == -1)
    need = np.zeros((R + 1, C + 1), dtype=bool)
    need[sr, sc] = True; need[sr + 1, sc] = True
    need[er, ec] = True; need[er + 1, ec] = True
    mr = np.zeros((R + 2, C), dtype=bool)
    mr[1:-1] = m
    hr, hc = np.nonzero(mr[:-1] != mr[1:])
    need[hr, hc] = True; need[hr, hc + 1] = True
    nr, nc = np.nonzero(need)
    W1 = C + 1
    key_all = nr.astype(np.int64) * W1 + nc
    r64 = sr.astype(np.int64)
    bs = np.searchsorted(key_all, r64 * W1 + sc, "left")
    be = np.searchsorted(key_all, r64 * W1 + ec, "right")
    ts = np.searchsorted(key_all, (r64 + 1) * W1 + sc, "left")
    te = np.searchsorted(key_all, (r64 + 1) * W1 + ec, "right")
    # bottom-line segments -> apex: last top-line vertex strictly left of segment end
    s, g = _segments(bs, be - bs - 1)
    p = np.searchsorted(key_all, (r64[s] + 1) * W1 + nc[g + 1], "left") - 1
    a = np.column_stack([g, g + 1, p])
    # top-line segments -> apex: last bottom-line vertex not right of segment end
    s, g = _segments(ts, te - ts - 1)
    q = np.searchsorted(key_all, r64[s] * W1 + nc[g + 1], "right") - 1
    b = np.column_stack([q, g + 1, g])
    return need, (nr, nc), np.concatenate([a, b])


def heightfield_solid(mask: np.ndarray, top, bottom: float, dx: float, dy: float,
                      ox: float = 0.0, oy: float = 0.0, min_thickness: float = 0.01) -> Mesh | None:
    m = np.asarray(mask, dtype=bool)
    R, C = m.shape
    if not m.any():
        return None
    flat_top = np.isscalar(top)
    if not flat_top:
        top = np.maximum(np.asarray(top, dtype=np.float64), bottom + min_thickness)
        assert top.shape == (R + 1, C + 1), (top.shape, m.shape)

    # --- runs along rows (needed for vertical walls)
    mp = np.zeros((R, C + 2), dtype=np.int8)
    mp[:, 1:-1] = m
    d = np.diff(mp, axis=1)  # (R, C+1): +1 start, -1 end at vertex column
    mr = np.zeros((R + 2, C), dtype=bool)
    mr[1:-1] = m
    hb = mr[:-1] != mr[1:]  # (R+1, C) horizontal boundary edges per vertex line
    hr, hc = np.nonzero(hb)

    # --- bottom (always flat): strips over the whole mask
    need, (nr, nc), strips_up = _strips(m)
    nneed = len(nr)
    bot_id = np.full((R + 1, C + 1), -1, dtype=np.int64)
    bot_id[nr, nc] = np.arange(nneed)
    verts = [np.column_stack([ox + nc * dx, oy + nr * dy, np.full(nneed, float(bottom))])]
    tris = [strips_up[:, ::-1]]  # bottom face (-z): reversed winding

    if flat_top:
        top_id = np.where(bot_id >= 0, bot_id + nneed, -1)
        verts.append(np.column_stack([ox + nc * dx, oy + nr * dy, np.full(nneed, float(top))]))
        tris.append(strips_up + nneed)
    else:
        # cells whose four corners are exactly equal are triangulated as strips
        t00, t01, t10, t11 = top[:-1, :-1], top[:-1, 1:], top[1:, :-1], top[1:, 1:]
        flat = m & (t00 == t01) & (t00 == t10) & (t00 == t11)
        nf = m & ~flat
        used = np.zeros((R + 1, C + 1), dtype=bool)
        used[:-1, :-1] |= nf; used[:-1, 1:] |= nf; used[1:, :-1] |= nf; used[1:, 1:] |= nf
        if flat.any():
            need_f, (fr, fc), fstrips = _strips(flat)
            used |= need_f
        ur, uc = np.nonzero(used)
        top_id = np.full((R + 1, C + 1), -1, dtype=np.int64)
        top_id[ur, uc] = nneed + np.arange(len(ur))
        verts.append(np.column_stack([ox + uc * dx, oy + ur * dy, top[ur, uc]]))
        if flat.any():
            tris.append(top_id[fr, fc][fstrips])
        cr, cc = np.nonzero(nf)
        v00 = top_id[cr, cc]; v01 = top_id[cr, cc + 1]
        v10 = top_id[cr + 1, cc]; v11 = top_id[cr + 1, cc + 1]
        use_a = (np.abs(t00[cr, cc] - t11[cr, cc]) <= np.abs(t01[cr, cc] - t10[cr, cc]))[:, None]
        ta1 = np.column_stack([v00, v01, v11]); ta2 = np.column_stack([v00, v11, v10])
        tb1 = np.column_stack([v00, v01, v10]); tb2 = np.column_stack([v01, v11, v10])
        tris.append(np.where(use_a, ta1, tb1))
        tris.append(np.where(use_a, ta2, tb2))

    # --- walls along x (horizontal edges)
    north_in = mr[hr + 1, hc]
    a_b, b_b = bot_id[hr, hc], bot_id[hr, hc + 1]
    a_t, b_t = top_id[hr, hc], top_id[hr, hc + 1]
    n_in = north_in[:, None]
    w1 = np.where(n_in, np.column_stack([a_b, b_b, b_t]), np.column_stack([b_b, a_b, a_t]))
    w2 = np.where(n_in, np.column_stack([a_b, b_t, a_t]), np.column_stack([b_b, a_t, b_t]))
    tris += [w1, w2]

    # --- walls along y (vertical edges)
    vr, vc = np.nonzero(d != 0)
    east_in = (d[vr, vc] == 1)[:, None]
    a_b, b_b = bot_id[vr, vc], bot_id[vr + 1, vc]
    a_t, b_t = top_id[vr, vc], top_id[vr + 1, vc]
    w1 = np.where(east_in, np.column_stack([b_b, a_b, a_t]), np.column_stack([a_b, b_b, b_t]))
    w2 = np.where(east_in, np.column_stack([b_b, a_t, b_t]), np.column_stack([a_b, b_t, a_t]))
    tris += [w1, w2]

    V = np.concatenate(verts).astype(np.float32)
    T = np.concatenate(tris).astype(np.int64)
    if (T < 0).any():
        raise RuntimeError("internal mesh error: unresolved vertex index")
    return Mesh(V, T.astype(np.int32))


def check_manifold(mesh: Mesh) -> dict:
    """Edge-use statistics; a closed oriented manifold has every directed edge
    exactly once and its reverse exactly once."""
    t = mesh.triangles.astype(np.int64)
    e = np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])
    n = int(mesh.vertices.shape[0]) + 1
    directed = e[:, 0] * n + e[:, 1]
    reverse = e[:, 1] * n + e[:, 0]
    u, cnt = np.unique(directed, return_counts=True)
    dup_directed = int((cnt > 1).sum())
    missing_reverse = int((~np.isin(reverse, u)).sum())
    v = mesh.vertices.astype(np.float64)[t]
    area2 = np.linalg.norm(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1)
    vol = float(np.einsum("ij,ij->i", v[:, 0], np.cross(v[:, 1], v[:, 2])).sum() / 6.0)
    return {"triangles": len(t), "dup_directed_edges": dup_directed,
            "open_edges": missing_reverse, "degenerate": int((area2 < 1e-12).sum()), "volume": vol}
