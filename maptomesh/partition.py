"""Exact iso-line partition of a triangulated grid, and watertight solids
built from the resulting pieces.

The vertex grid is triangulated (two triangles per cell). A sequence of
*steps* splits every triangle along the iso-line ``field == threshold`` of a
per-vertex scalar field (shape outline, smoothed class indicators, elevation
contours). Because the fields are linear on each triangle the cuts are exact
straight segments, so region outlines follow smooth lines instead of cell
staircases. Points created on a shared triangle edge get a key derived from
the edge and step, so neighbouring triangles reuse the same vertex and the
resulting surfaces are watertight.

``build_part`` then turns a set of labels into a closed 2-manifold solid with
an arbitrary top and bottom height field (vertical walls on the outline,
knife edges where top meets bottom, flat areas strip-triangulated).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mesh import Mesh, _strips

ACTIVE, DROPPED, SPLIT = -1, -2, -3
TAU = 0.1          # min. relative distance of a cut from an edge end (avoids slivers)
MAX_STEPS = 64
INTERIOR_BASE = np.int64(1) << np.int64(61)


@dataclass
class FieldSpec:
    """value = clip(interp(base) + offset, lo, hi); ``base`` is a scalar or a
    flat per-vertex array."""
    base: object
    offset: float = 0.0
    lo: float = -np.inf
    hi: float = np.inf

    def at_grid(self, vid: np.ndarray) -> np.ndarray:
        v = np.full(len(vid), float(self.base)) if np.isscalar(self.base) else self.base[vid].astype(np.float64)
        return np.clip(v + self.offset, self.lo, self.hi)

    def at_points(self, part: "Partition", tri: np.ndarray, xy: np.ndarray) -> np.ndarray:
        if np.isscalar(self.base):
            v = np.full(len(tri), float(self.base))
        else:
            v = part.interp(self.base, tri, xy)
        return np.clip(v + self.offset, self.lo, self.hi)


class Partition:
    def __init__(self, R: int, C: int, diag_a: np.ndarray):
        self.R, self.C = R, C
        self.W1 = C + 1
        self.NV = (R + 1) * (C + 1)
        r, c = np.mgrid[0:R, 0:C]
        v00 = (r * self.W1 + c).ravel().astype(np.int64)
        v01, v10 = v00 + 1, v00 + self.W1
        v11 = v10 + 1
        a = diag_a.ravel()[:, None]
        t1 = np.where(a, np.column_stack([v00, v01, v11]), np.column_stack([v00, v01, v10]))
        t2 = np.where(a, np.column_stack([v00, v11, v10]), np.column_stack([v01, v11, v10]))
        self.tris = np.empty((2 * R * C, 3), np.int64)
        self.tris[0::2] = t1
        self.tris[1::2] = t2
        self.tri_label = np.full(len(self.tris), ACTIVE, np.int16)
        # polygons (dense, padded)
        self.M = 3
        self.p_keys = np.zeros((0, 3), np.int64)
        self.p_xy = np.zeros((0, 3, 2))
        self.p_org = np.zeros((0, 3), np.int64)
        self.p_cnt = np.zeros(0, np.int64)
        self.p_tri = np.zeros(0, np.int64)
        self.p_label = np.zeros(0, np.int16)
        self.steps: list[tuple[np.ndarray, float, int | None, int | None]] = []
        self._interior = INTERIOR_BASE

    # ------------------------------------------------------------ geometry helpers
    def vxy(self, vid: np.ndarray) -> np.ndarray:
        return np.column_stack([vid % self.W1, vid // self.W1]).astype(np.float64)

    def interp(self, field: np.ndarray, tri: np.ndarray, xy: np.ndarray) -> np.ndarray:
        """Linear interpolation of a per-vertex field inside source triangles."""
        t = self.tris[tri]
        A, B, Cc = self.vxy(t[:, 0]), self.vxy(t[:, 1]), self.vxy(t[:, 2])
        e1, e2, d = B - A, Cc - A, xy - A
        det = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
        l1 = (d[:, 0] * e2[:, 1] - d[:, 1] * e2[:, 0]) / det
        l2 = (e1[:, 0] * d[:, 1] - e1[:, 1] * d[:, 0]) / det
        fa, fb, fc = field[t[:, 0]], field[t[:, 1]], field[t[:, 2]]
        return fa + l1 * (fb - fa) + l2 * (fc - fa)

    def _poly_field(self, field: np.ndarray, idx: np.ndarray) -> np.ndarray:
        keys = self.p_keys[idx]
        out = np.zeros(keys.shape)
        valid = np.arange(self.M)[None, :] < self.p_cnt[idx][:, None]
        grid = valid & (keys < self.NV) & (keys >= 0)
        out[grid] = field[keys[grid]]  # exact values at grid vertices
        other = valid & ~grid
        if other.any():
            rows, cols = np.nonzero(other)
            out[rows, cols] = self.interp(field, self.p_tri[idx][rows], self.p_xy[idx][rows, cols])
        return out

    def _grow(self, M: int) -> None:
        if M <= self.M:
            return
        k = M - self.M
        self.p_keys = np.pad(self.p_keys, ((0, 0), (0, k)), constant_values=-1)
        self.p_xy = np.pad(self.p_xy, ((0, 0), (0, k), (0, 0)))
        self.p_org = np.pad(self.p_org, ((0, 0), (0, k)), constant_values=-1)
        self.M = M

    # ------------------------------------------------------------ steps
    def step(self, field: np.ndarray, th: float, above: int | None = None, below: int | None = None) -> None:
        """Split along ``field == th``. ``above``/``below``: final label for that side
        (None keeps it active for later steps, DROPPED discards it)."""
        field = np.asarray(field, dtype=np.float64).ravel()
        sidx = len(self.steps)
        assert sidx < MAX_STEPS
        self.steps.append((field, th, above, below))

        # full triangles
        act = np.nonzero(self.tri_label == ACTIVE)[0]
        fv = field[self.tris[act]]
        fmax, fmin = fv.max(axis=1), fv.min(axis=1)
        out = fmax <= th
        inn = ~out & (fmin >= th)
        cross = ~out & ~inn
        if below is not None:
            self.tri_label[act[out]] = below
        if above is not None:
            self.tri_label[act[inn]] = above
        ct = act[cross]
        if len(ct):
            self.tri_label[ct] = SPLIT
            t = self.tris[ct]
            n = len(ct)
            keys = np.full((n, self.M), -1, np.int64)
            keys[:, :3] = t
            xy = np.zeros((n, self.M, 2))
            for k in range(3):
                xy[:, k] = self.vxy(t[:, k])
            org = np.full((n, self.M), -1, np.int64)
            for k in range(3):
                a, b = t[:, k], t[:, (k + 1) % 3]
                org[:, k] = np.minimum(a, b) * self.NV + np.maximum(a, b)
            self.p_keys = np.concatenate([self.p_keys, keys])
            self.p_xy = np.concatenate([self.p_xy, xy])
            self.p_org = np.concatenate([self.p_org, org])
            self.p_cnt = np.concatenate([self.p_cnt, np.full(n, 3)])
            self.p_tri = np.concatenate([self.p_tri, ct])
            self.p_label = np.concatenate([self.p_label, np.full(n, ACTIVE, np.int16)])

        # polygons
        pidx = np.nonzero(self.p_label == ACTIVE)[0]
        if not len(pidx):
            return
        f = self._poly_field(field, pidx)
        M = self.M
        valid = np.arange(M)[None, :] < self.p_cnt[pidx][:, None]
        fmax = np.where(valid, f, -np.inf).max(axis=1)
        fmin = np.where(valid, f, np.inf).min(axis=1)
        out = fmax <= th
        inn = ~out & (fmin >= th)
        if below is not None:
            self.p_label[pidx[out]] = below
        if above is not None:
            self.p_label[pidx[inn]] = above
        cr = ~out & ~inn
        if not cr.any():
            return
        ci = pidx[cr]
        self._clip(ci, f[cr], th, sidx, above, below)

    def _clip(self, ci: np.ndarray, f: np.ndarray, th: float, sidx: int, above, below) -> None:
        M = self.M
        n = len(ci)
        cnt = self.p_cnt[ci]
        keys, xy, org = self.p_keys[ci], self.p_xy[ci], self.p_org[ci]
        ii = np.arange(M)[None, :]
        valid = ii < cnt[:, None]
        nxt = (ii + 1) % cnt[:, None]
        s = np.sign(f - th)
        s[~valid] = 0
        s_n = np.take_along_axis(s, nxt, 1)
        f_n = np.take_along_axis(f, nxt, 1)
        xy_n = np.take_along_axis(xy, nxt[..., None], 1)
        cross = valid & (s * s_n < 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.clip((f - th) / (f - f_n), TAU, 1 - TAU)
        t = np.where(cross, t, 0.0)
        pxy = xy + t[..., None] * (xy_n - xy)
        pkey = np.where(org >= 0, self.NV + org * MAX_STEPS + sidx, -1)
        need_int = cross & (org < 0)
        k = int(need_int.sum())
        if k:
            pkey[need_int] = self._interior + np.arange(k)
            self._interior += k
        # candidate slots: 2i -> vertex i, 2i+1 -> crossing on edge i
        ck = np.empty((n, 2 * M), np.int64); ck[:, 0::2] = keys; ck[:, 1::2] = pkey
        cxy = np.empty((n, 2 * M, 2)); cxy[:, 0::2] = xy; cxy[:, 1::2] = pxy
        slot = np.broadcast_to(np.arange(2 * M), (n, 2 * M))
        newM = M + 1
        self._grow(newM)

        results = []
        for side in (1, -1):
            pres = np.zeros((n, 2 * M), bool)
            pres[:, 0::2] = valid & (s * side >= 0)
            pres[:, 1::2] = cross
            order = np.argsort(~pres, axis=1, kind="stable")
            ocnt = pres.sum(axis=1)
            sk = np.take_along_axis(ck, order, 1)[:, :newM]
            sxy = np.take_along_axis(cxy, order[..., None], 1)[:, :newM]
            ss = np.take_along_axis(slot, order, 1)[:, :newM]
            pos = np.arange(newM)[None, :]
            ovalid = pos < ocnt[:, None]
            nslot = np.take_along_axis(ss, (pos + 1) % np.maximum(ocnt, 1)[:, None], 1)
            edge_i = ss // 2
            on_edge = (nslot == 2 * edge_i + 1) | (nslot == 2 * ((edge_i + 1) % cnt[:, None]))
            sorg = np.where(on_edge & ovalid, np.take_along_axis(org, np.minimum(edge_i, M - 1), 1), -1)
            sk = np.where(ovalid, sk, -1)
            results.append((sk, sxy, sorg, ocnt))

        (ka, xa, oa, na), (kb, xb, ob, nb) = results
        # inside part replaces the original row, outside part is appended
        self.p_keys[ci] = ka; self.p_xy[ci] = xa; self.p_org[ci] = oa; self.p_cnt[ci] = na
        self.p_label[ci] = ACTIVE if above is None else above
        lab_b = ACTIVE if below is None else below
        self.p_keys = np.concatenate([self.p_keys, kb])
        self.p_xy = np.concatenate([self.p_xy, xb])
        self.p_org = np.concatenate([self.p_org, ob])
        self.p_cnt = np.concatenate([self.p_cnt, nb])
        self.p_tri = np.concatenate([self.p_tri, self.p_tri[ci]])
        self.p_label = np.concatenate([self.p_label, np.full(n, lab_b, np.int16)])

    def finish(self, label: int) -> None:
        self.tri_label[self.tri_label == ACTIVE] = label
        self.p_label[self.p_label == ACTIVE] = label

    def classify(self, values: list[np.ndarray], default: int) -> np.ndarray:
        """Apply the recorded steps to arbitrary sample points (e.g. cell centres)."""
        lab = np.full(values[0].shape, ACTIVE, np.int16)
        for (field, th, above, below), v in zip(self.steps, values):
            act = lab == ACTIVE
            if below is not None:
                lab[act & (v <= th)] = below
            if above is not None:
                lab[act & (v > th)] = above
        lab[lab == ACTIVE] = default
        return lab

    # ------------------------------------------------------------ solids
    def build_part(self, labels, top: FieldSpec, bottom: FieldSpec, window: tuple[int, int, int, int],
                   scale: float, ox: float, oy: float) -> Mesh | None:
        r0, r1, c0, c1 = window
        labels = np.asarray(list(labels), dtype=np.int16)
        cell_of = np.arange(len(self.tris)) // 2
        cr, cc = cell_of // self.C, cell_of % self.C
        in_win_t = (cr >= r0) & (cr < r1) & (cc >= c0) & (cc < c1)
        ft = np.nonzero(np.isin(self.tri_label, labels) & in_win_t)[0]
        pcell = self.p_tri // 2
        pin = ((pcell // self.C >= r0) & (pcell // self.C < r1) & (pcell % self.C >= c0) & (pcell % self.C < c1))
        pp = np.nonzero(np.isin(self.p_label, labels) & pin)[0]
        if not len(ft) and not len(pp):
            return None

        full = self.tris[ft]
        pcnt = self.p_cnt[pp]
        pvalid = np.arange(self.M)[None, :] < pcnt[:, None]
        pkeys = self.p_keys[pp][pvalid]          # flat, row-major
        pxy = self.p_xy[pp][pvalid]
        ptri = np.repeat(self.p_tri[pp], pcnt)
        poff = np.concatenate([[0], np.cumsum(pcnt)])

        all_keys = np.concatenate([full.ravel(), pkeys])
        uk, inv = np.unique(all_keys, return_inverse=True)
        nv = len(uk)
        isg = uk < self.NV
        pos = np.zeros((nv, 2))
        pos[isg] = self.vxy(uk[isg])
        inv_full = inv[:full.size].reshape(-1, 3)
        inv_p = inv[full.size:]
        first = np.full(nv, -1, np.int64)
        first[inv_p[::-1]] = np.arange(len(inv_p))[::-1]  # first occurrence in polygons
        ng = ~isg
        pos[ng] = pxy[first[ng]]

        def values(spec: FieldSpec) -> np.ndarray:
            v = np.empty(nv)
            v[isg] = spec.at_grid(uk[isg])
            if ng.any():
                v[ng] = spec.at_points(self, ptri[first[ng]], pos[ng])
            return v

        topv, botv = values(top), values(bottom)
        topv = np.maximum(topv, botv)

        # --- fine-topology directed edges -> outline
        e_full = np.concatenate([inv_full[:, [0, 1]], inv_full[:, [1, 2]], inv_full[:, [2, 0]]])
        nxt = np.arange(len(inv_p)) + 1
        last = poff[1:] - 1
        nxt[last] = poff[:-1]
        e_poly = np.column_stack([inv_p, inv_p[nxt]])
        edges = np.concatenate([e_full, e_poly])
        ukey = np.minimum(edges[:, 0], edges[:, 1]) * nv + np.maximum(edges[:, 0], edges[:, 1])
        _, ui, uc = np.unique(ukey, return_inverse=True, return_counts=True)
        bnd = edges[uc[ui] == 1]
        on_bnd = np.zeros(nv, bool)
        on_bnd[bnd.ravel()] = True
        # exact zero thickness on the outline -> knife edge (shared vertex); walls thinner than
        # 10 um are lifted to 10 um so they survive coordinate rounding in the 3MF
        thin = on_bnd & (topv - botv < 0.01)
        knife = thin & (topv - botv < 1e-6)
        lift = thin & ~knife
        topv[lift] = botv[lift] + 0.01
        topv[knife] = botv[knife]
        top_id = np.arange(nv)
        bot_id = np.where(knife, top_id, nv + np.cumsum(~knife) - 1)
        nb = int((~knife).sum())

        # --- faces: full cells (both halves selected) may be strip-decimated when flat
        cells_t = cell_of[ft]
        uc_cells, ccount = np.unique(cells_t, return_counts=True)
        full_cells = uc_cells[ccount == 2]
        win_h, win_w = r1 - r0, c1 - c0

        def surface(vals: np.ndarray, ids: np.ndarray, flip: bool) -> list[np.ndarray]:
            out = []
            fc = full_cells
            v00 = (fc // self.C) * self.W1 + fc % self.C
            corner = np.column_stack([v00, v00 + 1, v00 + self.W1, v00 + self.W1 + 1])
            li = np.searchsorted(uk, corner)
            cv = vals[li]
            flat = (cv[:, 0] == cv[:, 1]) & (cv[:, 0] == cv[:, 2]) & (cv[:, 0] == cv[:, 3])
            flat_cells = fc[flat]
            tri_sel = ~np.isin(cells_t, flat_cells)
            tris = ids[inv_full[tri_sel]]
            out.append(tris)
            if len(flat_cells):
                m = np.zeros((win_h, win_w), bool)
                m[flat_cells // self.C - r0, flat_cells % self.C - c0] = True
                _, (nr, nc), st = _strips(m)
                gid = (nr + r0) * self.W1 + (nc + c0)
                out.append(ids[np.searchsorted(uk, gid)][st])
            # polygon fans
            if len(pp):
                kmax = int(pcnt.max())
                rows = []
                for j in range(1, kmax - 1):
                    sel = pcnt > j + 1
                    base = poff[:-1][sel]
                    rows.append(np.column_stack([inv_p[base], inv_p[base + j], inv_p[base + j + 1]]))
                if rows:
                    out.append(ids[np.concatenate(rows)])
            if flip:
                out = [o[:, ::-1] for o in out]
            return out

        faces = surface(topv, top_id, False) + surface(botv, bot_id, True)
        # walls on the outline: top edge u->w  =>  (w_t, u_t, u_b), (w_t, u_b, w_b)
        u, w = bnd[:, 0], bnd[:, 1]
        faces.append(np.column_stack([top_id[w], top_id[u], bot_id[u]]))
        faces.append(np.column_stack([top_id[w], bot_id[u], bot_id[w]]))
        T = np.concatenate([f for f in faces if len(f)])
        T = T[(T[:, 0] != T[:, 1]) & (T[:, 1] != T[:, 2]) & (T[:, 0] != T[:, 2])]

        V = np.zeros((nv + nb, 3))
        V[:nv, :2] = pos
        V[:nv, 2] = topv
        V[nv:, :2] = pos[~knife]
        V[nv:, 2] = botv[~knife]
        used = np.zeros(len(V), bool)
        used[T.ravel()] = True
        remap = np.cumsum(used) - 1
        V = V[used]
        T = remap[T]
        V[:, 0] = V[:, 0] * scale + ox
        V[:, 1] = V[:, 1] * scale + oy
        return Mesh(V.astype(np.float32), T.astype(np.int32))


# ---------------------------------------------------------------- shapes

def shape_field(shape: str, R: int, C: int, w: float, h: float, corner_r: float = 0.0,
                grow: float = 0.0) -> np.ndarray:
    """Approximate signed distance (cells, positive inside) on the (R+1, C+1)
    vertex grid of a centred shape of size w x h cells, offset by ``grow``."""
    Y, X = np.mgrid[0:R + 1, 0:C + 1].astype(np.float64)
    x, y = np.abs(X - C / 2), np.abs(Y - R / 2)
    hw, hh = w / 2, h / 2
    if shape == "circle":
        rx, ry = hw + grow, hh + grow
        k = np.sqrt((x / rx) ** 2 + (y / ry) ** 2)
        return (1 - k) * min(rx, ry)
    if shape == "hexagon":
        n = np.hypot(hh, hw / 2)
        slanted = (hh * hw - hh * x - (hw / 2) * y) / n
        return np.minimum(hh - y, slanted) + grow
    if shape == "rounded" and corner_r > 0:
        r = min(corner_r, hw, hh)
        qx, qy = x - (hw - r), y - (hh - r)
        out = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - r
        return -out + grow
    return np.minimum(hw - x, hh - y) + grow
