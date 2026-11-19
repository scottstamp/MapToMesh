//! `maptomesh.partition.Partition.build_part` (+ `_quadtree`, `mesh._strips`) in Rust.
//!
//! Same algorithm and the same output order as the NumPy version, so the meshes match it: unique
//! vertex keys sorted ascending, faces in the order quadtree fans, plain triangles, flat strips,
//! polygon fans (top surface, then the flipped bottom), then the two wall triangle sets.

use rayon::prelude::*;

/// A per-vertex field: `clip(base + offset, lo, hi)`, base a flat per-grid-vertex array or a scalar.
pub struct Spec<'a> {
    pub base: Option<&'a [f64]>,
    pub scalar: f64,
    pub offset: f64,
    pub lo: f64,
    pub hi: f64,
}

impl Spec<'_> {
    fn clip(&self, v: f64) -> f64 {
        // np.clip: NaN stays NaN
        let a = if v < self.lo { self.lo } else { v };
        if a > self.hi { self.hi } else { a }
    }
}

/// The partition's arrays (see `Partition`).
pub struct Part<'a> {
    pub tris: &'a [i64],
    pub tri_label: &'a [i16],
    pub p_keys: &'a [i64],
    pub p_xy: &'a [f64],
    pub p_cnt: &'a [i64],
    pub p_tri: &'a [i64],
    pub p_label: &'a [i16],
    pub m: usize,
    pub nv: i64,
    pub c: i64,
}

impl Part<'_> {
    fn vxy(&self, vid: i64) -> (f64, f64) {
        let w1 = self.c + 1;
        ((vid % w1) as f64, (vid / w1) as f64)
    }

    /// Linear interpolation of a per-vertex field inside source triangle `tri` (Partition.interp).
    fn interp(&self, field: &[f64], tri: i64, x: f64, y: f64) -> f64 {
        let t = &self.tris[3 * tri as usize..3 * tri as usize + 3];
        let (ax, ay) = self.vxy(t[0]);
        let (bx, by) = self.vxy(t[1]);
        let (cx, cy) = self.vxy(t[2]);
        let (e1x, e1y, e2x, e2y, dx, dy) = (bx - ax, by - ay, cx - ax, cy - ay, x - ax, y - ay);
        let det = e1x * e2y - e1y * e2x;
        let l1 = (dx * e2y - dy * e2x) / det;
        let l2 = (e1x * dy - e1y * dx) / det;
        let (fa, fb, fc) = (field[t[0] as usize], field[t[1] as usize], field[t[2] as usize]);
        fa + l1 * (fb - fa) + l2 * (fc - fa)
    }
}

fn np_max(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() { f64::NAN } else if a >= b { a } else { b }
}

/// mesh._strips: zipper-triangulate a cell mask (h x w, row-major) as row strips.
/// Returns (vertex rows, vertex cols, triangles indexing that row-major vertex list).
fn strips(m: &[bool], h: usize, w: usize) -> (Vec<i64>, Vec<i64>, Vec<[usize; 3]>) {
    let w1 = w + 1;
    let mut need = vec![false; (h + 1) * w1];
    let mut runs: Vec<(usize, usize, usize)> = Vec::new(); // (row, start col, end col)
    for r in 0..h {
        let row = &m[r * w..(r + 1) * w];
        let mut c = 0;
        while c < w {
            if row[c] {
                let s = c;
                while c < w && row[c] {
                    c += 1;
                }
                runs.push((r, s, c));
            } else {
                c += 1;
            }
        }
    }
    for &(r, s, e) in &runs {
        need[r * w1 + s] = true;
        need[(r + 1) * w1 + s] = true;
        need[r * w1 + e] = true;
        need[(r + 1) * w1 + e] = true;
    }
    // horizontal boundaries: vertex row hr between cell rows hr-1 and hr
    for hr in 0..=h {
        for c in 0..w {
            let above = hr < h && m[hr * w + c];
            let below = hr > 0 && m[(hr - 1) * w + c];
            if above != below {
                need[hr * w1 + c] = true;
                need[hr * w1 + c + 1] = true;
            }
        }
    }
    let mut keys: Vec<i64> = Vec::new();
    let (mut nr, mut nc) = (Vec::new(), Vec::new());
    for (k, &n) in need.iter().enumerate() {
        if n {
            keys.push(k as i64);
            nr.push((k / w1) as i64);
            nc.push((k % w1) as i64);
        }
    }
    let left = |v: i64| keys.partition_point(|&x| x < v);
    let right = |v: i64| keys.partition_point(|&x| x <= v);
    let w1i = w1 as i64;
    let mut a = Vec::new();
    let mut b = Vec::new();
    for &(r, s, e) in &runs {
        let (r, s, e) = (r as i64, s as i64, e as i64);
        let (bs, be) = (left(r * w1i + s), right(r * w1i + e));
        for g in bs..be.saturating_sub(1).max(bs) {
            if g + 1 >= be {
                break;
            }
            let p = left((r + 1) * w1i + nc[g + 1]).checked_sub(1).unwrap_or(keys.len() - 1);
            a.push([g, g + 1, p]);
        }
    }
    for &(r, s, e) in &runs {
        let (r, s, e) = (r as i64, s as i64, e as i64);
        let (ts, te) = (left((r + 1) * w1i + s), right((r + 1) * w1i + e));
        for g in ts..te {
            if g + 1 >= te {
                break;
            }
            let q = right(r * w1i + nc[g + 1]).checked_sub(1).unwrap_or(keys.len() - 1);
            b.push([q, g + 1, g]);
        }
    }
    a.extend(b);
    (nr, nc, a)
}

/// Partition._fan_weights: weights mapping (z00, zs0, z0s, zss, zc) to the 4-triangle fan
/// surface at every vertex (u, v) of an s x s block (row-major over v, u).
fn fan_weights(s: usize) -> Vec<[f64; 5]> {
    let h = s as f64 / 2.0;
    let sf = s as f64;
    let mut wt = vec![[0.0f64; 5]; (s + 1) * (s + 1)];
    let mut k = 0;
    for v in 0..=s {
        for u in 0..=s {
            let (uf, vf) = (u as f64, v as f64);
            let (dx, dy) = (uf - h, vf - h);
            let (a, b, pa, pb) = if dy.abs() >= dx.abs() {
                if dy < 0.0 { (0, 1, (0.0, 0.0), (sf, 0.0)) } else { (3, 2, (sf, sf), (0.0, sf)) }
            } else if dx > 0.0 {
                (1, 3, (sf, 0.0), (sf, sf))
            } else {
                (2, 0, (0.0, sf), (0.0, 0.0))
            };
            let ((x1, y1), (x2, y2), (x3, y3)) = (pa, pb, (h, h));
            let det = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3);
            let l1 = ((y2 - y3) * (uf - x3) + (x3 - x2) * (vf - y3)) / det;
            let l2 = ((y3 - y1) * (uf - x3) + (x1 - x3) * (vf - y3)) / det;
            wt[k][a] += l1;
            wt[k][b] += l2;
            wt[k][4] += 1.0 - l1 - l2;
            k += 1;
        }
    }
    wt
}

struct Ctx<'a> {
    p: &'a Part<'a>,
    uk: &'a [i64],
    lut: &'a [u32],
    inv_full: &'a [usize],
    inv_p: &'a [usize],
    pcnt: &'a [usize],
    poff: &'a [usize],
    cells_t: &'a [i64],
    ft: &'a [usize],
    full_cells: &'a [i64],
    pkeys: &'a [i64],
    bnd: &'a [(usize, usize)],
    win: (i64, i64, i64, i64),
    simplify: f64,
}

impl Ctx<'_> {
    fn local(&self, cell: i64) -> usize {
        let (r0, _, c0, c1) = self.win;
        ((cell / self.p.c - r0) * (c1 - c0) + (cell % self.p.c - c0)) as usize
    }

    fn li(&self, gid: i64) -> usize {
        self.lut[gid as usize] as usize
    }

    /// Partition._quadtree for one surface. Returns (fan triangles, merged cell ids).
    fn quadtree(&self, vals: &[f64], ids: &[usize], flat: &[bool]) -> (Vec<[usize; 3]>, Vec<i64>) {
        let (r0, r1, c0, c1) = self.win;
        let (hh, wd) = ((r1 - r0) as usize, (c1 - c0) as usize);
        let (cc_, w1) = (self.p.c, self.p.c + 1);
        let fc = self.full_cells;
        let mut nf = vec![false; hh * wd];
        let mut z = vec![f64::NAN; (hh + 1) * (wd + 1)];
        for (i, &cell) in fc.iter().enumerate() {
            let (cr, cc) = ((cell / cc_ - r0) as usize, (cell % cc_ - c0) as usize);
            if !flat[i] {
                nf[cr * wd + cc] = true;
            }
            for dr in 0..2 {
                for dc in 0..2 {
                    let gid = (cr as i64 + r0 + dr as i64) * w1 + (cc as i64 + c0 + dc as i64);
                    z[(cr + dr) * (wd + 1) + cc + dc] = vals[self.li(gid)];
                }
            }
        }
        let half_tol = self.simplify / 2.0;
        let max_level = 5;
        let mut prev = nf.clone();
        let mut prev_w = wd;
        let mut blocks: Vec<(usize, Vec<bool>, usize, usize)> = Vec::new(); // (k, ok, ph, pw)
        for k in 1..=max_level {
            let s = 1usize << k;
            let (ph, pw) = (hh / s, wd / s);
            if ph == 0 || pw == 0 {
                break;
            }
            let wt = fan_weights(s);
            let mut ok = vec![false; ph * pw];
            for bi in 0..ph {
                for bj in 0..pw {
                    let ch = prev[(2 * bi) * prev_w + 2 * bj]
                        && prev[(2 * bi) * prev_w + 2 * bj + 1]
                        && prev[(2 * bi + 1) * prev_w + 2 * bj]
                        && prev[(2 * bi + 1) * prev_w + 2 * bj + 1];
                    if !ch {
                        continue;
                    }
                    let zb = |v: usize, u: usize| z[(bi * s + v) * (wd + 1) + bj * s + u];
                    let corners = [zb(0, 0), zb(0, s), zb(s, 0), zb(s, s), zb(s / 2, s / 2)];
                    let mut err = f64::NAN;
                    let mut q = 0;
                    for v in 0..=s {
                        for u in 0..=s {
                            let w = &wt[q];
                            // as BLAS computes `corners @ W.T` (fused multiply-adds, k ascending)
                            let mut fan = corners[0] * w[0];
                            for kk in 1..5 {
                                fan = corners[kk].mul_add(w[kk], fan);
                            }
                            let e = (zb(v, u) - fan).abs();
                            if !e.is_nan() && (err.is_nan() || e > err) {
                                err = e;
                            }
                            q += 1;
                        }
                    }
                    if err <= half_tol {
                        ok[bi * pw + bj] = true;
                    }
                }
            }
            let any = ok.iter().any(|&o| o);
            blocks.push((k, ok.clone(), ph, pw));
            prev = ok;
            prev_w = pw;
            if !any {
                break;
            }
        }
        // leaves: merged at level k, not inside a merged block of a higher level
        let mut covered = vec![false; hh * wd];
        let mut leaves: Vec<(usize, Vec<(usize, usize)>)> = Vec::new();
        for (k, ok, ph, pw) in blocks.iter().rev() {
            let s = 1usize << k;
            let mut keep = Vec::new();
            for bi in 0..*ph {
                for bj in 0..*pw {
                    if ok[bi * pw + bj] && !covered[(bi * s) * wd + bj * s] {
                        keep.push((bi, bj));
                    }
                }
            }
            if keep.is_empty() {
                continue;
            }
            for &(bi, bj) in &keep {
                for r in bi * s..(bi + 1) * s {
                    for c in bj * s..(bj + 1) * s {
                        covered[r * wd + c] = true;
                    }
                }
            }
            leaves.push((*k, keep));
        }
        if leaves.is_empty() {
            return (Vec::new(), Vec::new());
        }
        // vertices every leaf must include on its perimeter
        let wv = wd + 1;
        let mut need = vec![false; (hh + 1) * wv];
        let mut is_full = vec![false; hh * wd];
        let mut flat_cell = vec![false; hh * wd];
        for (i, &cell) in fc.iter().enumerate() {
            let l = self.local(cell);
            is_full[l] = true;
            if flat[i] {
                flat_cell[l] = true;
            }
        }
        for r in 0..hh {
            for c in 0..wd {
                let l = r * wd + c;
                if (nf[l] && !covered[l]) || flat_cell[l] {
                    need[r * wv + c] = true;
                    need[r * wv + c + 1] = true;
                    need[(r + 1) * wv + c] = true;
                    need[(r + 1) * wv + c + 1] = true;
                }
            }
        }
        let mut mark = |g: i64| {
            let (gr, gc) = (g / w1 - r0, g % w1 - c0);
            if gr >= 0 && gr <= hh as i64 && gc >= 0 && gc <= wd as i64 {
                need[gr as usize * wv + gc as usize] = true;
            }
        };
        for (i, &t) in self.ft.iter().enumerate() {
            if !is_full[self.local(self.cells_t[i])] {
                for k in 0..3 {
                    mark(self.p.tris[3 * t + k]);
                }
            }
        }
        for &k in self.pkeys {
            if k >= 0 && k < self.p.nv {
                mark(k);
            }
        }
        for &(a, b) in self.bnd {
            for v in [a, b] {
                if self.uk[v] < self.p.nv {
                    mark(self.uk[v]);
                }
            }
        }
        for (k, keep) in &leaves {
            let s = 1usize << k;
            for &(bi, bj) in keep {
                for dr in [0, s] {
                    for dc in [0, s] {
                        need[(bi * s + dr) * wv + bj * s + dc] = true;
                    }
                }
            }
        }
        let mut tris = Vec::new();
        for (k, keep) in &leaves {
            let s = 1usize << k;
            // CCW perimeter (u = col, v = row): bottom, right, top, left (corners once each)
            let per: Vec<(usize, usize)> = (0..4 * s)
                .map(|p| match p / s {
                    0 => (p, 0),
                    1 => (s, p - s),
                    2 => (s - (p - 2 * s), s),
                    _ => (0, s - (p - 3 * s)),
                })
                .collect();
            for &(bi, bj) in keep {
                let gid = |r: usize, c: usize| (r as i64 + r0) * w1 + (c as i64 + c0);
                let centre = ids[self.li(gid(bi * s + s / 2, bj * s + s / 2))];
                let needed: Vec<usize> = per
                    .iter()
                    .filter(|&&(u, v)| need[(bi * s + v) * wv + bj * s + u])
                    .map(|&(u, v)| ids[self.li(gid(bi * s + v, bj * s + u))])
                    .collect();
                for i in 0..needed.len() {
                    tris.push([centre, needed[i], needed[(i + 1) % needed.len()]]);
                }
            }
        }
        let mut merged = Vec::new();
        for r in 0..hh {
            for c in 0..wd {
                if covered[r * wd + c] {
                    merged.push((r as i64 + r0) * cc_ + (c as i64 + c0));
                }
            }
        }
        (tris, merged)
    }

    /// One surface (top, or bottom flipped) of the solid.
    fn surface(&self, vals: &[f64], ids: &[usize], flip: bool) -> Vec<[usize; 3]> {
        let (r0, r1, c0, c1) = self.win;
        let (hh, wd) = ((r1 - r0) as usize, (c1 - c0) as usize);
        let (cc_, w1) = (self.p.c, self.p.c + 1);
        let fc = self.full_cells;
        let mut out = Vec::new();
        let flat: Vec<bool> = fc
            .iter()
            .map(|&cell| {
                let v00 = (cell / cc_) * w1 + cell % cc_;
                let (a, b, c, d) = (vals[self.li(v00)], vals[self.li(v00 + 1)], vals[self.li(v00 + w1)], vals[self.li(v00 + w1 + 1)]);
                a == b && a == c && a == d
            })
            .collect();
        let mut skip = vec![false; hh * wd];
        let mut flat_mask = vec![false; hh * wd];
        let mut any_flat = false;
        for (i, &cell) in fc.iter().enumerate() {
            if flat[i] {
                let l = self.local(cell);
                skip[l] = true;
                flat_mask[l] = true;
                any_flat = true;
            }
        }
        if self.simplify > 0.0 && flat.iter().any(|&f| !f) {
            let (qt, merged) = self.quadtree(vals, ids, &flat);
            out.extend(qt);
            for cell in merged {
                skip[self.local(cell)] = true;
            }
        }
        for (i, &cell) in self.cells_t.iter().enumerate() {
            if !skip[self.local(cell)] {
                let t = &self.inv_full[3 * i..3 * i + 3];
                out.push([ids[t[0]], ids[t[1]], ids[t[2]]]);
            }
        }
        if any_flat {
            let (nr, nc, st) = strips(&flat_mask, hh, wd);
            let vid: Vec<usize> = nr.iter().zip(&nc).map(|(&r, &c)| ids[self.li((r + r0) * w1 + (c + c0))]).collect();
            for t in st {
                out.push([vid[t[0]], vid[t[1]], vid[t[2]]]);
            }
        }
        if !self.pcnt.is_empty() {
            let kmax = *self.pcnt.iter().max().unwrap();
            for j in 1..kmax.saturating_sub(1) {
                for (q, &cnt) in self.pcnt.iter().enumerate() {
                    if cnt > j + 1 {
                        let base = self.poff[q];
                        out.push([ids[self.inv_p[base]], ids[self.inv_p[base + j]], ids[self.inv_p[base + j + 1]]]);
                    }
                }
            }
        }
        if flip {
            for t in out.iter_mut() {
                t.swap(0, 2);
            }
        }
        out
    }
}

/// Partition.build_part: a closed solid from the pieces labelled `labels` inside `win`
/// (cells r0..r1 x c0..c1), with top / bottom fields. Returns (vertices xyz f32, triangles i32).
#[allow(clippy::too_many_arguments)]
pub fn build_part(p: &Part, labels: &[i16], top: &Spec, bottom: &Spec, win: (i64, i64, i64, i64), scale: f64, ox: f64,
                  oy: f64, simplify: f64) -> Option<(Vec<f32>, Vec<i32>)> {
    let (r0, r1, c0, c1) = win;
    let cc_ = p.c;
    let in_win = |cell: i64| {
        let (r, c) = (cell / cc_, cell % cc_);
        r >= r0 && r < r1 && c >= c0 && c < c1
    };
    let lab = |l: i16| labels.contains(&l);
    let ft: Vec<usize> = (0..p.tri_label.len())
        .into_par_iter()
        .filter(|&t| lab(p.tri_label[t]) && in_win((t / 2) as i64))
        .collect();
    let pp: Vec<usize> = (0..p.p_cnt.len())
        .into_par_iter()
        .filter(|&q| lab(p.p_label[q]) && in_win(p.p_tri[q] / 2))
        .collect();
    if ft.is_empty() && pp.is_empty() {
        return None;
    }
    // keys: full triangles (row-major), then the polygon rings
    let mut keys: Vec<i64> = Vec::with_capacity(3 * ft.len());
    for &t in &ft {
        keys.extend_from_slice(&p.tris[3 * t..3 * t + 3]);
    }
    let nfull = keys.len();
    let (mut pcnt, mut poff, mut pxy, mut ptri) = (Vec::new(), vec![0usize], Vec::new(), Vec::new());
    for &q in &pp {
        let cnt = p.p_cnt[q] as usize;
        pcnt.push(cnt);
        for i in 0..cnt {
            keys.push(p.p_keys[q * p.m + i]);
            pxy.push((p.p_xy[(q * p.m + i) * 2], p.p_xy[(q * p.m + i) * 2 + 1]));
            ptri.push(p.p_tri[q]);
        }
        poff.push(poff.last().unwrap() + cnt);
    }
    // sorted unique keys: grid vertices through a presence table, cut points sorted
    let nvg = p.nv as usize;
    let mut present = vec![false; nvg];
    let mut others: Vec<i64> = Vec::new();
    for &k in &keys {
        if k < p.nv { present[k as usize] = true } else { others.push(k) }
    }
    let mut lut = vec![u32::MAX; nvg];
    let mut uk: Vec<i64> = Vec::new();
    for (v, &pr) in present.iter().enumerate() {
        if pr {
            lut[v] = uk.len() as u32;
            uk.push(v as i64);
        }
    }
    drop(present);
    let ng0 = uk.len();
    others.sort_unstable();
    others.dedup();
    uk.extend_from_slice(&others);
    let nv = uk.len();
    let inv: Vec<usize> = keys
        .par_iter()
        .map(|&k| if k < p.nv { lut[k as usize] as usize } else { ng0 + others.binary_search(&k).unwrap() })
        .collect();
    let (inv_full, inv_p) = inv.split_at(nfull);
    let mut first = vec![usize::MAX; nv];
    for (i, &u) in inv_p.iter().enumerate() {
        if first[u] == usize::MAX {
            first[u] = i;
        }
    }
    let pos: Vec<(f64, f64)> = (0..nv).map(|u| if u < ng0 { p.vxy(uk[u]) } else { pxy[first[u]] }).collect();
    let values = |s: &Spec| -> Vec<f64> {
        (0..nv)
            .into_par_iter()
            .map(|u| {
                let v = match s.base {
                    None => s.scalar,
                    Some(b) => {
                        if u < ng0 { b[uk[u] as usize] } else { p.interp(b, ptri[first[u]], pos[u].0, pos[u].1) }
                    }
                };
                s.clip(v + s.offset)
            })
            .collect()
    };
    let mut topv = values(top);
    let botv = values(bottom);
    for u in 0..nv {
        topv[u] = np_max(topv[u], botv[u]);
    }
    // outline: edges used by one face only
    let nft = ft.len();
    let mut edges: Vec<(usize, usize)> = Vec::with_capacity(3 * nft + inv_p.len());
    for (a, b) in [(0, 1), (1, 2), (2, 0)] {
        for t in 0..nft {
            edges.push((inv_full[3 * t + a], inv_full[3 * t + b]));
        }
    }
    for (q, &cnt) in pcnt.iter().enumerate() {
        let o = poff[q];
        for i in 0..cnt {
            edges.push((inv_p[o + i], inv_p[o + (i + 1) % cnt]));
        }
    }
    let nvu = nv as u64;
    let mut kk: Vec<(u64, u32)> = edges
        .par_iter()
        .enumerate()
        .map(|(i, &(a, b))| ((a.min(b) as u64) * nvu + a.max(b) as u64, i as u32))
        .collect();
    kk.par_sort_unstable();
    let mut single = vec![false; edges.len()];
    let mut i = 0;
    while i < kk.len() {
        let mut j = i + 1;
        while j < kk.len() && kk[j].0 == kk[i].0 {
            j += 1;
        }
        if j == i + 1 {
            single[kk[i].1 as usize] = true;
        }
        i = j;
    }
    drop(kk);
    let bnd: Vec<(usize, usize)> = edges.iter().zip(&single).filter(|(_, s)| **s).map(|(e, _)| *e).collect();
    drop(edges);
    let mut on_bnd = vec![false; nv];
    for &(a, b) in &bnd {
        on_bnd[a] = true;
        on_bnd[b] = true;
    }
    // zero thickness on the outline -> knife edge; walls thinner than 10 um lifted to 10 um
    let mut knife = vec![false; nv];
    for u in 0..nv {
        if on_bnd[u] {
            let d = topv[u] - botv[u];
            if d < 0.01 {
                if d < 1e-6 { knife[u] = true } else { topv[u] = botv[u] + 0.01 }
            }
        }
    }
    for u in 0..nv {
        if knife[u] {
            topv[u] = botv[u];
        }
    }
    let top_id: Vec<usize> = (0..nv).collect();
    let mut bot_id = vec![0usize; nv];
    let mut nb = 0;
    for u in 0..nv {
        if knife[u] {
            bot_id[u] = u;
        } else {
            bot_id[u] = nv + nb;
            nb += 1;
        }
    }
    let cells_t: Vec<i64> = ft.iter().map(|&t| (t / 2) as i64).collect();
    let mut full_cells = Vec::new();
    let mut i = 0;
    while i < cells_t.len() {
        if i + 1 < cells_t.len() && cells_t[i + 1] == cells_t[i] {
            full_cells.push(cells_t[i]);
            i += 2;
        } else {
            i += 1;
        }
    }
    let pkeys = &keys[nfull..];
    let ctx = Ctx {
        p, uk: &uk, lut: &lut, inv_full, inv_p, pcnt: &pcnt, poff: &poff, cells_t: &cells_t, ft: &ft,
        full_cells: &full_cells, pkeys, bnd: &bnd, win, simplify,
    };
    let mut faces = ctx.surface(&topv, &top_id, false);
    faces.extend(ctx.surface(&botv, &bot_id, true));
    // walls on the outline: top edge u->w  =>  (w_t, u_t, u_b), (w_t, u_b, w_b)
    for &(u, w) in &bnd {
        faces.push([top_id[w], top_id[u], bot_id[u]]);
    }
    for &(u, w) in &bnd {
        faces.push([top_id[w], bot_id[u], bot_id[w]]);
    }
    faces.retain(|t| t[0] != t[1] && t[1] != t[2] && t[0] != t[2]);
    // vertices: tops, then bottoms of the non-knife vertices; keep the used ones
    let total = nv + nb;
    let mut used = vec![false; total];
    for t in &faces {
        for &v in t {
            used[v] = true;
        }
    }
    let mut remap = vec![0i32; total];
    let mut n_used = 0i32;
    for v in 0..total {
        if used[v] {
            remap[v] = n_used;
            n_used += 1;
        }
    }
    let mut verts: Vec<f32> = Vec::with_capacity(3 * n_used as usize);
    let mut push = |x: f64, y: f64, z: f64| {
        verts.push((x * scale + ox) as f32);
        verts.push((y * scale + oy) as f32);
        verts.push(z as f32);
    };
    for u in 0..nv {
        if used[u] {
            push(pos[u].0, pos[u].1, topv[u]);
        }
    }
    let mut k = nv;
    for u in 0..nv {
        if !knife[u] {
            if used[k] {
                push(pos[u].0, pos[u].1, botv[u]);
            }
            k += 1;
        }
    }
    let tris: Vec<i32> = faces.iter().flat_map(|t| [remap[t[0]], remap[t[1]], remap[t[2]]]).collect();
    Some((verts, tris))
}
