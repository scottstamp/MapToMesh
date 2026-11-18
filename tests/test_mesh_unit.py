"""Mesh / raster unit checks: python tests/test_mesh_unit.py (no network)."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from maptomesh.mesh import heightfield_solid, check_manifold
from maptomesh.raster import fix_pinches, fix_mask_pinches, fill_polygon, rasterize_lines, fill_nan
from scipy import ndimage
rng = np.random.default_rng(1)
for trial in range(6):
    R, C = rng.integers(5, 60, 2)
    m = ndimage.gaussian_filter(rng.random((R, C)), 1.5) > 0.5
    m = fix_mask_pinches(m)
    if not m.any(): continue
    # flat
    me = heightfield_solid(m, 2.0, 0.5, 0.3, 0.3)
    st = check_manifold(me)
    exp = m.sum()*0.09*1.5
    assert st['open_edges']==0 and st['dup_directed_edges']==0 and st['degenerate']==0, st
    assert abs(st['volume']-exp) < 1e-3*exp, (st, exp)
    # varying top
    T = 1 + rng.random((R+1, C+1))*3
    me = heightfield_solid(m, T, 0.0, 0.3, 0.3)
    st = check_manifold(me)
    assert st['open_edges']==0 and st['dup_directed_edges']==0 and st['degenerate']==0, st
    assert st['volume'] > 0
    print(trial, R, C, st)
# labels pinch fix
L = rng.integers(0, 4, (80, 90)); L[:3] = -1
L2 = fix_pinches(L, protect=-1)
for k in range(4):
    mm = L2 == k
    A,B,Cc,D = mm[:-1,:-1], mm[:-1,1:], mm[1:,:-1], mm[1:,1:]
    print('pinches label', k, int((A&D&~B&~Cc).sum() + (B&Cc&~A&~D).sum()), 'changed', int((L2!=L).sum()))
# big perf
m = np.ones((900, 900), bool); m[:, :50] = False
T = rng.random((901, 901))
t=time.time(); me = heightfield_solid(m, T, 0.0, 0.25, 0.25); print('900^2', time.time()-t, check_manifold(me)['open_edges'], len(me.triangles))
t=time.time(); me = heightfield_solid(m, 3.0, 0.0, 0.25, 0.25); print('flat', time.time()-t, len(me.triangles))
# polygon fill: square
sq = fill_polygon([np.array([[2.5,2.5],[7.5,2.5],[7.5,7.5],[2.5,7.5]])], (10,10)); print(sq.sum(), sq[3:8,3:8].all())
ln = rasterize_lines([(np.array([[0,5],[20,5]]), 1.5)], (12, 22)); print(ln.sum(axis=0)[:5])
a = rng.random((50,60)); a[10:30, 10:40] = np.nan; f = fill_nan(a); print(np.isnan(f).sum(), f[20,25])
