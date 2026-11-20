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
from maptomesh.geo import Grid  # noqa: E402
from maptomesh.options import Options  # noqa: E402
from maptomesh.partition import DROPPED, Partition, shape_field  # noqa: E402


def _elev(ground, area_m, **arrays):
    """An Elevation like a surface model without classes (DTM ground, DSM in ``unclass``)."""
    n = ground.shape[0]
    nanf = np.full(ground.shape, np.nan)
    a = {"ground": ground, "water": np.zeros(ground.shape, np.float32), "building": nanf.copy(),
         "bridge": nanf.copy(), "veg": nanf.copy(), "unclass": nanf.copy(), "bld_cls": np.zeros(ground.shape, bool),
         "veg_cls": np.zeros(ground.shape, bool), "density": np.ones(ground.shape, np.float32),
         "covered": np.ones(ground.shape, bool), "veg_known": np.ones(ground.shape, bool),
         "surface_only": np.zeros(ground.shape, bool)}
    a.update(arrays)
    return model.Elevation("dem", Grid(n - 1, n - 1, area_m / (n - 1), -area_m / 2, -area_m / 2), a, {})


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


def test_noise_not_trees():
    """Clusters of noise returns (a km above a lake, 400 m above woods) are neither trees nor
    buildings, and don't set the height range; a 300 m tower inside a mapped footprint in data
    that classifies buildings but not trees (its class-1 returns: trees) stays a building."""
    n, area = 201, 1000.0
    rng = np.random.default_rng(2)
    yy, xx = np.mgrid[0:n, 0:n]
    ground = np.full((n, n), 50.0)
    lake = xx > 140
    U = np.where((xx < 100) & (yy < 100), 70.0 + 3 * rng.random((n, n)), np.nan)     # 20 m woods
    U[150:170, 150:180] = 1150.0                                                    # over the lake
    U[20:35, 20:40] = 450.0                                                         # in the woods
    U[150:160, 40:50] = 350.0                                                       # tower (footprint)
    bld = np.zeros((n, n), bool)
    bld[150:160, 40:50] = True
    elev = _elev(ground, area, unclass=U, bld_cls=np.ones((n, n), bool))
    for mode in ("single", "landcover", "elevation"):
        o = Options(lat=45, lon=-75, width_m=area, height_m=area, grid_cells=n - 1, color_mode=mode,
                    surface="full", osm_roads="none")
        surf = model.compute_surface(elev, o, {"water": lake, "buildings": bld})
        veg = surf.masks[model.VEG]
        assert not (veg & lake).any()
        assert ((surf.canopy - surf.ground)[veg] < model.TREE_MAX_M).all()
        assert veg[60:90, 60:90].all()
        assert surf.masks[model.BUILDING][152:158, 42:48].all()
        assert abs(surf.roof[155, 45] - 350.0) < 1.0
        res = model.build_parts(elev, surf, o, lambda f, m: None)
        assert res.stats["elevation_range_m"][1] < 360.0, res.stats["elevation_range_m"]


def test_bridge_under_canopy():
    """An OSM bridge under tree canopy in a surface model without classes keeps its ends' height,
    not the canopy's."""
    n, area = 301, 600.0
    step = area / (n - 1)
    rng = np.random.default_rng(1)
    yy, xx = np.mgrid[0:n, 0:n] * step
    ground = np.full((n, n), 20.0)
    ground[np.abs(xx - 300) < 3] = 18.5                              # a 6 m stream
    woods = np.hypot(xx - 300, yy - 300) < 200
    U = np.where(woods, 34.0 + 3 * rng.random((n, n)), np.nan)        # 14-17 m canopy
    bridge = (np.abs(yy - 300) <= 4) & (np.abs(xx - 300) <= 15)       # 30 x 8 m
    o = Options(lat=51.5, lon=-0.1, width_m=area, height_m=area, grid_cells=n - 1, color_mode="landcover",
                surface="full", osm_roads="none")
    surf = model.compute_surface(_elev(ground, area, unclass=U), o, {"bridges": bridge, "buildings": np.zeros((n, n), bool)})
    d = (surf.bridge - surf.ground)[surf.bridge_mask]
    assert surf.bridge_mask.sum() > 50 and d.max() < 2.0, (d.min(), d.max())


def _strait(cells, mode, min_feature=1.0, width_m=12.0, area_m=2000.0, **kw):
    step = area_m / cells
    n = cells + 1
    yy, xx = np.mgrid[0:n, 0:n] * step
    ground = np.full((n, n), 80.0)
    lake = (xx > area_m * 0.25) & (xx < area_m * 0.75)
    ground[lake] = 74.0
    bridge = (np.abs(yy - area_m / 2) <= width_m / 2) & (xx > area_m * 0.2) & (xx < area_m * 0.8)
    U = np.where(bridge & lake, 84.0, np.nan)
    o = Options(lat=45, lon=-75, width_m=area_m, height_m=area_m, grid_cells=cells, color_mode=mode,
                surface="bridges", osm_roads="none", min_feature_mm=min_feature, **kw)
    elev = _elev(ground, area_m, unclass=U)
    surf = model.compute_surface(elev, o, {"water": lake, "bridges": bridge})
    res = model.build_parts(elev, surf, o, lambda f, m: None)
    span = surf.bridge_mask & lake
    cm = span[:-1, :-1] & span[1:, 1:]
    wet = (res.cell_rgb[cm] == np.array(model._hex_rgb(o.colors.water))).all(axis=1).mean()
    return cm.sum(), wet, res


def test_narrow_bridge_over_water():
    """A deck about the minimum printable width across open water survives the smoothed shore
    (the water was cut first and closed over it), in land cover and in elevation bands."""
    for mode, kw in (("landcover", {}), ("elevation", {"elevation_water": True})):
        n, wet, res = _strait(500, mode, min_feature=1.5, **kw)
        assert n > 100 and wet == 0.0, (mode, n, wet)
        names = [p.name for p in res.parts]
        assert any("ridge" in nm for nm in names), names


def test_skerry_kept():
    """A thin skerry the shore smoothing shrinks as a whole keeps its piece (it used to be deleted
    as a "fragment" of itself); a pier's specks still go."""
    shape = (60, 80)
    wet = np.ones(shape, bool)
    wet[:, :10] = False                       # coast
    wet[30, 10:60] = False                    # 1-cell pier with wider spots
    for c in range(16, 56, 8):
        wet[29:32, c:c + 3] = False
    for r, c in ((9, 50), (9, 51), (10, 49), (10, 50), (10, 51), (11, 48), (11, 49), (12, 48), (12, 49)):
        wet[r, c] = False                     # a 9-cell skerry, one or two cells wide
    land = model._water_indicator(wet, np.zeros(shape, bool), 4, 1.2) < 0.5
    assert land[7:15, 46:54].any()
    assert not land[28:33, 16:60].any()


def test_single_colour_nothing_to_cut():
    """Open ocean (flat, all water), single colour, no frame: no partition steps at all."""
    n = 51
    elev = _elev(np.zeros((n, n)), 500.0, water=np.ones((n, n), np.float32), density=np.zeros((n, n), np.float32),
                 veg_known=np.zeros((n, n), bool))
    for mode in ("single", "landcover", "elevation"):
        o = Options(lat=30, lon=-40, width_m=500, height_m=500, grid_cells=50, color_mode=mode, frame=False)
        res = model.build_parts(elev, model.compute_surface(elev, o, {}), o, lambda f, m: None)
        assert res.stats["triangles"] > 0 and res.cell_rgb.shape == (50, 50, 3)


if __name__ == "__main__":
    for fn in (test_label_reduce, test_partition_copy_and_cells, test_partition_step_sides,
               test_shore_cap_identical, test_noise_not_trees, test_bridge_under_canopy,
               test_narrow_bridge_over_water, test_skerry_kept, test_single_colour_nothing_to_cut):
        fn()
        print(f"{fn.__name__}: ok")
    print("all passed")
