"""Generation pipeline: elevation -> surface -> colour regions -> mesh parts."""

from __future__ import annotations

import copy
import hashlib
import json
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy import ndimage

from . import config
from .geo import AOI, Grid
from .mesh import Mesh
from .options import Options
from .partition import DROPPED, FieldSpec, Partition, shape_field
from .raster import despike, fill_nan, remove_small, smooth

ProgressFn = Callable[[float, str], None]

# region labels
FRAME, GROUND, VEG, WATER, ROAD, BUILDING = 0, 1, 2, 3, 4, 5
RIVER = 7  # rivers / streams: water colour, but sloped (lakes and sea are flat WATER)
LABEL_NAMES = {GROUND: "ground", VEG: "vegetation", WATER: "water", ROAD: "roads", BUILDING: "buildings"}


# ---------------------------------------------------------------- elevation (cached)

@dataclass
class Elevation:
    source: str
    grid: Grid
    arrays: dict[str, np.ndarray]
    info: dict


_MEM: "OrderedDict[str, Elevation]" = OrderedDict()
_MEM_LOCK = threading.Lock()


def _cache_path(key: str):
    return config.ELEV_CACHE_DIR / (hashlib.sha1(key.encode()).hexdigest()[:20] + ".npz")


def get_elevation(opts: Options, progress: ProgressFn, cancel: threading.Event) -> Elevation:
    key = opts.elevation_key()
    with _MEM_LOCK:
        if key in _MEM:
            _MEM.move_to_end(key)
            progress(1.0, "Elevation data (cached)")
            return _MEM[key]
    aoi = AOI(opts.lat, opts.lon, opts.width_m, opts.height_m)
    grid = Grid.for_aoi(aoi, min(opts.grid_cells, config.MAX_GRID_CELLS))
    path = _cache_path(key)
    if path.exists():
        z = np.load(path, allow_pickle=False)
        arrays = {k: z[k] for k in z.files if k != "info"}
        info = json.loads(bytes(z["info"]).decode())
        elev = Elevation(opts.source, grid, arrays, info)
        progress(1.0, "Elevation data (disk cache)")
    else:
        if opts.source == "lidar":
            from .lidar_index import INDEX
            from .pointcloud import fetch_lidar
            r = fetch_lidar(aoi, grid, INDEX.get(), progress, cancel, max_tiles=config.MAX_TILES)
            arrays = {"ground": r.ground, "water": r.water_frac, "building": r.building, "bridge": r.bridge,
                      "veg": r.veg, "unclass": r.unclass, "bld_cls": r.bld_classified, "veg_cls": r.veg_classified,
                      "density": r.density.astype(np.float32), "covered": r.covered}
            info = r.info
        else:
            from .hrdem import fetch_hrdem
            r = fetch_hrdem(aoi, grid, progress, cancel)
            arrays = {"dtm": r["dtm"], "dsm": r["dsm"]}
            info = r["info"]
        info["grid"] = {"nx": grid.nx, "ny": grid.ny, "step_m": grid.step}
        np.savez_compressed(path, info=np.frombuffer(json.dumps(info).encode(), dtype=np.uint8), **arrays)
        elev = Elevation(opts.source, grid, arrays, info)
    with _MEM_LOCK:
        _MEM[key] = elev
        while len(_MEM) > 4:
            _MEM.popitem(last=False)
    return elev


# ---------------------------------------------------------------- surface

@dataclass
class Surface:
    ground: np.ndarray        # bare earth; water bodies set to their level (m)
    surface: np.ndarray       # printed surface for single-colour / elevation-band models (m)
    roof: np.ndarray          # building roof field, filled everywhere (m)
    canopy: np.ndarray        # tree canopy field, filled everywhere (m)
    bridge: np.ndarray        # bridge deck field, filled everywhere (m)
    level: np.ndarray         # water level, extended beyond the water by nearest value (m)
    masks: dict[int, np.ndarray]  # vertex masks for BUILDING / ROAD / WATER / VEG
    bridge_mask: np.ndarray
    ground_land: np.ndarray = None   # ground with land extrapolated over the water (shore walls)
    surface_land: np.ndarray = None  # same for the printed surface
    river: np.ndarray = None         # OSM river / stream centreline corridors (never dropped)
    notes: list[str] = field(default_factory=list)


def _close_and_fill(values: np.ndarray, mask: np.ndarray, min_cells: int) -> tuple[np.ndarray, np.ndarray]:
    m = ndimage.binary_closing(mask, structure=np.ones((3, 3)), iterations=1, border_value=0) | mask
    m = remove_small(m, min_cells)
    v = fill_nan(np.where(mask & np.isfinite(values), values, np.nan)) if m.any() else values
    return m, v


def _water_level_map(G: np.ndarray, W: np.ndarray, hint: np.ndarray | None, cov: np.ndarray,
                     step: float) -> np.ndarray:
    """One water level per water body.

    Lakes and sea get a single constant level (median of LiDAR water returns,
    or a low percentile of the ground surface when there are none), so they are
    perfectly flat regardless of wave / scan noise. Only bodies whose measured
    water surface genuinely spans more than 3 m (rivers with a gradient) keep a
    smoothed profile, and even there the near-level stretches are made exactly
    flat.
    """
    L = np.full(G.shape, np.nan)
    lab, n = ndimage.label(W, structure=np.ones((3, 3)))
    if n == 0:
        return L
    sig = max(2.0, 40.0 / step)
    for k, sl in enumerate(ndimage.find_objects(lab), start=1):
        if sl is None:
            continue
        comp = lab[sl] == k
        g = G[sl]
        h = comp & hint[sl] if hint is not None else np.zeros_like(comp)
        hv = g[h]
        meas = hv if hv.size >= 20 else g[comp & cov[sl]]
        if meas.size == 0:
            meas = g[comp]
        spread = float(np.percentile(meas, 90) - np.percentile(meas, 10))
        if hv.size >= 20 and spread > 3.0:
            surf = ndimage.gaussian_filter(fill_nan(np.where(h, g, np.nan)), sig, mode="nearest")
            gy, gx = np.gradient(surf, step)
            flat = comp & (np.hypot(gx, gy) < 0.003)
            sub, m = ndimage.label(flat, structure=np.ones((3, 3)))
            if m:
                med = np.asarray(ndimage.median(surf, sub, np.arange(1, m + 1)), float)
                surf = np.where(sub > 0, np.concatenate([[0.0], med])[sub], surf)
            Lc = surf
        else:
            lvl = float(np.median(hv)) if hv.size >= 20 else float(np.percentile(meas, 10))
            Lc = np.full(g.shape, lvl)
        view = L[sl]
        view[comp] = Lc[comp]
    return L


def _smooth_steep(G: np.ndarray, step: float, strength: float) -> np.ndarray:
    """Despike and smooth only where the terrain is steep (cliffs, bluffs).

    LiDAR ground on near-vertical faces is sparse and noisy, which prints as jagged,
    striated cliffs. Weight ramps from 0 at ~19 deg to 1 at 45 deg (feathered so the
    cliff top and foot blend too); flat and gently sloping terrain is left untouched.
    """
    if strength <= 0:
        return G
    base = ndimage.gaussian_filter(G, 1.0, mode="nearest")
    gy, gx = np.gradient(base, step)
    w = np.clip((np.hypot(gx, gy) - 0.35) / 0.65, 0, 1)
    w = np.clip(ndimage.gaussian_filter(w, 1.5, mode="nearest") * 1.5, 0, 1)
    med = ndimage.median_filter(G, size=3, mode="nearest")
    G1 = np.where((w > 0.1) & (np.abs(G - med) > 2.0), med, G)
    S = ndimage.gaussian_filter(G1, strength, mode="nearest")
    return G1 + w * (S - G1)


def _planar_components(U: np.ndarray, mask: np.ndarray, step: float) -> np.ndarray:
    """Connected raised regions whose surface is locally planar (roofs, incl.
    pitched) rather than rough (tree crowns)."""
    lab, n = ndimage.label(mask)
    if n == 0:
        return np.zeros_like(mask)
    idx = np.arange(1, n + 1)
    lap = np.abs(U - ndimage.uniform_filter(U, size=3, mode="nearest"))
    rough = np.asarray(ndimage.mean(lap, lab, idx))
    size = np.asarray(ndimage.sum_labels(np.ones_like(U), lab, idx))
    ok = (rough < max(0.35, 0.12 * step)) & (size * step * step >= 25)
    return np.concatenate([[False], ok])[lab]


def compute_surface(elev: Elevation, opts: Options, osm: dict[str, np.ndarray] | None = None) -> Surface:
    g = elev.grid
    a = elev.arrays
    step = g.step
    notes: list[str] = []
    cells = lambda m2: max(1, int(round(m2 / (step * step))))  # noqa: E731
    shape = g.vshape
    zeros = np.zeros(shape, bool)
    osm = osm or {}
    osm_water = osm.get("water") if opts.osm_water else None
    osm_roads = osm.get("roads")
    osm_bld = osm.get("buildings")
    hint = None

    if elev.source == "lidar":
        G = a["ground"]
        cov = a["covered"].astype(bool)
        dens = a["density"]
        if cov.mean() < 0.999:
            notes.append(f"{100 * (1 - cov.mean()):.1f}% of the area has no LiDAR coverage (interpolated).")
        W = remove_small(a["water"] >= 0.3, cells(30))
        if opts.voids_as_water:
            k = max(3, int(np.ceil(4.0 / step)) | 1)
            void = cov & (ndimage.uniform_filter(dens.astype(np.float32), size=k) == 0)
            W |= remove_small(void, cells(600))
        if osm_water is not None:
            W |= osm_water
        hint = a["water"] >= 0.5
        Gf = _smooth_steep(smooth(fill_nan(G), opts.smoothing), step, opts.cliff_smoothing)
        measured = np.where(np.isfinite(G) & (a["water"] < 0.3), G, np.nan)

        def above(v, d):
            return np.isfinite(v) & (np.nan_to_num(v, nan=-1e9) > Gf + d)

        bld_v, veg_v = a["building"].copy(), a["veg"].copy()
        # route unclassified (0/1) returns depending on what the owning project classifies
        U = a["unclass"]
        bcls, vcls = a["bld_cls"].astype(bool), a["veg_cls"].astype(bool)
        Uf = despike(np.where(np.isfinite(U), U, Gf), np.isfinite(U), thresh=12.0, size=5)
        um = above(Uf, 2.0) & np.isfinite(U)
        to_veg = um & bcls & ~vcls        # buildings classified -> class 1 is mostly trees
        maybe_bld = um & ~bcls & vcls     # trees classified -> class 1 is buildings + clutter
        unknown = um & ~bcls & ~vcls      # nothing classified
        cand = maybe_bld | unknown
        if cand.any():
            if osm_bld is not None and osm_bld.any():
                near = ndimage.binary_dilation(osm_bld, iterations=max(1, int(round(2.0 / step))))
                is_b = cand & near
                notes.append("Some LiDAR here has no building class; OpenStreetMap footprints identified buildings.")
            else:
                is_b = _planar_components(Uf, cand, step) & cand
                notes.append("Some LiDAR here has no building class; a roof-flatness heuristic was used "
                             "(OpenStreetMap footprints give better results).")
            bld_v = np.where(is_b, np.fmax(bld_v, Uf), bld_v)
            to_veg |= unknown & ~is_b
        veg_v = np.where(to_veg, np.fmax(veg_v, Uf), veg_v)
        Bm, Bv = _close_and_fill(bld_v, above(bld_v, 2.0), cells(12))
        BRm, BRv = _close_and_fill(a["bridge"], above(a["bridge"], 1.0), cells(20))
        vfin = np.isfinite(veg_v)
        V0 = despike(np.where(vfin, veg_v, Gf), vfin, thresh=10.0, size=5)
        vmask = vfin & (V0 > Gf + 1.5) & ~Bm
        Vm, Vv = _close_and_fill(V0, vmask, cells(25))
        Vm &= ~Bm
    else:
        dtm, dsm = a["dtm"], a["dsm"]
        cov = np.isfinite(dtm)
        if cov.mean() < 0.999:
            notes.append(f"{100 * (1 - cov.mean()):.1f}% of the area has no HRDEM coverage (interpolated).")
        Gf = _smooth_steep(smooth(fill_nan(dtm), opts.smoothing), step, opts.cliff_smoothing)
        measured = None
        Df = fill_nan(np.where(np.isfinite(dsm), dsm, dtm))
        W = osm_water.copy() if osm_water is not None else zeros.copy()
        raised = remove_small((Df - Gf > 2.5) & ~W, cells(12))
        if osm_bld is not None and osm_bld.any():
            near = ndimage.binary_dilation(osm_bld, iterations=max(1, int(round(2.0 / step))))
            Bm = raised & near
            Vm = remove_small(raised & ~Bm, cells(25))
            notes.append("HRDEM has no classification: buildings come from OpenStreetMap footprints, "
                         "other raised objects are treated as trees.")
        else:
            Bm, Vm = raised, zeros
            notes.append("HRDEM has no classification: all raised objects use the building colour.")
        Bv = Vv = Df
        BRm, BRv = zeros, Gf

    # --- water: clean the mask, one flat level per water body
    W = remove_small(W, cells(60))
    W |= ~W & ~remove_small(~W, cells(40))  # fill pin-holes (wave dropouts, boats)
    if opts.surface != "terrain":
        W &= ~BRm
    # rivers / streams: OSM centrelines (continuous, printable width) - the LiDAR only sees
    # scattered water returns on narrow streams, which printed as dashed colour islands
    river = np.zeros_like(W)
    osm_ways = osm.get("waterways")
    if osm_ways is not None and osm_ways.any():
        river = osm_ways & ~W
        if osm_roads is not None:
            river &= ~osm_roads           # roads cross over streams (bridges / culverts)
        if osm_bld is not None:
            river &= ~osm_bld
    if river.any() and W.any():
        # LiDAR water returns along a stream belong to the stream, not to separate ponds: absorb
        # water patches touching the corridor that are narrow (widest inscribed circle < ~15 m)
        # or small. Lakes and the sea stay flat bodies of their own.
        lab, n = ndimage.label(W, structure=np.ones((3, 3)))
        touch = np.unique(lab[ndimage.binary_dilation(river, iterations=2) & (lab > 0)])
        if len(touch):
            radius = np.asarray(ndimage.maximum(ndimage.distance_transform_edt(W), lab, touch))
            size = np.bincount(lab.ravel(), minlength=n + 1)[touch]
            take = touch[(radius * step <= max(15.0, 3 * step)) | (size < cells(3000))]
            if len(take):
                absorb = np.isin(lab, take)
                river |= absorb
                W &= ~absorb
    if W.any():
        if opts.flatten_water:
            L = _water_level_map(Gf, W, hint, cov, step)
            if measured is not None:
                # OSM coastlines / voids that overlap land LiDAR measured well above the water
                # would otherwise pull cliff tops down to sea level (notched cliff edges)
                land = W & np.isfinite(measured) & (np.nan_to_num(measured, nan=-1e9) > L + 3.0)
                if land.any():
                    W &= ~land
                    W = remove_small(W, cells(60))
                    L = _water_level_map(Gf, W, hint, cov, step)
            Gf = np.where(W, L, Gf)
        else:
            L = np.where(W, Gf, np.nan)
    else:
        L = np.full(Gf.shape, np.nan)
    if river.any():
        # river surface: channel bottom near the centreline, smoothed along the river; where a
        # river touches a lake / the sea it takes that body's level (no step at the mouth)
        chan = ndimage.minimum_filter(Gf, size=3, mode="nearest")
        sig = max(1.5, 15.0 / step)
        num = ndimage.gaussian_filter(np.where(river, chan, 0.0), sig, mode="nearest")
        den = ndimage.gaussian_filter(river.astype(np.float64), sig, mode="nearest")
        Lr = np.minimum(np.divide(num, den, out=chan.copy(), where=den > 1e-6), Gf)
        if W.any():
            near_body = ndimage.binary_dilation(W, iterations=2) & river
            _, (bi, bj) = ndimage.distance_transform_edt(~W, return_indices=True)
            Lr = np.where(near_body, L[bi, bj], Lr)
        Gf = np.where(river, Lr, Gf)
        L = np.where(river, Lr, L)
    W_area = W.copy()
    W = W | river
    if W.any():
        _, (ri, ci) = ndimage.distance_transform_edt(~W, return_indices=True)
        level = L[ri, ci]
    else:
        level = Gf.copy()

    S = Gf.copy()
    if opts.surface in ("buildings", "full"):
        S = np.where(Bm, np.maximum(S, Bv), S)
        S = np.where(BRm, np.maximum(S, BRv), S)
    if opts.surface == "full":
        S = np.where(Vm, np.maximum(S, Vv), S)

    roads = osm_roads.copy() if osm_roads is not None else zeros.copy()
    if opts.surface != "terrain":
        roads |= BRm
    masks = {
        BUILDING: Bm if opts.surface != "terrain" else zeros,
        ROAD: roads,
        WATER: W_area,
        RIVER: river,
        VEG: Vm,
    }
    roof = np.maximum(np.where(np.isfinite(Bv), Bv, Gf), Gf)
    canopy = np.maximum(np.where(np.isfinite(Vv), Vv, Gf), Gf)
    bridge = np.maximum(np.where(np.isfinite(BRv), BRv, Gf), Gf)
    # Land surface continued over the water: where a land part meets the (separate, flat) water
    # part, the shore becomes one clean vertical wall along the smooth outline instead of the
    # land dipping to the water level at every water vertex (cell staircase at cliff feet).
    if W.any():
        # nearest land value: keeps shore walls at the real cliff-foot height (a smooth fill
        # would average in higher inland ground and make the walls too tall)
        _, (ri, ci) = ndimage.distance_transform_edt(W, return_indices=True)
        ext = Gf[ri, ci]
        # soften steps where the nearest land vertex switches along the shore (water side only)
        ground_land = np.where(W, np.maximum(ndimage.gaussian_filter(ext, 1.0, mode="nearest"), level), Gf)
        surface_land = np.where(W, ground_land, S)
    else:
        ground_land, surface_land = Gf, S
    return Surface(Gf, S, roof, canopy, bridge, level, masks,
                   BRm if opts.surface != "terrain" else zeros, ground_land, surface_land, river, notes)


# ---------------------------------------------------------------- parts

MODEL = 6
BAND0 = 10


@dataclass
class Part:
    name: str
    color: str
    mesh: Mesh
    tile: str
    category: str


@dataclass
class BuildResult:
    parts: list[Part]
    stats: dict
    filaments: list[dict]
    color_changes: list[dict]
    notes: list[str]
    cell_rgb: np.ndarray | None = None   # (R, C, 3) uint8 colour per cell, for thumbnails
    base_parts: list | None = None       # base plate printed separately (own file)
    z_mm: np.ndarray | None = None       # (R+1, C+1) top heights


def _hex_rgb(c: str) -> tuple[int, int, int]:
    return int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)


def _indicator(mask: np.ndarray, min_cells: int, sigma: float = 0.7) -> np.ndarray:
    """Smooth 0..1 field whose 0.5 iso-line is a rounded version of the mask
    outline; specks and pin-holes smaller than ``min_cells`` are removed."""
    m = remove_small(mask, min_cells)
    m |= ~m & ~remove_small(~m, min_cells)
    f = m.astype(np.float64)
    return ndimage.gaussian_filter(f, sigma, mode="nearest") if sigma > 0 else f


def _cut_water(part: Partition, lakes: np.ndarray, rivers: np.ndarray, min_cells: int,
               land: Partition | None = None) -> None:
    """One outline for all water (label WATER), then the river share of it relabelled RIVER."""
    wet = lakes | rivers
    if not wet.any():
        return
    ind = _indicator(wet, min_cells)
    part.step(ind, 0.5, above=WATER)
    if land is not None:
        land.step(ind, 0.5, above=WATER)
    if rivers.any():
        if lakes.any():
            split = (ndimage.gaussian_filter(rivers.astype(np.float64), 1.0, mode="nearest")
                     - ndimage.gaussian_filter(lakes.astype(np.float64), 1.0, mode="nearest"))
            part.step(split, 0.0123, above=RIVER, only=WATER)
        else:
            part.step(np.ones(wet.shape), 0.0, above=RIVER, only=WATER)


def _drop_swap_patches(masks: dict[int, np.ndarray], tops: dict[int, np.ndarray], ground: np.ndarray,
                       depth: float, cell_mm: float, budget_mm2: float, max_patch_mm2: float = 20.0,
                       layer: float = 0.2, keep: np.ndarray | None = None
                       ) -> tuple[dict[int, np.ndarray], int, float, int]:
    """Recolour (to ground) small patches that are the *only* reason their colour is needed in
    some layers - e.g. a lone pond or hilltop house at an elevation where nothing else of that
    colour exists. Greedy by layers saved per mm2, within an area budget.
    Returns (masks, patches removed, area removed, layers saved)."""
    comps = []  # (label, component id, area, lo layer, hi layer)
    labels = {}
    nl = int(np.ceil(max(float(t.max()) for t in tops.values()) / layer)) + 2
    for lab, m in masks.items():
        if not m.any():
            continue
        lm, n = ndimage.label(m, structure=np.ones((3, 3)))
        labels[lab] = lm
        idx = np.arange(1, n + 1)
        area = np.bincount(lm.ravel(), minlength=n + 1)[1:] * cell_mm * cell_mm
        hi = np.ceil(np.asarray(ndimage.maximum(tops[lab], lm, idx)) / layer).astype(int)
        base_f = tops[lab] if lab in (WATER, RIVER) else ground
        lo = np.floor((np.asarray(ndimage.minimum(base_f, lm, idx)) - depth) / layer).astype(int)
        kept = set(np.unique(lm[keep & (lm > 0)]).tolist()) if keep is not None else set()
        for k in range(n):
            if k + 1 in kept:
                continue  # e.g. river segments: continuity matters more than one swap
            comps.append((lab, k + 1, float(area[k]), max(lo[k], 0), min(hi[k], nl)))
    if not comps:
        return masks, 0, 0.0, 0
    lab_a = np.array([c[0] for c in comps]); area_a = np.array([c[2] for c in comps])
    lo_a = np.array([c[3] for c in comps]); hi_a = np.array([c[4] for c in comps])
    active = np.ones(len(comps), bool)
    lost, saved = 0.0, 0
    for _ in range(500):
        best, best_score, best_uniq = -1, 0.0, 0
        for lab in np.unique(lab_a):
            sel = np.nonzero((lab_a == lab) & active)[0]
            cov = np.zeros(nl + 1, int)
            np.add.at(cov, lo_a[sel], 1)
            np.add.at(cov, hi_a[sel], -1)
            cov = np.cumsum(cov)[:nl]
            pre = np.concatenate([[0], np.cumsum(cov == 1)])
            cand = sel[(area_a[sel] < max_patch_mm2) & (lost + area_a[sel] <= budget_mm2)]
            if not len(cand):
                continue
            uniq = pre[hi_a[cand]] - pre[lo_a[cand]]
            score = uniq / np.maximum(area_a[cand], 1e-6)
            j = int(np.argmax(score))
            if uniq[j] > 0 and score[j] > best_score:
                best, best_score, best_uniq = int(cand[j]), float(score[j]), int(uniq[j])
        if best < 0:
            break
        active[best] = False
        lost += area_a[best]
        saved += best_uniq
    out = {lab: m.copy() for lab, m in masks.items()}
    removed = np.nonzero(~active)[0]
    for i in removed:
        lab, cid = comps[i][0], comps[i][1]
        out[lab][labels[lab] == cid] = False
    return out, int(len(removed)), float(lost), int(saved)


def _layer_color_stats(parts: list, zmax: float, layer: float = 0.2) -> tuple[int, int]:
    """Estimate filament swaps: per 0.2 mm layer, the number of distinct colours whose solids
    cross that layer (a horizontal plane through a closed solid always cuts its surface)."""
    nl = int(np.ceil(zmax / layer)) + 1
    per_color: dict[str, np.ndarray] = {}
    for p in parts:
        z = p.mesh.vertices[p.mesh.triangles][:, :, 2]
        lo = np.floor(z.min(axis=1) / layer - 0.5).astype(np.int64) + 1  # first layer centre above zmin
        hi = np.ceil(z.max(axis=1) / layer - 0.5).astype(np.int64) - 1   # last layer centre below zmax
        ok = hi >= lo
        diff = per_color.setdefault(p.color, np.zeros(nl + 2, np.int64))
        np.add.at(diff, np.clip(lo[ok], 0, nl), 1)
        np.add.at(diff, np.clip(hi[ok] + 1, 0, nl + 1), -1)
    counts = sum((np.cumsum(dd)[:nl] > 0).astype(np.int64) for dd in per_color.values())
    return int(np.maximum(counts - 1, 0).sum()), int(counts.max())


def _terraced(top: np.ndarray, depth: float, base: float, q: float = 0.5) -> np.ndarray:
    """Colour-skin interface: ``top - depth`` rounded down to ``q`` mm terraces (flat where the
    surface is flat, so it triangulates cheaply), kept between the base and just under ``top``."""
    t = base + np.floor((top - depth - base) / q) * q
    return np.round(np.minimum(np.maximum(t, base + 0.2), top - 0.1), 3)


def _flat_interface(mask: np.ndarray, top: np.ndarray, depth: float, floor: float) -> np.ndarray:
    """Per connected (dilated) region: min(top) - depth, at least ``floor``."""
    md = ndimage.binary_dilation(mask, iterations=3)
    lab, n = ndimage.label(md, structure=np.ones((3, 3)))
    out = np.maximum(top - depth, floor)
    if n:
        mins = np.asarray(ndimage.minimum(top, lab, np.arange(1, n + 1)), float)
        lvl = np.maximum(np.concatenate([[0.0], mins])[lab] - depth, floor)
        out = np.where(lab > 0, lvl, out)
    return np.round(out, 3)


def build_parts(elev: Elevation, surf: Surface, opts: Options, progress: ProgressFn) -> BuildResult:
    g = elev.grid
    nx, ny = g.nx, g.ny
    cell_mm = opts.size_mm / max(nx, ny)
    mm_per_m = cell_mm / g.step
    zscale = mm_per_m * opts.z_exaggeration
    notes = list(surf.notes)
    col = opts.colors
    base = float(opts.base_mm)

    # --- padded vertex grid (room for the frame ring) and the shape outline field
    frame_cells = opts.frame_width_mm / cell_mm if opts.frame else 0.0
    pad = int(np.ceil(frame_cells)) + 2 if opts.frame else 0
    R, C = ny + 2 * pad, nx + 2 * pad

    def P(a: np.ndarray) -> np.ndarray:
        return np.pad(a, pad, mode="edge") if pad else a

    corner_cells = opts.corner_radius_mm / cell_mm
    f_model = shape_field(opts.shape, R, C, nx, ny, corner_cells)
    inside_v = f_model >= 0
    Sm = P(surf.surface)
    hmin, hmax = float(Sm[inside_v].min()), float(Sm[inside_v].max())
    lowest = base + min(0.3, opts.floor_mm)
    zcap = base + opts.max_height_mm if opts.max_height_mm else np.inf

    def zmm(h: np.ndarray) -> np.ndarray:
        return np.minimum((P(h) - hmin) * zscale + base + opts.floor_mm, zcap)

    S_mm = np.round(zmm(surf.surface), 3)

    # --- triangulation follows ridges of the printed surface
    a00, a01, a10, a11 = S_mm[:-1, :-1], S_mm[:-1, 1:], S_mm[1:, :-1], S_mm[1:, 1:]
    part = Partition(R, C, np.abs(a00 - a11) <= np.abs(a01 - a10))
    progress(0.05, "Cutting outline")
    if opts.frame:
        part.step(shape_field(opts.shape, R, C, nx, ny, corner_cells, grow=frame_cells), 0.0, below=DROPPED)
        part.step(f_model, 0.0, below=FRAME)
    elif opts.shape != "rectangle":
        part.step(f_model, 0.0, below=DROPPED)
    # the base plate spans every colour region: build it from the outline cuts only, so later
    # colour cuts cannot leave T-junctions inside it
    outline = copy.deepcopy(part)
    outline.finish(GROUND)
    land = outline

    defs: list[tuple[str, str, str, set, FieldSpec, FieldSpec]] = []
    color_changes: list[dict] = []
    flat = lambda a: np.ascontiguousarray(a, dtype=np.float64).ravel()  # noqa: E731
    min_cells = max(1, int(round(opts.min_region_mm2 / (cell_mm * cell_mm))))

    if opts.color_mode == "single":
        part.finish(MODEL)
        defs.append(("model", "Model", col.single, {MODEL}, FieldSpec(flat(S_mm)), FieldSpec(0.0)))
        default, label_color = MODEL, {MODEL: col.single}
    elif opts.color_mode == "landcover":
        G_mm = np.round(zmm(surf.ground_land), 3)
        fields = {
            GROUND: G_mm,
            BUILDING: np.maximum(np.round(zmm(surf.roof), 3), G_mm),
            VEG: np.maximum(np.round(zmm(surf.canopy), 3), G_mm) if opts.surface == "full" else G_mm,
            WATER: np.maximum(np.round(zmm(surf.level), 3) - opts.water_recess_mm, lowest),
        }
        bridge_mm = np.round(zmm(surf.bridge), 3)
        road = np.where(P(surf.bridge_mask), np.maximum(bridge_mm, G_mm), G_mm) + opts.road_emboss_mm
        fields[ROAD] = np.maximum(road, lowest)
        fields[RIVER] = fields[WATER]
        names = {GROUND: "Ground", BUILDING: "Buildings", WATER: "Water", ROAD: "Roads", VEG: "Trees",
                 RIVER: "Rivers"}
        label_color = {GROUND: col.ground, BUILDING: col.buildings, WATER: col.water, RIVER: col.water,
                       ROAD: col.roads,
                       VEG: col.vegetation}
        cmasks = {lab: P(surf.masks[lab]) for lab in (BUILDING, ROAD, WATER, RIVER, VEG)}
        if opts.color_depth_mm > 0 and opts.swap_patch_budget_mm2 > 0:
            cmasks, n_rm, a_rm, saved = _drop_swap_patches(
                cmasks, {lab: fields[lab] for lab in cmasks}, G_mm, opts.color_depth_mm, cell_mm,
                opts.swap_patch_budget_mm2, keep=P(surf.river) if surf.river is not None else None)
            if n_rm:
                notes.append(f"Recoloured {n_rm} small patch(es) ({a_rm:.0f} mm2) that each forced their colour "
                             f"into extra layers: ~{saved} fewer filament changes.")
        progress(0.1, "Cutting colour regions")
        # Water is cut first, so "land" is exactly the complement of the water outline. A second
        # partition with only the outline + water cut then gives one seamless ground-coloured body
        # under all land colours (no walls along every colour boundary). LiDAR bridges were
        # already removed from the water mask, so they survive as roads.
        land = copy.deepcopy(part)
        # all water (lakes, sea, rivers) is cut as ONE region, then divided into lake / river
        # inside it - separate outlines would leave slivers of land between a river and LiDAR
        # water patches or a lake it runs into
        _cut_water(part, cmasks[WATER], cmasks[RIVER], min_cells, land=land)
        for i, lab in enumerate((BUILDING, ROAD, VEG), start=2):  # priority order
            m = cmasks[lab]
            if m.any():
                # distinct thresholds: two class outlines never cross a grid edge at the very
                # same point (which would pinch the region between them to a single point)
                sigma = 1.2 if lab == VEG else 0.7  # tree canopies are ragged: smoother outline
                part.step(_indicator(m, min_cells, sigma), 0.5 + 0.0173 * i, above=lab)
        part.finish(GROUND)
        land.finish(GROUND)
        default = GROUND
        present = set(np.unique(part.tri_label)) | set(np.unique(part.p_label))
        defs.append(("base", "Base", col.base, {GROUND}, FieldSpec(base), FieldSpec(0.0)))
        d = opts.color_depth_mm
        if d > 0:
            # Layered colouring: every land colour is a thin skin on a ground-coloured body, so a
            # layer only carries the colours whose visible surface is within ~d of it (plus
            # building walls). The skin/body interface is the ground surface minus d, rounded down
            # to 0.5 mm terraces: shared by all classes and mostly flat (cheap to triangulate).
            I = FieldSpec(flat(_terraced(G_mm + min(0.0, opts.road_emboss_mm), d, base)))
            # one ground-coloured body under all land (outline + water cut only)
            defs.append(("body", "Body", col.ground, {GROUND}, I, FieldSpec(base)))
        for lab in (GROUND, BUILDING, WATER, RIVER, ROAD, VEG):
            if lab not in present:
                continue
            T = flat(fields[lab])
            if d <= 0:  # legacy: full-height colour columns
                defs.append((names[lab].lower(), names[lab], label_color[lab], {lab}, FieldSpec(T), FieldSpec(base)))
            elif lab in (WATER, RIVER):
                # water colour cap (flat on lakes, following the profile on rivers) on a ground body
                body_top = FieldSpec(flat(_terraced(fields[lab], d, base)))
                defs.append(("water", names[lab], label_color[lab], {lab}, FieldSpec(T), body_top))
                defs.append(("fill", f"{names[lab]} (body)", col.ground, {lab}, body_top, FieldSpec(base)))
            else:  # skin on the shared body
                defs.append((names[lab].lower(), names[lab], label_color[lab], {lab}, FieldSpec(T), I))
    else:  # elevation bands
        bands = col.bands
        nb = len(bands)
        hv = Sm[inside_v]
        if col.band_thresholds_m and len(col.band_thresholds_m) == nb - 1:
            th = sorted(col.band_thresholds_m)
        elif col.band_mode == "quantile":
            th = list(np.percentile(hv, np.linspace(0, 100, nb + 1)[1:-1]))
        else:
            th = list(np.linspace(hmin, hmax, nb + 1)[1:-1])
        lo_mm = [base] + [round(float(min((t - hmin) * zscale + base + opts.floor_mm, zcap)), 3) for t in th]
        wmask = surf.masks[WATER] | surf.masks[RIVER]
        water_sep = opts.elevation_water and wmask.any()
        if water_sep:  # flat, recessed water in its own colour; bands cover the land only
            progress(0.08, "Cutting water")
            _cut_water(part, P(surf.masks[WATER]), P(surf.masks[RIVER]), min_cells)
        # with separate water the bands only cover land: use the land surface continued over
        # the water so shores are clean walls (see Surface.surface_land)
        Sb_mm = np.round(zmm(surf.surface_land), 3) if water_sep else S_mm
        progress(0.1, "Cutting elevation contours")
        for k in range(nb - 1, 0, -1):
            part.step(flat(Sb_mm), lo_mm[k], above=BAND0 + k)
        part.finish(BAND0)
        default = BAND0
        label_color = {BAND0 + k: bands[k] for k in range(nb)}
        present_w = [lab for lab in (WATER, RIVER)
                     if water_sep and ((part.tri_label == lab).any() or (part.p_label == lab).any())]
        if present_w:
            wtop = np.maximum(np.round(zmm(surf.level), 3) - opts.water_recess_mm, lowest)
            d = opts.color_depth_mm
            for lab in present_w:
                nm = "Water" if lab == WATER else "Rivers"
                label_color[lab] = col.water
                if d > 0:
                    body = FieldSpec(flat(_terraced(wtop, d, base)))
                    defs.append(("water", nm, col.water, {lab}, FieldSpec(flat(wtop)), body))
                    defs.append(("fill", f"{nm} (fill)", bands[0], {lab}, body, FieldSpec(base)))
                else:
                    defs.append(("water", nm, col.water, {lab}, FieldSpec(flat(wtop)), FieldSpec(base)))
            notes.append("Water has its own colour: printing it needs a multi-material printer "
                         "(turn off 'separate water colour' for single-extruder colour changes).")
        defs.append(("base", "Base", col.base, {GROUND}, FieldSpec(base), FieldSpec(0.0)))
        for k in range(nb):
            hi = lo_mm[k + 1] if k + 1 < nb else np.inf
            lbl = f"Band {k + 1}" + (f" ({th[k - 1]:.0f} m+)" if k else "")
            defs.append(("band", lbl, bands[k], {BAND0 + j for j in range(k, nb)},
                         FieldSpec(flat(Sb_mm), hi=hi), FieldSpec(lo_mm[k])))
            if k > 0:
                color_changes.append({"z_mm": round(lo_mm[k], 2), "color": bands[k],
                                      "elevation_m": round(float(th[k - 1]), 1)})
        if col.base.upper() != bands[0].upper():
            color_changes.insert(0, {"z_mm": round(base, 2), "color": bands[0], "elevation_m": None})
    if opts.frame:
        defs.append(("frame", "Frame", col.frame, {FRAME}, FieldSpec(base + opts.frame_height_mm), FieldSpec(0.0)))
        label_color[FRAME] = col.frame

    # --- base plate as its own print: drop it from the model, which then starts on top of it
    base_defs: list = []
    if opts.base_separate:
        defs = [dd for dd in defs if dd[0] != "base"]
        base_defs.append(("base", "Base plate", col.base, {GROUND, FRAME} if opts.frame else {GROUND},
                          FieldSpec(base), FieldSpec(0.0)))
        sep = []
        for cat, name, colr, labels, top, bottom in defs:
            if cat in ("model", "frame"):  # these used to reach down to the bed
                if cat == "frame" and opts.frame_height_mm <= 0:
                    continue
                bottom = FieldSpec(base)
            sep.append((cat, name, colr, labels, top, bottom))
        defs = sep
        color_changes = [{**c, "z_mm": round(c["z_mm"] - base, 2)} for c in color_changes if c["z_mm"] - base > 1e-6]

    # --- meshes per tile
    parts: list[Part] = []
    base_parts: list[Part] = []
    re_ = np.linspace(0, R, opts.tiles_y + 1).round().astype(int)
    ce_ = np.linspace(0, C, opts.tiles_x + 1).round().astype(int)
    ntiles = opts.tiles_x * opts.tiles_y
    multi = ntiles > 1
    done = 0
    total = ntiles * (len(defs) + len(base_defs))
    for ty in range(opts.tiles_y):
        for tx in range(opts.tiles_x):
            win = (re_[ty], re_[ty + 1], ce_[tx], ce_[tx + 1])
            tname = f"Tile {ty * opts.tiles_x + tx + 1}" if multi else opts.name
            for target, dlist in ((parts, defs), (base_parts, base_defs)):
                for cat, name, color, labels, top, bottom in dlist:
                    src = outline if cat == "base" else land if cat == "body" else part
                    mesh = src.build_part(labels, top, bottom, win, cell_mm,
                                          tx * opts.tile_gap_mm, ty * opts.tile_gap_mm)
                    done += 1
                    progress(0.2 + 0.8 * done / total, f"Meshing {name}" + (f" ({tname})" if multi else ""))
                    if mesh is not None and len(mesh.triangles):
                        if base_defs and target is parts:
                            mesh = mesh.translate(0, 0, -base)  # sits on the bed; the base is its own print
                        target.append(Part(name if not multi else f"{tname} - {name}", color.upper(), mesh,
                                           tname, cat))

    # --- filaments
    colors_order: list[str] = []
    for p in parts:
        if p.color not in colors_order:
            colors_order.append(p.color)
    filaments = [{"slot": i + 1, "color": c,
                  "used_by": sorted({p.name.split(" - ")[-1] for p in parts if p.color == c})}
                 for i, c in enumerate(colors_order)]

    vmax = max(float(p.mesh.vertices[:, 2].max()) for p in parts)
    swaps, max_colors = _layer_color_stats(parts, vmax)
    stats = {
        "size_mm": [round(C * cell_mm, 1), round(R * cell_mm, 1), round(vmax, 1)],
        "cell_mm": round(cell_mm, 3),
        "ground_m_per_cell": round(g.step, 3),
        "scale": f"1:{round(1000 / mm_per_m):,}",
        "mm_per_m": mm_per_m,
        "elevation_range_m": [round(hmin, 1), round(hmax, 1)],
        "relief_mm": round((hmax - hmin) * zscale, 2),
        "triangles": int(sum(len(p.mesh.triangles) for p in parts)),
        "vertices": int(sum(len(p.mesh.vertices) for p in parts)),
        "parts": len(parts),
        "tiles": ntiles,
        "colour_changes_est": swaps,
        "max_colours_per_layer": max_colors,
    }
    if ntiles > 1:
        stats["tile_size_mm"] = [round(float((ce_[1] - ce_[0]) * cell_mm), 1),
                                 round(float((re_[1] - re_[0]) * cell_mm), 1)]

    # --- per-cell colour map for the thumbnail (labels sampled at cell centres)
    def centre(a: np.ndarray) -> np.ndarray:
        a = a.reshape(R + 1, C + 1)
        return (a[:-1, :-1] + a[:-1, 1:] + a[1:, :-1] + a[1:, 1:]) / 4

    cl = part.classify([centre(f) for f, *_ in part.steps], default)
    rgb = np.full((R, C, 3), 255, np.uint8)
    for lab, colr in label_color.items():
        rgb[cl == lab] = _hex_rgb(colr)
    rgb[cl == DROPPED] = 255
    zthumb = S_mm.copy()
    if opts.frame:
        zthumb = np.where(f_model >= 0, zthumb, base + opts.frame_height_mm)
    if base_parts:
        stats["base_plate_mm"] = [round(C * cell_mm, 1), round(R * cell_mm, 1), round(base, 2)]
        notes.append(f"Base plate ({base:g} mm, {col.base}) is a separate file; the model starts on top of it "
                     "and has the same footprint.")
    return BuildResult(parts, stats, filaments, color_changes, notes, cell_rgb=rgb, z_mm=zthumb,
                       base_parts=base_parts or None)
