"""Generation pipeline: elevation -> surface -> colour regions -> mesh parts."""

from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy import ndimage

from . import config
from .geo import AOI, Grid, shape_mask
from .mesh import Mesh, heightfield_solid
from .options import Options
from .raster import (despike, fill_nan, fix_mask_pinches, fix_pinches, remove_small, smooth,
                     vertex_to_cell_frac)

ProgressFn = Callable[[float, str], None]

# label ids (cell grid)
OUTSIDE, FRAME, GROUND, VEG, WATER, ROAD, BUILDING = -1, 0, 1, 2, 3, 4, 5
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
    height_m: np.ndarray            # vertex heights (metres)
    masks: dict[int, np.ndarray]    # vertex label masks
    water: np.ndarray               # vertex water mask (after cleanup)
    roads: np.ndarray
    notes: list[str] = field(default_factory=list)


def _close_and_fill(values: np.ndarray, mask: np.ndarray, min_cells: int) -> tuple[np.ndarray, np.ndarray]:
    m = ndimage.binary_closing(mask, structure=np.ones((3, 3)), iterations=1, border_value=0) | mask
    m = remove_small(m, min_cells)
    v = fill_nan(np.where(mask & np.isfinite(values), values, np.nan)) if m.any() else values
    return m, v


def _water_levels(G: np.ndarray, W: np.ndarray, hint: np.ndarray | None, step: float,
                  tol: float = 2.0, max_slope: float = 0.01, uncovered: np.ndarray | None = None) -> np.ndarray:
    """Flatten lakes / sea to a level without carving sloped rivers.

    Water is split into near-level sub-regions (surface slope < ``max_slope``);
    each is flattened to a robust low level (10th percentile, or the median of
    LiDAR water returns when available), but only where the terrain is within
    ``tol`` metres of that level. Remaining sloped water is lightly smoothed and
    never raised above the original surface.
    """
    out = G.copy()
    if not W.any():
        return out
    Gs = ndimage.gaussian_filter(G, 2.0, mode="nearest")
    gy, gx = np.gradient(Gs, step)
    flat = W & (np.hypot(gx, gy) < max_slope)
    if uncovered is not None:
        flat |= W & uncovered  # no data there: heights are pure interpolation
    lab, n = ndimage.label(flat, structure=np.ones((3, 3)))
    done = np.zeros_like(W)
    if n:
        idx = np.arange(1, n + 1)
        src = G if uncovered is None else np.where(uncovered, np.inf, G)  # level from measured water only
        lvl = np.asarray(ndimage.labeled_comprehension(src, lab, idx, lambda v: np.percentile(v[np.isfinite(v)], 10)
                                                       if np.isfinite(v).any() else np.percentile(v, 10),
                                                       np.float64, np.nan))
        if uncovered is not None:
            lvl_all = np.asarray(ndimage.labeled_comprehension(G, lab, idx, lambda v: np.percentile(v, 10),
                                                               np.float64, np.nan))
            lvl = np.where(np.isfinite(lvl), lvl, lvl_all)
        if hint is not None and hint.any():
            lab_h = np.where(hint, lab, 0)
            cnt_h = np.asarray(ndimage.sum_labels(np.ones_like(G), lab_h, idx))
            med_h = np.nan_to_num(np.asarray(ndimage.median(G, lab_h, idx), float), nan=np.inf)
            lvl = np.where(cnt_h >= 5, np.minimum(med_h, lvl + tol), lvl)
        L = np.concatenate([[np.inf], lvl])[lab]
        done = (lab > 0) & (G <= L + tol)
        if uncovered is not None:
            done |= (lab > 0) & uncovered
        out[done] = L[done]
    rest = W & ~done
    if rest.any() and done.any():
        # shore / cliff-adjacent water with interpolated heights joins the nearby body's level;
        # only reaches further away (real sloped rivers, rapids) keep their own profile
        dist, (ri, ci) = ndimage.distance_transform_edt(~done, return_indices=True)
        near = rest & (dist * step <= max(30.0, 3 * step))
        out[near] = out[ri[near], ci[near]]
        rest &= ~near
    if rest.any():
        wsum = ndimage.gaussian_filter(W.astype(np.float64), 1.5)
        vsum = ndimage.gaussian_filter(np.where(W, out, 0.0), 1.5)
        sm = np.divide(vsum, wsum, out=out.copy(), where=wsum > 1e-6)
        out[rest] = np.minimum(out[rest], sm[rest])
    return out


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
    osm_ways = osm.get("waterways") if opts.osm_water else None
    osm_roads = osm.get("roads")
    osm_bld = osm.get("buildings")

    if elev.source == "lidar":
        G = a["ground"]
        cov = a["covered"].astype(bool)
        dens = a["density"]
        if cov.mean() < 0.999:
            notes.append(f"{100 * (1 - cov.mean()):.1f}% of the area has no LiDAR coverage (interpolated).")
        W = a["water"] >= 0.3
        W = remove_small(W, cells(30))
        if opts.voids_as_water:
            k = max(3, int(np.ceil(4.0 / step)) | 1)
            void = cov & (ndimage.uniform_filter(dens.astype(np.float32), size=k) == 0)
            void = remove_small(void, cells(600))
            W |= void
        if osm_water is not None:
            W |= osm_water
        Gf = smooth(fill_nan(G), opts.smoothing)
        water_hint = a["water"] >= 0.5
        if opts.flatten_water and W.any():
            Gf = _water_levels(Gf, W, water_hint, step, uncovered=~cov)
        if osm_ways is not None:
            W |= osm_ways  # streams: coloured / recessed, never flattened
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
        Gf = smooth(fill_nan(dtm), opts.smoothing)
        Df = fill_nan(np.where(np.isfinite(dsm), dsm, dtm))
        W = osm_water.copy() if osm_water is not None else zeros.copy()
        if opts.flatten_water and W.any():
            Gf = _water_levels(Gf, W, None, step, uncovered=~cov)
        if osm_ways is not None:
            W |= osm_ways
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
        WATER: W & ~(BRm if opts.surface != "terrain" else zeros),
        VEG: Vm,
    }
    return Surface(S, masks, masks[WATER], roads, notes)


# ---------------------------------------------------------------- parts

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
    z_mm: np.ndarray | None = None       # (R+1, C+1) top heights


def _hex_rgb(c: str) -> tuple[int, int, int]:
    return int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)


def _cell_labels(surf: Surface, R: int, C: int) -> np.ndarray:
    lab = np.full((R, C), GROUND, dtype=np.int8)
    for k in (VEG, WATER, ROAD, BUILDING):  # increasing priority
        frac = vertex_to_cell_frac(surf.masks[k])
        lab[frac >= 0.5] = k
    return lab


def _merge_small_regions(lab: np.ndarray, valid: np.ndarray, min_cells: int) -> np.ndarray:
    if min_cells <= 1:
        return lab
    out = lab.copy()
    unknown = np.zeros(lab.shape, bool)
    for k in np.unique(lab[valid]):
        m = (lab == k) & valid
        comps, n = ndimage.label(m)
        if n == 0:
            continue
        sizes = np.bincount(comps.ravel())
        small = sizes < min_cells
        small[0] = False
        unknown |= small[comps]
    if unknown.any() and (valid & ~unknown).any():
        src = ~(valid & ~unknown)
        _, (ri, ci) = ndimage.distance_transform_edt(src, return_indices=True)
        out[unknown] = lab[ri[unknown], ci[unknown]]
    return out


def build_parts(elev: Elevation, surf: Surface, opts: Options, progress: ProgressFn) -> BuildResult:
    g = elev.grid
    nx, ny = g.nx, g.ny
    cell_mm = opts.size_mm / max(nx, ny)
    mm_per_m = cell_mm / g.step
    notes = list(surf.notes)

    # --- padded label grid (frame ring around the model)
    pad = int(np.ceil(opts.frame_width_mm / cell_mm)) if opts.frame else 0
    corner_cells = opts.corner_radius_mm / cell_mm
    inside0 = shape_mask(opts.shape, ny, nx, nx, ny, corner_cells, pad=pad)
    R, C = inside0.shape
    L = np.full((R, C), OUTSIDE, dtype=np.int8)
    if opts.frame:
        outer = shape_mask(opts.shape, ny, nx, nx, ny, corner_cells, pad=pad, grow=pad)
        L[outer] = FRAME
    L[inside0] = GROUND
    if opts.color_mode == "landcover":
        lab = _cell_labels(surf, ny, nx)
        Lin = L[pad:pad + ny, pad:pad + nx]
        Lin[inside0[pad:pad + ny, pad:pad + nx]] = lab[inside0[pad:pad + ny, pad:pad + nx]]
        min_cells = int(round(opts.min_region_mm2 / (cell_mm * cell_mm)))
        L[pad:pad + ny, pad:pad + nx] = np.where(
            Lin >= GROUND, _merge_small_regions(Lin, Lin >= GROUND, min_cells), Lin)
    progress(0.1, "Resolving region topology")
    L = fix_pinches(L, protect=OUTSIDE)
    inside = L >= GROUND

    # --- heights in mm on the padded vertex grid
    H = surf.height_m
    iv = np.zeros((ny + 1, nx + 1), bool)
    ic = inside[pad:pad + ny, pad:pad + nx]
    iv[:-1, :-1] |= ic; iv[:-1, 1:] |= ic; iv[1:, :-1] |= ic; iv[1:, 1:] |= ic
    hmin = float(H[iv].min()); hmax = float(H[iv].max())
    zscale = mm_per_m * opts.z_exaggeration
    z = (H - hmin) * zscale + opts.base_mm + opts.floor_mm
    lowest = opts.base_mm + min(0.3, opts.floor_mm)
    if opts.water_recess_mm > 0 and surf.water.any():
        z = np.where(surf.water, np.maximum(z - opts.water_recess_mm, lowest), z)
    if opts.road_emboss_mm != 0 and surf.roads.any():
        z = np.where(surf.roads, np.maximum(z + opts.road_emboss_mm, lowest), z)
    if opts.max_height_mm:
        z = np.minimum(z, opts.base_mm + opts.max_height_mm)
    zp = np.full((R + 1, C + 1), opts.base_mm + opts.floor_mm)
    zp[pad:pad + ny + 1, pad:pad + nx + 1] = z
    zp = np.round(zp, 3)  # match the 3MF's 0.001 mm precision so flat/thin tests stay exact

    # --- part definitions: (category, name, color, mask, top, bottom)
    col = opts.colors
    defs: list[tuple[str, str, str, np.ndarray, object, float]] = []
    color_changes: list[dict] = []
    if opts.color_mode == "single":
        defs.append(("model", "Model", col.single, inside, zp, 0.0))
    else:
        defs.append(("base", "Base", col.base, inside, float(opts.base_mm), 0.0))
        if opts.color_mode == "landcover":
            cat_color = {GROUND: col.ground, VEG: col.vegetation, WATER: col.water,
                         ROAD: col.roads, BUILDING: col.buildings}
            groups: dict[str, list[int]] = {}
            for k in (GROUND, WATER, VEG, ROAD, BUILDING):
                if (L == k).any():
                    groups.setdefault(cat_color[k].upper(), []).append(k)
            for color, ks in groups.items():
                name = " + ".join(LABEL_NAMES[k].capitalize() for k in ks)
                defs.append((LABEL_NAMES[ks[0]], name, color, np.isin(L, ks), zp, float(opts.base_mm)))
        else:  # elevation bands
            bands = col.bands
            nb = len(bands)
            hv = H[iv]
            if col.band_thresholds_m and len(col.band_thresholds_m) == nb - 1:
                th = sorted(col.band_thresholds_m)
            elif col.band_mode == "quantile":
                th = list(np.percentile(hv, np.linspace(0, 100, nb + 1)[1:-1]))
            else:
                th = list(np.linspace(hmin, hmax, nb + 1)[1:-1])
            lo_mm = [round(float(opts.base_mm), 3)] + [round((t - hmin) * zscale + opts.base_mm + opts.floor_mm, 3)
                                                       for t in th]
            cmax = np.maximum.reduce([zp[:-1, :-1], zp[:-1, 1:], zp[1:, :-1], zp[1:, 1:]])
            for k in range(nb):
                lo = lo_mm[k]
                hi = lo_mm[k + 1] if k + 1 < nb else np.inf
                m = inside if k == 0 else fix_mask_pinches(inside & (cmax > lo + 0.01), allowed=inside)
                if not m.any():
                    continue
                top = np.clip(zp, lo, hi)
                lbl = f"Band {k + 1}" + (f" ({th[k - 1]:.0f} m+)" if k else "")
                defs.append(("band", lbl, bands[k], m, top, lo))
                if k > 0:
                    color_changes.append({"z_mm": round(lo, 2), "color": bands[k], "elevation_m": round(th[k - 1], 1)})
            if col.base.upper() != bands[0].upper():
                color_changes.insert(0, {"z_mm": round(opts.base_mm, 2), "color": bands[0], "elevation_m": None})
    if opts.frame and (L == FRAME).any():
        defs.append(("frame", "Frame", col.frame, L == FRAME, float(opts.base_mm + opts.frame_height_mm), 0.0))

    # --- tiles
    parts: list[Part] = []
    re_ = np.linspace(0, R, opts.tiles_y + 1).round().astype(int)
    ce_ = np.linspace(0, C, opts.tiles_x + 1).round().astype(int)
    ntiles = opts.tiles_x * opts.tiles_y
    multi = ntiles > 1
    done = 0
    for ty in range(opts.tiles_y):
        for tx in range(opts.tiles_x):
            r0, r1, c0, c1 = re_[ty], re_[ty + 1], ce_[tx], ce_[tx + 1]
            tname = f"Tile {ty * opts.tiles_x + tx + 1}" if multi else opts.name
            ox = c0 * cell_mm + tx * opts.tile_gap_mm
            oy = r0 * cell_mm + ty * opts.tile_gap_mm
            for cat, name, color, mask, top, bottom in defs:
                sub = mask[r0:r1, c0:c1]
                if not sub.any():
                    continue
                t = top if np.isscalar(top) else top[r0:r1 + 1, c0:c1 + 1]
                mesh = heightfield_solid(sub, t, bottom, cell_mm, cell_mm, ox, oy)
                if mesh is not None:
                    parts.append(Part(f"{name}" if not multi else f"{tname} - {name}", color.upper(), mesh, tname, cat))
            done += 1
            progress(0.2 + 0.8 * done / ntiles, f"Meshing {done}/{ntiles}")

    # --- filaments
    colors_order: list[str] = []
    for p in parts:
        if p.color not in colors_order:
            colors_order.append(p.color)
    filaments = []
    for i, c in enumerate(colors_order):
        filaments.append({"slot": i + 1, "color": c,
                          "used_by": sorted({p.name.split(" - ")[-1] for p in parts if p.color == c})})

    vmax = max(float(p.mesh.vertices[:, 2].max()) for p in parts)
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
    }
    if ntiles > 1:
        stats["tile_size_mm"] = [round((ce_[1] - ce_[0]) * cell_mm, 1), round((re_[1] - re_[0]) * cell_mm, 1)]

    # --- per-cell colour map (thumbnail / 2D preview)
    rgb = np.zeros((R, C, 3), np.uint8)
    rgb[:] = 255
    zc = (zp[:-1, :-1] + zp[:-1, 1:] + zp[1:, :-1] + zp[1:, 1:]) / 4
    for cat, name, color, mask, top, bottom in defs:
        if cat == "base":
            continue
        if cat == "band":
            m = mask & (zc > bottom)
        else:
            m = mask
        rgb[m] = _hex_rgb(color)
    rgb[L == OUTSIDE] = 255
    return BuildResult(parts, stats, filaments, color_changes, notes, rgb, zp)
