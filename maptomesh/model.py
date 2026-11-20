"""Generation pipeline: elevation -> surface -> colour regions -> mesh parts."""

from __future__ import annotations

import hashlib
import io
import os
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
from . import labels as lbl
from .raster import despike, fill_nan, masked_median, remove_small, smooth

ProgressFn = Callable[[float, str], None]

# region labels
FRAME, GROUND, VEG, WATER, ROAD, BUILDING = 0, 1, 2, 3, 4, 5
RIVER = 7  # rivers / streams: water colour, but sloped (lakes and sea are flat WATER)
BRIDGE = 30  # bridge decks: a region of their own (deck height); road colour in land cover
BLD_BAND0, BRG_BAND0 = 40, 60  # elevation bands of buildings / bridge decks (band 0: BUILDING / BRIDGE)
LABEL, LIP = 8, 9
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
        arrays, info = _fetch_elevation(opts, aoi, grid, progress, cancel)
        info["grid"] = {"nx": grid.nx, "ny": grid.ny, "step_m": grid.step}
        buf = io.BytesIO()
        np.savez_compressed(buf, info=np.frombuffer(json.dumps(info).encode(), dtype=np.uint8), **arrays)
        config.atomic_write(path, buf.getvalue())
        elev = Elevation(opts.source, grid, arrays, info)
    with _MEM_LOCK:
        _MEM[key] = elev
        while len(_MEM) > 4:
            _MEM.popitem(last=False)
    return elev


TREES_MIN_KNOWN = 0.9   # share of the area whose data shows trees, for trees to be drawn at all

MAPTERHORN_FIRST = os.environ.get("MAPTOMESH_MAPTERHORN_FIRST", "1") != "0"


def _models_suffice(opts: Options, aoi: AOI, grid: Grid) -> bool:
    """Bare-terrain jobs where Mapterhorn's elevation models are at least as fine as the grid
    cells over the whole area: its tiles (a few MB) replace the point clouds (often hundreds of MB
    from the agencies' servers). The ground then differs by a few cm (median) to a few dm on steep
    slopes - hundredths of a millimetre on the print."""
    if not (MAPTERHORN_FIRST and opts.ground_only and opts.dem_preference == "mapterhorn"):
        return False
    from .providers.mapterhorn import models_suffice
    return models_suffice(aoi, grid)


def _fetch_elevation(opts: Options, aoi: AOI, grid: Grid, progress: ProgressFn,
                     cancel: threading.Event) -> tuple[dict, dict]:
    """Best available elevation for the area, as per-vertex arrays:

    1. classified LiDAR point clouds (unless ``source`` is an elevation-models-only mode),
    2. elevation models for everything the point clouds don't cover, highest resolution first,
       down to Copernicus 30 m (``maptomesh.dem.best_dems``).

    Elevation-model areas are expressed like an unclassified point cloud: the DTM is the ground
    and a DSM, where there is one, provides "class 1" heights that the surface builder turns into
    buildings (OSM footprints) and trees, exactly as for LiDAR projects without classification."""
    from . import providers  # noqa: F401 - registers the elevation providers
    from .dem import best_dems
    from .pointcloud import empty_result, fetch_lidar, provider_credit, provider_licence_note

    sub = lambda a, b: (lambda f, m: progress(a + (b - a) * f, m))  # noqa: E731
    lidar = opts.source == "lidar"
    models_suffice = lidar and _models_suffice(opts, aoi, grid)
    lidar = lidar and not models_suffice
    if lidar:
        from .lidar_index import INDEX
        from .limits import lidar_caps
        max_tiles, max_ept_km2 = lidar_caps(opts)
        r = fetch_lidar(aoi, grid, INDEX.get() if INDEX.ready else None, sub(0.0, 0.8), cancel,
                        max_tiles=max_tiles, us_source=opts.us_source, allow_empty=True,
                        ground_only=opts.ground_only, max_ept_km2=max_ept_km2)
    else:
        r = empty_result(grid)
    info = r.info
    if models_suffice:
        info.setdefault("skipped", []).append(
            "LiDAR point clouds: not needed for bare terrain at this detail (the elevation models "
            "are as fine as the grid)")
    ground, unclass = r.ground.copy(), r.unclass.copy()
    water, density = r.water_frac.copy(), r.density.astype(np.float32)
    bld_cls, veg_cls, covered = r.bld_classified.copy(), r.veg_classified.copy(), r.covered.copy()
    # where the data can show trees: point clouds, and elevation models with a surface model (DSM);
    # bare-earth models (FABDEM, Mapterhorn, ...) can't
    veg_known = covered.copy()
    # surface-only fallbacks (Copernicus 30 m, ...): their "ground" holds buildings and trees, blurred
    surface_only = np.zeros(covered.shape, bool)
    sources = []
    credits = []
    warnings = [w for w in (provider_licence_note(k) for k in r.info.get("providers", [])) if w]
    n = covered.size
    if r.info.get("points"):
        label = ", ".join(provider_credit(k)[0] for k in r.info.get("providers", [])) or "LiDAR"
        sources.append({"label": f"LiDAR point clouds: {label}", "share": float(covered.mean()), "lidar": True})
        credits += [provider_credit(k)[1] for k in r.info.get("providers", [])]
    need = ~covered
    if need.any():
        lo = 0.8 if lidar else 0.0
        comp = best_dems(aoi, grid, need=need, ref=np.where(covered, ground, np.nan),
                         progress=sub(lo, 1.0), cancel=cancel, dsm=not opts.ground_only,
                         shared_first=opts.dem_preference == "mapterhorn")
        take = need & np.isfinite(comp.dtm)
        ground[take] = comp.dtm[take]
        water[take] = 0.0
        raised = take & np.isfinite(comp.dsm) & (comp.dsm > comp.dtm + 1.0)
        unclass[raised] = comp.dsm[raised]
        bld_cls[take] = False
        veg_cls[take] = False
        veg_known[take] = np.isfinite(comp.dsm[take])
        so = [i for i, u in enumerate(comp.used) if u.get("surface_only")]
        if so:
            surface_only[take] = np.isin(comp.source[take], so)
        density[take] = np.maximum(density[take], 1.0)  # data, not a void (voids become water)
        covered |= take
        for u in comp.used:
            sources.append({"label": u["label"], "share": u["share"] * need.sum() / n, "res_m": u["res_m"],
                            "offset_m": u["offset_m"], "detail": u["detail"], "dsm": u["dsm"]})
            credits.append(u["attribution"])
            warnings += u.get("licence_notes", [])
        info.setdefault("errors", [])
        info["errors"] = (info["errors"] + comp.errors)[:8]
        info["skipped"] = info.get("skipped", []) + comp.skipped
        if not covered.any():
            raise RuntimeError("No elevation data covers this area." + (f" {comp.errors[0]}" if comp.errors else ""))
    info["sources"] = sources
    info["attribution"] = [c for i, c in enumerate(credits) if c and c not in credits[:i]]
    info["licence_warnings"] = [w for i, w in enumerate(warnings) if w and w not in warnings[:i]]
    info["coverage"] = float(covered.mean())
    arrays = {"ground": ground, "water": water, "building": r.building, "bridge": r.bridge,
              "veg": r.veg, "unclass": unclass, "bld_cls": bld_cls, "veg_cls": veg_cls,
              "density": density, "covered": covered, "veg_known": veg_known, "surface_only": surface_only}
    return arrays, info


def _source_notes(info: dict) -> list[str]:
    srcs = info.get("sources") or []
    notes = [f"Licence restriction: {w}" for w in info.get("licence_warnings", [])]
    if len(srcs) > 1 or (srcs and not srcs[0].get("lidar")):
        parts = [f"{s['label']} {100 * s['share']:.0f}%" for s in srcs if s["share"] >= 0.0005]
        notes.append("Elevation: " + "; ".join(parts) + ".")
    for s in info.get("skipped", []) or []:
        notes.append(s)
    return notes


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


def _osm_bridges(osm_br, BRm, BRv, U, Gf, W, opts, notes, step):
    """Add OpenStreetMap bridges that the data has no bridge class for.

    Elevation models (and LiDAR projects without class 17) carry no bridge information, so the
    water cut swallowed every deck (London's Thames bridges). The deck height comes from the
    surface heights on the bridge (DSM / class-1 returns) where they stand clear of the ground or
    water below; spans without them carry the height of the bridge's ends across.

    OpenStreetMap ways are drawn wider than many decks (or a little offset): cells beside a deck
    the data does classify take that deck's height. The returns there are railings, lamp posts,
    lift-bridge towers and cables, or the ground below; taken as the deck they printed as a comb
    of spikes along the bridge (QEW Burlington). What stands on a deck elsewhere (posts, lift
    towers, cables) is removed by a low percentile along each bridge: the deck is the lowest
    surface above the ground or water, and it is smooth."""
    if osm_br is None or not osm_br.any() or not opts.show_bridges:
        return BRm, BRv
    nb = osm_br & ~BRm
    if not nb.any():
        return BRm, BRv
    deck = np.where(nb & np.isfinite(U) & (np.nan_to_num(U, nan=-1e9) > Gf + 1.0), U, np.nan)
    if BRm.any():
        dist, (bi, bj) = ndimage.distance_transform_edt(~BRm, return_indices=True)
        beside = nb & (dist <= max(2.0, 8.0 / step))
        deck = np.where(beside, BRv[bi, bj], deck)
    # approaches on land: at least the ground; cells near deck heights (e.g. the edge of a wide
    # way where the returns miss) are interpolated instead of dropping to the ground mid-span
    r = max(1, int(round(4.0 / step)))
    near = ndimage.maximum_filter(np.isfinite(deck), size=2 * r + 1)
    ends = nb & ~W & np.isnan(deck) & ~near
    deck = np.where(ends, Gf, deck)
    if not np.isfinite(deck[nb]).any():
        return BRm, BRv
    lab, n = ndimage.label(nb, structure=np.ones((3, 3)))
    filled = np.full(deck.shape, np.nan)
    for k, sl in enumerate(ndimage.find_objects(lab), start=1):   # interpolate within each bridge
        if sl is None:
            continue
        m = lab[sl] == k
        d = np.where(m, deck[sl], np.nan)
        if np.isfinite(d).any():
            filled[sl] = np.where(m, fill_nan(d), filled[sl])
    nb &= np.isfinite(filled)
    size = min(11, max(3, int(round(30.0 / step)) | 1))     # ~30 m: wider than a lift tower
    filled = masked_median(filled, nb, nb, size, q=20.0)
    BRv = np.where(nb, filled, BRv)
    notes.append("Bridges without a bridge class in the elevation data come from OpenStreetMap.")
    return BRm | nb, BRv


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


def _bare_earth_buildings(bld_v, Gf, bare, so, osm, step, notes):
    """Buildings where the elevation data is bare earth (no point cloud, no surface model), which
    shows none: PLATEAU 3D city models where there are some (Japan; ``maptomesh.buildings``), else
    OpenStreetMap footprints raised to their mapped heights (LoD1 blocks), with heights OSM lacks
    (and footprints it doesn't have) from Overture."""
    cells = lambda m2: max(1, int(round(m2 / (step * step))))  # noqa: E731
    h = osm.get("building_h")
    h = np.full(Gf.shape, np.nan) if h is None else np.asarray(h, dtype=np.float64)
    osm_b = osm.get("buildings")
    osm_b = np.isfinite(h) if osm_b is None else osm_b
    pl, ov = osm.get("plateau"), osm.get("overture")
    memo = {}

    def ground(where):
        """The ground buildings stand on. A 30 m surface model already holds the buildings,
        blurred: roofs go on the ground under them (an opening ~150 m wide removes such bumps and
        keeps slopes)."""
        if "b" not in memo:
            memo["b"] = Gf
        if "o" not in memo and so is not None and (where & so).any():
            k = max(3, int(round(150.0 / step)) | 1)
            memo["b"] = np.where(so, np.minimum(Gf, ndimage.grey_opening(Gf, size=(k, k), mode="nearest")), Gf)
            memo["o"] = True
        return memo["b"]

    if ov is not None:
        # OpenMapTiles' render_height is 5 m where OSM has no height or floors: Overture's (OSM tags
        # again, Microsoft ML heights in North America, ...) where it has one; its footprints from
        # other datasets where OSM has no building
        unknown = osm_b & ~(np.nan_to_num(h, nan=0.0) > 5.0)
        filled = unknown & np.isfinite(ov["h"]) & bare
        new = ov["other"] & ~ndimage.binary_dilation(osm_b, iterations=1) & bare
        h = np.where(filled, ov["h"], h)
        h = np.where(new, np.where(np.isfinite(ov["h"]), ov["h"], 5.0), h)
        if filled.sum() + new.sum() >= cells(200):
            notes.append("Buildings OpenStreetMap has no height for (or no footprint) take Overture Maps' "
                         "heights and footprints where it has them.")
    if pl is not None:
        rel, ids = pl["rel"], pl["id"]
        pm = np.isfinite(rel) & bare
        if pm.sum() >= cells(200):
            # OSM buildings PLATEAU also has give way to its (surveyed, roof-shaped) ones
            lab, _ = ndimage.label(np.isfinite(h) & bare)
            dup = np.unique(lab[ndimage.binary_dilation(pm, iterations=1) & (lab > 0)])
            h = np.where(np.isin(lab, dup), np.nan, h)
            # each building stands on its lowest ground
            lab_b = np.where(pm, ids, 0)
            idx = np.unique(lab_b[lab_b > 0])
            lut = np.zeros(int(ids.max()) + 1)
            lut[idx] = _label_reduce(np.minimum, ground(pm), lab_b, int(ids.max()))[idx - 1]
            bld_v = np.where(pm, np.fmax(bld_v, lut[ids] + rel), bld_v)
            notes.append(f"Buildings: Project PLATEAU 3D city models ({', '.join(pl.get('cities', []))}): "
                         "surveyed roof shapes (LoD2) or blocks at the measured height (LoD1).")
    lift = bare & np.isfinite(h)
    if lift.sum() >= cells(200):
        bld_v = np.where(lift, np.fmax(bld_v, ground(lift) + h), bld_v)
        notes.append("Buildings where the elevation data is bare earth are OpenStreetMap footprints at "
                     "their mapped heights (tagged height or floors, otherwise a default).")
    return bld_v


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

    # every source arrives as point-cloud-style arrays (elevation models look like a project
    # without classification: DTM = ground, DSM = class 1)
    G = a["ground"]
    cov = a["covered"].astype(bool)
    dens = a["density"]
    notes += _source_notes(elev.info)
    if cov.mean() < 0.999:
        notes.append(f"{100 * (1 - cov.mean()):.1f}% of the area has no elevation data (interpolated).")
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
    if osm.get("bridges") is not None and opts.show_bridges:
        # decks over water in surface models are bridges (_osm_bridges), not trees or buildings
        um &= ~ndimage.binary_dilation(osm["bridges"], iterations=max(1, int(round(3.0 / step))))
    to_veg = um & bcls & ~vcls        # buildings classified -> class 1 is mostly trees
    maybe_bld = um & ~bcls & vcls     # trees classified -> class 1 is buildings + clutter
    unknown = um & ~bcls & ~vcls      # nothing classified
    both = um & bcls & vcls           # both classified -> class 1 is leftovers, e.g. skyscrapers
    cand = maybe_bld | unknown
    has_osm = osm_bld is not None and osm_bld.any()
    near = ndimage.binary_dilation(osm_bld, iterations=max(1, int(round(2.0 / step)))) if has_osm else None
    if both.any() and has_osm:
        # classifiers often leave the tallest towers unclassified (Chicago 2017: class 6 tops out
        # at ~120 m, Willis Tower is class 1); inside a mapped footprint that is the building
        bld_v = np.where(both & near, np.fmax(bld_v, Uf), bld_v)
    if cand.any():
        show = opts.show_buildings
        if has_osm:
            is_b = cand & near
            if show:
                notes.append("Part of the data has no building class: OpenStreetMap footprints identified "
                             "buildings, other raised objects count as trees.")
        else:
            is_b = _planar_components(Uf, cand, step) & cand
            if show:
                notes.append("Part of the data has no building class; a roof-flatness heuristic was used "
                             "(OpenStreetMap footprints give better results).")
        bld_v = np.where(is_b, np.fmax(bld_v, Uf), bld_v)
        to_veg |= unknown & ~is_b
    veg_v = np.where(to_veg, np.fmax(veg_v, Uf), veg_v)
    known = a.get("veg_known")
    notes += osm.get("building_notes", [])
    if known is not None and opts.show_buildings:
        bld_v = _bare_earth_buildings(bld_v, Gf, ~known.astype(bool), a.get("surface_only"), osm, step, notes)
    Bm, Bv = _close_and_fill(bld_v, above(bld_v, 2.0), cells(12))
    BRm, BRv = _close_and_fill(a["bridge"], above(a["bridge"], 1.0), cells(20))
    BRm, BRv = _osm_bridges(osm.get("bridges"), BRm, BRv, a["unclass"], Gf, W, opts, notes, step)
    if BRm.any():
        # a deck is smooth: cell-to-cell noise (railings, vehicles, returns off its edges) printed
        # as dents and bumps along every bridge
        k = min(9, max(3, int(round(10.0 / step)) | 1))
        BRv = np.where(BRm, masked_median(np.where(BRm, BRv, np.nan), BRm, BRm, k), BRv)
        # small steps (twin decks a metre or two apart: the Skyway) followed the grid diagonally,
        # a zigzag groove along the deck; softened into a ramp. Larger ones (one bridge over
        # another, a deck beside its ramp) stay.
        num = ndimage.gaussian_filter(np.where(BRm, BRv, 0.0), 1.0, mode="nearest")
        den = ndimage.gaussian_filter(BRm.astype(np.float64), 1.0, mode="nearest")
        soft = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-3)
        BRv = np.where(BRm & (np.abs(soft - BRv) < 2.5), soft, BRv)
    vfin = np.isfinite(veg_v)
    V0 = despike(np.where(vfin, veg_v, Gf), vfin, thresh=10.0, size=5)
    vmask = vfin & (V0 > Gf + 1.5) & ~Bm
    Vm, Vv = _close_and_fill(V0, vmask, cells(25))
    Vm &= ~Bm
    # trees only where (nearly) the whole area has data that shows them: a patch of forest that
    # ends at a data boundary reads as a real edge; plain ground everywhere doesn't mislead
    known = a.get("veg_known")
    land = cov & ~W                   # (water can't hold trees: a sea filled by a bare-earth model doesn't count)
    if known is not None and Vm.any() and land.any():
        frac = float(known[land].mean())
        if frac < TREES_MIN_KNOWN:
            Vm[:] = False
            notes.append(f"Trees left out: only {100 * frac:.0f}% of the area has data that shows them (LiDAR or "
                         "elevation models with a surface model); the rest is bare-earth elevation.")

    # --- water: clean the mask, one flat level per water body
    W = remove_small(W, cells(60))
    W |= ~W & ~remove_small(~W, cells(40))  # fill pin-holes (wave dropouts, boats)
    if opts.show_bridges:
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
        if opts.show_bridges:
            river &= ~BRm                 # rail / foot bridges too (not in the road mask)
    if river.any() and W.any():
        # LiDAR water returns along a stream belong to the stream, not to separate ponds: absorb
        # water patches touching the corridor that are narrow (widest inscribed circle < ~15 m)
        # or small. Lakes and the sea stay flat bodies of their own.
        lab, n = ndimage.label(W, structure=np.ones((3, 3)))
        touch = np.unique(lab[ndimage.binary_dilation(river, iterations=2) & (lab > 0)])
        if len(touch):
            radius = _label_reduce(np.maximum, ndimage.distance_transform_edt(W), lab, n)[touch - 1]
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
    if opts.show_buildings:
        S = np.where(Bm, np.maximum(S, Bv), S)
    if opts.show_bridges:
        S = np.where(BRm, np.maximum(S, BRv), S)
    if opts.surface == "full":
        S = np.where(Vm, np.maximum(S, Vv), S)

    roads = osm_roads.copy() if osm_roads is not None else zeros.copy()
    if opts.show_bridges:
        roads |= BRm
    masks = {
        BUILDING: Bm if opts.show_buildings else zeros,
        ROAD: roads,
        WATER: W_area,
        RIVER: river,
        VEG: Vm,
    }
    roof = np.maximum(np.where(np.isfinite(Bv), Bv, Gf), Gf)
    canopy = np.maximum(np.where(np.isfinite(Vv), Vv, Gf), Gf)
    if BRm.any():
        # beyond the deck: its nearest height, so the deck's colour outline (between grid
        # vertices) meets the deck height rather than a slope down to the ground (build_parts)
        _, (bi, bj) = ndimage.distance_transform_edt(~BRm, return_indices=True)
        BRv = np.where(BRm, BRv, BRv[bi, bj])
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
                   BRm if opts.show_bridges else zeros, ground_land, surface_land, river, notes)


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
    tips: list | None = None             # slicer / printing advice for this model
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


def _iso_distance(diag: np.ndarray, f: np.ndarray, th: float, reach: float,
                  query: np.ndarray | None = None) -> np.ndarray:
    """Signed distance (cells) from every vertex to the ``th`` iso-line of the piecewise-linear
    field ``f`` on the grid triangulated like ``Partition(R, C, diag)`` - the exact line the
    partition cuts along. Positive where ``f < th``; NaN farther than ``reach`` cells, and where
    the (optional) vertex mask ``query`` is False."""
    W1 = f.shape[1]
    out = np.full(f.size, np.nan)
    wg = f >= th
    coarse = ndimage.distance_transform_edt(~wg) + ndimage.distance_transform_edt(wg)
    sel = coarse <= reach + 1.5
    if query is not None:
        sel &= query
    idx = np.nonzero(sel.ravel())[0]
    if len(idx):
        dist = _iso_query(diag, f, th, wg, idx, reach + 1.5)
        if dist is not None:
            out[idx] = np.where(np.isfinite(dist), np.where(wg.ravel()[idx], -dist, dist), np.nan)
    return out.reshape(f.shape)


def _iso_query(diag: np.ndarray, f: np.ndarray, th: float, wg: np.ndarray, idx: np.ndarray,
               bound: float) -> np.ndarray | None:
    """Unsigned distance from the flat vertex indices ``idx`` to the iso-line (inf beyond
    ``bound``); None when there is no iso-line."""
    from scipy.spatial import cKDTree

    W1 = f.shape[1]
    fv = f.ravel()
    wet = wg.ravel()
    c4 = (wg[:-1, :-1], wg[:-1, 1:], wg[1:, :-1], wg[1:, 1:])
    rr, cc = np.nonzero((c4[0] | c4[1] | c4[2] | c4[3]) & ~(c4[0] & c4[1] & c4[2] & c4[3]))
    if not len(rr):
        return None
    v00 = rr * W1 + cc
    v01, v10 = v00 + 1, v00 + W1
    v11 = v10 + 1
    a = diag[rr, cc][:, None]
    tris = np.concatenate([np.where(a, np.column_stack([v00, v01, v11]), np.column_stack([v00, v01, v10])),
                           np.where(a, np.column_stack([v00, v11, v10]), np.column_stack([v01, v11, v10]))])
    sw = wet[tris]
    t = tris[sw.any(axis=1) & ~sw.all(axis=1)]
    fa = fv[t]
    xy = np.stack([t % W1, t // W1], axis=-1).astype(np.float64)        # (x = col, y = row)
    pts = []
    for i, j in ((0, 1), (1, 2), (2, 0)):
        cr = wet[t[:, i]] != wet[t[:, j]]
        u = np.where(cr, (th - fa[:, i]) / np.where(cr, fa[:, j] - fa[:, i], 1.0), np.nan)
        pts.append(xy[:, i] + u[:, None] * (xy[:, j] - xy[:, i]))
    P = np.stack(pts, axis=1)                                            # (M, 3, 2), NaN: no crossing
    first2 = np.argsort(~np.isfinite(P[:, :, 0]), axis=1, kind="stable")[:, :2]
    seg = np.take_along_axis(P, first2[:, :, None], axis=1)              # the iso segment per triangle
    w = np.linspace(0.0, 1.0, 12)[None, :, None]                         # <= 0.13 cells apart
    tree = cKDTree((seg[:, :1] * (1 - w) + seg[:, 1:] * w).reshape(-1, 2))
    dist, _ = tree.query(np.column_stack([idx % W1, idx // W1]).astype(np.float64),
                         distance_upper_bound=bound, workers=_WORKERS)
    return dist


_WORKERS = -1   # cKDTree queries on all cores


def _shore_cap(diag: np.ndarray, ind: np.ndarray, wtop: np.ndarray, wall: float, k: float,
               top: np.ndarray, hmax: float | None = None) -> np.ndarray:
    """``top`` (mm) capped near water: the cap is ``wall`` above the water surface exactly on the
    water outline, rising inland at ``k`` mm per cell. Land surfaces capped by it descend to one
    uniform shore edge along the smooth outline - sampled on the grid, a cliff that meets the water
    otherwise ends in a wall whose height jumps from vertex to vertex (jagged cliff tops). On the
    water side the cap continues below the water, so the land's edge is interpolated to exactly
    ``wall`` above the water.

    The cap reaches as far inland as the highest point (``hmax``, default ``top.max()``) could
    bind, but distances to the outline are only computed for vertices it can bind: a vertex
    ``c`` cells (grid distance) from the nearest vertex across the outline is at least
    ``c - sqrt(2)`` cells from the outline on land and at most ``c + 1`` in the water."""
    if hmax is None:
        hmax = float(np.max(top))
    reach = max(2.0, (hmax - float(np.min(wtop)) - wall) / k + 2.0)
    wg = ind >= 0.5
    d_dry = ndimage.distance_transform_edt(~wg)
    d_wet = ndimage.distance_transform_edt(wg)
    low = np.where(wg, -(d_wet + 1.5), d_dry - 1.5)        # lower bound of the signed distance
    query = (top > wtop + wall + k * low) & (d_dry + d_wet <= reach + 1.5)
    del low, d_dry, d_wet
    idx = np.nonzero(query.ravel())[0]
    if not len(idx):
        return top
    dist = _iso_query(diag, ind, 0.5, wg, idx, reach + 1.5)
    if dist is None:
        return top
    ok = np.isfinite(dist)
    idx, dist = idx[ok], dist[ok]
    d = np.where(wg.ravel()[idx], -dist, dist)
    out = top.copy().ravel()
    out[idx] = np.minimum(out[idx], wtop.ravel()[idx] + wall + k * d)
    return out.reshape(top.shape)


def _water_indicator(lakes: np.ndarray, rivers: np.ndarray, min_cells: int, sigma: float) -> np.ndarray | None:
    """Field whose 0.5 iso-line is the water outline. Lake and sea shores are smoothed with
    ``sigma`` (cells; about the printable feature size - LiDAR water edges wobble from cell to
    cell, which printed as a ragged coast), rivers keep a tight outline so narrow streams
    survive."""
    wet = lakes | rivers
    if not wet.any():
        return None
    ind = _indicator(wet, min_cells)
    if sigma > 0.7 + 1e-6 and lakes.any():
        wide = _indicator(wet, min_cells, sigma)
        ind = np.maximum(wide, _indicator(rivers, min_cells)) if rivers.any() else wide
    frag = _land_fragments(ind < 0.5, ~wet, 4 * max(min_cells, 1))
    if frag.any():
        ind = np.where(frag, 1.0, ind)
    return ind


def _land_fragments(land: np.ndarray, land0: np.ndarray, max_cells: int) -> np.ndarray:
    """Specks the smoothing broke off land narrower than it (a pier or breakwater about one cell
    wide keeps only its wider spots: a dashed line of tiny islands into the water). A piece of the
    smoothed ``land`` is a fragment when it is smaller than ``max_cells`` and the unsmoothed land
    ``land0`` it came from is at least three times its size; small islands that survive whole
    stay."""
    st = np.ones((3, 3), bool)
    lab, n = ndimage.label(land, structure=st)
    if n == 0:
        return np.zeros_like(land)
    size = np.bincount(lab.ravel(), minlength=n + 1)
    lab0, _ = ndimage.label(land0, structure=st)
    size0 = np.bincount(lab0.ravel())
    size0[0] = 0
    ids = np.arange(1, n + 1)
    src = _label_reduce(np.maximum, size0[lab0], lab, n)   # the largest land each piece touches
    bad = np.zeros(n + 1, bool)
    bad[1:] = (size[1:] < max_cells) & (src >= 3 * size[1:])
    return bad[lab]


def _cut_water(part: Partition, lakes: np.ndarray, rivers: np.ndarray, min_cells: int,
               land: Partition | None = None, sigma: float = 0.7) -> np.ndarray | None:
    """One outline for all water (label WATER), then the river share of it relabelled RIVER.
    Returns the indicator field whose 0.5 iso-line is the water outline."""
    ind = _water_indicator(lakes, rivers, min_cells, sigma)
    if ind is None:
        return None
    part.step(ind, 0.5, above=WATER)
    if land is not None:
        land.step(ind, 0.5, above=WATER)
    if rivers.any():
        if lakes.any():
            split = (ndimage.gaussian_filter(rivers.astype(np.float64), 1.0, mode="nearest")
                     - ndimage.gaussian_filter(lakes.astype(np.float64), 1.0, mode="nearest"))
            part.step(split, 0.0123, above=RIVER, only=WATER)
        else:
            part.step(np.ones(ind.shape), 0.0, above=RIVER, only=WATER)
    return ind


def _label_reduce(op, values: np.ndarray, lab: np.ndarray, n: int) -> np.ndarray:
    """``op`` (np.maximum / np.minimum) of ``values`` over each label 1..n of ``lab`` (array of n).
    Same result as ``ndimage.maximum/minimum(values, lab, 1..n)``, which sorts the whole grid on
    every call; this only visits the labelled cells."""
    sel = lab > 0
    v = np.asarray(values)[sel]
    if v.dtype.kind == "f":
        init = -np.inf if op is np.maximum else np.inf
    else:
        info = np.iinfo(v.dtype)
        init = info.min if op is np.maximum else info.max
    out = np.full(n + 1, init, v.dtype)
    op.at(out, lab[sel], v)
    return out[1:]


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
        hi = np.ceil(_label_reduce(np.maximum, tops[lab], lm, n) / layer).astype(int)
        base_f = tops[lab] if lab in (WATER, RIVER) else ground
        lo = np.floor((_label_reduce(np.minimum, base_f, lm, n) - depth) / layer).astype(int)
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
    for lab in {comps[i][0] for i in removed}:   # one pass per colour, not per patch
        drop = np.zeros(int(labels[lab].max()) + 1, bool)
        drop[[comps[i][1] for i in removed if comps[i][0] == lab]] = True
        out[lab][drop[labels[lab]]] = False
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


class LayerGrid:
    """The printer's layer boundaries: origin + first layer + k * layer height."""

    def __init__(self, origin: float, first: float, height: float):
        self.o, self.f, self.h = origin, first, height

    def snap(self, z, how=np.round):
        k = how((np.asarray(z, dtype=np.float64) - self.o - self.f) / self.h)
        return np.round(self.o + self.f + np.maximum(k, 0) * self.h, 4)


def _terraced(top: np.ndarray, depth: float, base: float, grid: LayerGrid | None = None) -> np.ndarray:
    """Colour-skin interface: ``top - depth`` rounded down to terraces on the layer grid (about
    0.5 mm apart; flat where the surface is flat, so it triangulates cheaply), kept between the
    base and just under ``top``. On the layer grid, every colour change falls on a layer boundary
    (no half-layer colour slivers)."""
    if grid is None:
        grid = LayerGrid(0.0, 0.0, 0.5)
    q = grid.h * max(1, round(0.5 / grid.h))
    t = grid.o + grid.f + np.floor((top - depth - grid.o - grid.f) / q) * q
    floor = float(grid.snap(base + 1e-6, np.ceil))
    return np.round(np.minimum(np.maximum(t, floor), top - 0.1), 4)


def _disk(r: float) -> np.ndarray:
    k = int(np.ceil(r))
    y, x = np.mgrid[-k:k + 1, -k:k + 1]
    return x * x + y * y <= r * r + 1e-9


def _min_features(cm: dict[int, np.ndarray], r: float) -> dict[int, np.ndarray]:
    """Make colour regions printable at a nozzle width of 2r cells: grow buildings smaller than
    that, merge buildings separated by narrower gaps, and drop tree slivers thinner than it."""
    if r < 0.5:
        return cm
    out = dict(cm)
    disk = _disk(r)
    B = cm.get(BUILDING)
    if B is not None and B.any():
        lab, n = ndimage.label(B, structure=np.ones((3, 3)))
        idx = np.arange(1, n + 1)
        rad = _label_reduce(np.maximum, ndimage.distance_transform_edt(B), lab, n)
        small = idx[rad < r]
        if len(small):
            B = B | ndimage.binary_dilation(np.isin(lab, small), structure=disk)
        out[BUILDING] = ndimage.binary_closing(B, structure=disk) | B
    V = cm.get(VEG)
    if V is not None and V.any():
        out[VEG] = ndimage.binary_opening(V, structure=disk)
    return out


def _extent(parts: list) -> list[float]:
    lo = np.min([p.mesh.vertices.min(axis=0) for p in parts], axis=0)
    hi = np.max([p.mesh.vertices.max(axis=0) for p in parts], axis=0)
    return [round(float(v), 1) for v in hi - lo]


def _luminance(c: str) -> float:
    r, g, b = _hex_rgb(c)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _flat_interface(mask: np.ndarray, top: np.ndarray, depth: float, floor: float) -> np.ndarray:
    """Per connected (dilated) region: min(top) - depth, at least ``floor``."""
    md = ndimage.binary_dilation(mask, iterations=3)
    lab, n = ndimage.label(md, structure=np.ones((3, 3)))
    out = np.maximum(top - depth, floor)
    if n:
        mins = _label_reduce(np.minimum, np.asarray(top, float), lab, n)
        lvl = np.maximum(np.concatenate([[0.0], mins])[lab] - depth, floor)
        out = np.where(lab > 0, lvl, out)
    return np.round(out, 3)


def _label_field(opts: Options, nx: int, ny: int, pad: int, frame_cells: float, cell_mm: float,
                 mm_per_m: float, R: int, C: int, notes: list) -> np.ndarray | None:
    """Name (south side), scale bar and north arrow (north side) on the frame, as a signed
    stroke-distance field in cells."""
    if opts.shape == "circle":
        notes.append("Labels need a straight frame edge; not available for circles.")
        return None
    fw = opts.frame_width_mm
    sw = max(opts.min_feature_mm, 0.45)                 # stroke width
    h = min(fw - sw - 0.8, 0.62 * fw, 7.0)              # cap height inside the ring
    if h < 2.0:
        notes.append("Frame too narrow for labels (needs about 4 mm).")
        return None
    width_mm = nx * cell_mm
    if opts.shape == "hexagon":
        usable = width_mm / 2 - 2 * sw                    # flat bottom edge of the hexagon
    elif opts.shape == "rounded":
        usable = width_mm - 2 * opts.corner_radius_mm - 2 * sw
    else:
        usable = width_mm - 2 * sw
    cx = C * cell_mm / 2
    y_s = (pad - frame_cells / 2) * cell_mm - h / 2        # baseline of the south band
    y_n = (pad + ny + frame_cells / 2) * cell_mm           # centre of the north band
    segs = []
    text = lbl.normalize(opts.label_text or opts.name)
    bar = None
    if opts.label_scale_bar:
        L = lbl.nice_length(0.18 * width_mm / mm_per_m)
        if L > 0:
            bar = (L * mm_per_m, L)
    # fit: [scale bar]  gap  [text] centred on the south side
    th = h
    for _ in range(20):
        tw = lbl.text_width(text, th) if text else 0.0
        bw = 0.0
        if bar:
            bw = bar[0] + th * 0.6 + lbl.text_width(lbl.length_text(bar[1]), th * 0.8)
        need = tw + (bw + 2 * th if bar else 0.0)
        if need <= usable or th < 2.0:
            break
        th *= 0.9
    if need > usable:
        notes.append("Label text is too long for the frame; shortened to fit.")
        while text and lbl.text_width(text, th) + (bw + 2 * th if bar else 0.0) > usable:
            text = text[:-1].rstrip()
    total = tw + (bw + 2 * th if bar else 0.0)
    x = cx - total / 2
    if bar:
        bsegs, bw = lbl.scale_bar(bar[0], bar[1], th, x, y_s + th / 2)
        segs += bsegs
        x += bw + 2 * th
    if text:
        segs += lbl.text_segments(text, th, x, y_s + (h - th) / 2)
    if opts.label_north:
        segs += lbl.north_arrow(h, cx, y_n)
    if not segs:
        return None
    return lbl.stroke_field(segs, sw / 2, (R + 1, C + 1), cell_mm)


# threads meshing parts: each holds large temporaries (a 3 M-cell model peaks at 4.3 GB with 1,
# 6.0 GB with 2, 8.0 GB with 4 workers; build 30 / 25 / 23 s)
from .limits import PART_WORKERS  # noqa: E402 - 1 on small servers (memory), 2 above ~6 GB per job


def build_parts(elev: Elevation, surf: Surface, opts: Options, progress: ProgressFn) -> BuildResult:
    g = elev.grid
    nx, ny = g.nx, g.ny
    cell_mm = opts.size_mm / max(nx, ny)
    mm_per_m = cell_mm / g.step
    zscale = mm_per_m * opts.z_exaggeration
    notes = list(surf.notes)
    col = opts.colors
    lh, fl = opts.layer_height_mm, opts.first_layer_mm
    # base thickness on the layer grid of its own print (first layer + k layers)
    base = float(LayerGrid(0.0, fl, lh).snap(max(opts.base_mm, fl)))
    # layer grid of the multi-colour print (starts on top of a separately printed base)
    lg = LayerGrid(base if opts.base_separate else 0.0, fl, lh)

    # --- padded vertex grid (room for the frame ring and the base lip) and the outline field
    frame_cells = opts.frame_width_mm / cell_mm if opts.frame else 0.0
    use_lip = opts.base_separate and opts.base_lip
    lip_in = frame_cells + opts.lip_clearance_mm / cell_mm
    lip_out = lip_in + opts.lip_width_mm / cell_mm
    reach = max(frame_cells, lip_out if use_lip else 0.0)
    pad = int(np.ceil(reach)) + 2 if (opts.frame or use_lip) else 0
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
    # land meets water at most this steep, ending one uniform edge above the water surface
    shore_k = (float(np.tan(np.radians(opts.shore_slope_deg))) * cell_mm
               if opts.shore_slope_deg < 89.5 else 0.0)
    shore_wall = max(lh, opts.water_recess_mm)

    def zmm(h: np.ndarray) -> np.ndarray:
        return np.minimum((P(h) - hmin) * zscale + base + opts.floor_mm, zcap)

    S_mm = np.round(zmm(surf.surface), 3)

    # --- triangulation follows ridges of the printed surface
    def diagonals(Z: np.ndarray) -> np.ndarray:
        a00, a01, a10, a11 = Z[:-1, :-1], Z[:-1, 1:], Z[1:, :-1], Z[1:, 1:]
        return np.abs(a00 - a11) <= np.abs(a01 - a10)

    min_cells = max(1, int(round(opts.min_region_mm2 / (cell_mm * cell_mm))))
    shore_sigma = min(6.0, max(0.7, 0.8 * opts.min_feature_mm / cell_mm))
    diag = diagonals(S_mm)
    wet_cut = opts.color_mode == "landcover" or (opts.color_mode == "elevation" and opts.elevation_water)
    if shore_k > 0 and wet_cut:
        # the shore ramp reshapes cliffs: choose the cell diagonals along the ramped surface
        # (approximate: the final water masks differ slightly)
        ind0 = _water_indicator(P(surf.masks[WATER]), P(surf.masks[RIVER]), min_cells, shore_sigma)
        if ind0 is not None:
            w0 = np.maximum(zmm(surf.level) - opts.water_recess_mm, lowest)
            diag = diagonals(_shore_cap(diag, ind0, w0, shore_wall, shore_k, S_mm))
    part = Partition(R, C, diag)
    progress(0.05, "Cutting outline")
    if opts.frame:
        part.step(shape_field(opts.shape, R, C, nx, ny, corner_cells, grow=frame_cells), 0.0, below=DROPPED)
        part.step(f_model, 0.0, below=FRAME)
    elif opts.shape != "rectangle" or pad:
        part.step(f_model, 0.0, below=DROPPED)
    # the base plate spans every colour region: build it from the outline cuts only, so later
    # colour cuts cannot leave T-junctions inside it
    outline = part.copy()
    outline.finish(GROUND)
    land = outline

    defs: list[tuple[str, str, str, set, FieldSpec, FieldSpec]] = []
    color_changes: list[dict] = []
    flat = lambda a: np.ascontiguousarray(a, dtype=np.float64).ravel()  # noqa: E731

    # --- buildings and bridge decks: regions of their own, cut along smooth outlines, each with its
    # own top (roof / deck, continued past the outline so the cut takes its height). As part of one
    # height surface they dropped to the ground across one grid cell along the staircase edge of
    # their raster masks: sloped sawtooth walls that cost print time.
    bld_m = P(surf.masks[BUILDING])
    if bld_m.any():
        bld_m = _min_features({BUILDING: bld_m}, opts.min_feature_mm / cell_mm / 2)[BUILDING]
    brg_m = P(surf.bridge_mask)
    roof_mm, deck_mm = np.round(zmm(surf.roof), 3), np.round(zmm(surf.bridge), 3)

    def terrain_mm(over_water: bool) -> np.ndarray:
        """The surface without buildings and bridge decks (trees stay in a full surface);
        ``over_water``: the land continued over the water (``Surface.ground_land``)."""
        g = surf.ground_land if over_water else surf.ground
        if opts.surface == "full":
            g = np.where(surf.masks[VEG], np.maximum(surf.canopy, g), g)
        return np.round(zmm(g), 3)

    def cut_structures(p: Partition) -> list[int]:
        labs = []
        for lab, m, i in ((BUILDING, bld_m, 2), (BRIDGE, brg_m, 5)):
            if m.any():
                p.step(_indicator(m, min_cells), 0.5 + 0.0173 * i, above=lab)
                labs.append(lab)
        return [lab for lab in labs if (p.tri_label == lab).any() or (p.p_label == lab).any()]

    if opts.color_mode == "single":
        T = terrain_mm(False)
        labs = cut_structures(part)
        part.finish(MODEL)
        defs.append(("model", "Model", col.single, {MODEL}, FieldSpec(flat(T)), FieldSpec(0.0)))
        if BUILDING in labs:
            defs.append(("model", "Buildings", col.single, {BUILDING},
                         FieldSpec(flat(np.maximum(roof_mm, T))), FieldSpec(0.0)))
        if BRIDGE in labs:
            defs.append(("model", "Bridges", col.single, {BRIDGE},
                         FieldSpec(flat(np.maximum(deck_mm, T))), FieldSpec(0.0)))
        default, label_color = MODEL, {MODEL: col.single, BUILDING: col.single, BRIDGE: col.single}
    elif opts.color_mode == "landcover":
        G_mm = np.round(zmm(surf.ground_land), 3)
        canopy_mm = np.round(zmm(surf.canopy), 3)
        # Bridge decks are a region of their own inside the roads, cut along a smooth outline:
        # their side walls are clean vertical cuts. As part of the roads (one height field for
        # both), a deck dropped to the ground across one grid cell along the staircase edge of
        # its raster mask - steep sawtooth walls that cost print time. The deck height field
        # continues past the deck (``Surface.bridge``), so the outline takes the deck's height.
        bridge_mm, bridge_m = deck_mm, brg_m

        def land_fields(G: np.ndarray) -> dict:
            return {GROUND: G, BUILDING: np.maximum(roof_mm, G),
                    VEG: np.maximum(canopy_mm, G) if opts.surface == "full" else G,
                    ROAD: np.maximum(G + opts.road_emboss_mm, lowest),
                    BRIDGE: np.maximum(np.maximum(bridge_mm, G) + opts.road_emboss_mm, lowest)}

        fields = land_fields(G_mm)
        fields[WATER] = np.maximum(np.round(zmm(surf.level), 3) - opts.water_recess_mm, lowest)
        fields[RIVER] = fields[WATER]
        names = {GROUND: "Ground", BUILDING: "Buildings", WATER: "Water", ROAD: "Roads", VEG: "Trees",
                 RIVER: "Rivers", BRIDGE: "Bridges"}
        label_color = {GROUND: col.ground, BUILDING: col.buildings, WATER: col.water, RIVER: col.water,
                       ROAD: col.roads, BRIDGE: col.roads,
                       VEG: col.vegetation}
        cmasks = {lab: P(surf.masks[lab]) for lab in (BUILDING, ROAD, WATER, RIVER, VEG)}
        cmasks = _min_features(cmasks, opts.min_feature_mm / cell_mm / 2)
        # flat lakes / sea: water surface on a layer boundary (rivers keep their slope)
        wet = cmasks[WATER] | cmasks[RIVER]
        if wet.any():
            _, (wi, wj) = ndimage.distance_transform_edt(~wet, return_indices=True)
            lake_owned = cmasks[WATER][wi, wj]
            fields[WATER] = np.maximum(np.where(lake_owned, lg.snap(fields[WATER]), fields[WATER]), lowest)
            fields[RIVER] = fields[WATER]
        if opts.color_depth_mm > 0 and opts.swap_patch_budget_mm2 > 0:
            tops = {lab: fields[lab] for lab in cmasks}
            tops[ROAD] = np.where(bridge_m, fields[BRIDGE], fields[ROAD])
            keep = bridge_m | P(surf.river) if surf.river is not None else bridge_m
            cmasks, n_rm, a_rm, saved = _drop_swap_patches(
                cmasks, tops, G_mm, opts.color_depth_mm, cell_mm, opts.swap_patch_budget_mm2, keep=keep)
            if n_rm:
                notes.append(f"Recoloured {n_rm} small patch(es) ({a_rm:.0f} mm2) that each forced their colour "
                             f"into extra layers: ~{saved} fewer filament changes.")
        progress(0.1, "Cutting colour regions")
        # Water is cut first, so "land" is exactly the complement of the water outline. A second
        # partition with only the outline + water cut then gives one seamless ground-coloured body
        # under all land colours (no walls along every colour boundary). LiDAR bridges were
        # already removed from the water mask, so they survive as roads.
        land = part.copy()
        # all water (lakes, sea, rivers) is cut as ONE region, then divided into lake / river
        # inside it - separate outlines would leave slivers of land between a river and LiDAR
        # water patches or a lake it runs into
        ind = _cut_water(part, cmasks[WATER], cmasks[RIVER], min_cells, land=land, sigma=shore_sigma)
        if ind is not None and shore_k > 0:
            # on the water side G_mm drops below the water: only the shore edge interpolates it
            G_mm = _shore_cap(diag, ind, fields[WATER], shore_wall, shore_k, G_mm)
            fields.update(land_fields(G_mm))
        for i, lab in enumerate((BUILDING, ROAD, VEG), start=2):  # priority order
            m = cmasks[lab]
            if m.any():
                # distinct thresholds: two class outlines never cross a grid edge at the very
                # same point (which would pinch the region between them to a single point)
                sigma = 1.2 if lab == VEG else 0.7  # tree canopies are ragged: smoother outline
                part.step(_indicator(m, min_cells, sigma), 0.5 + 0.0173 * i, above=lab)
        if bridge_m.any() and cmasks[ROAD].any():
            # decks inside the roads (the road mask holds them): no slivers where deck meets road
            part.step(_indicator(bridge_m, min_cells), 0.5, above=BRIDGE, only=ROAD)
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
            I = FieldSpec(flat(_terraced(np.maximum(G_mm, lowest) + min(0.0, opts.road_emboss_mm), d, base, lg)))
            # one ground-coloured body under all land (outline + water cut only)
            defs.append(("body", "Body", col.ground, {GROUND}, I, FieldSpec(base)))
        for lab in (GROUND, BUILDING, WATER, RIVER, ROAD, BRIDGE, VEG):
            if lab not in present:
                continue
            T = flat(fields[lab])
            if d <= 0:  # legacy: full-height colour columns
                defs.append((names[lab].lower(), names[lab], label_color[lab], {lab}, FieldSpec(T), FieldSpec(base)))
            elif lab in (WATER, RIVER):
                # water colour cap (flat on lakes, following the profile on rivers) on a ground body
                body_top = FieldSpec(flat(_terraced(fields[lab], d, base, lg)))
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
        lo_mm = [base] + [float(lg.snap(min((t - hmin) * zscale + base + opts.floor_mm, zcap))) for t in th]
        for k in range(1, len(lo_mm)):  # snapped thresholds stay strictly increasing
            lo_mm[k] = max(lo_mm[k], lo_mm[k - 1] + lh)
        wmask = surf.masks[WATER] | surf.masks[RIVER]
        water_sep = opts.elevation_water and wmask.any()
        if water_sep:  # flat, recessed water in its own colour; bands cover the land only
            progress(0.08, "Cutting water")
            ind = _cut_water(part, P(surf.masks[WATER]), P(surf.masks[RIVER]), min_cells, sigma=shore_sigma)
            wtop = np.maximum(np.round(zmm(surf.level), 3) - opts.water_recess_mm, lowest)
            wet_e = P(surf.masks[WATER] | surf.masks[RIVER])
            _, (wi, wj) = ndimage.distance_transform_edt(~wet_e, return_indices=True)
            wtop = np.maximum(np.where(P(surf.masks[WATER])[wi, wj], lg.snap(wtop), wtop), lowest)
        # with separate water the bands only cover land: use the land surface continued over
        # the water so shores are clean walls (see Surface.surface_land)
        Sb_mm = terrain_mm(water_sep)
        if water_sep and ind is not None and shore_k > 0:
            Sb_mm = _shore_cap(diag, ind, wtop, shore_wall, shore_k, Sb_mm)
        # buildings and decks: their own regions, banded by their own tops (band 0 keeps the label)
        labs = cut_structures(part)
        tops = {BAND0: Sb_mm}
        firsts = {BAND0: BAND0}
        if BUILDING in labs:
            tops[BUILDING], firsts[BUILDING] = np.maximum(roof_mm, Sb_mm), BLD_BAND0
        if BRIDGE in labs:
            tops[BRIDGE], firsts[BRIDGE] = np.maximum(deck_mm, Sb_mm), BRG_BAND0
        # a surface lying exactly on a band floor (heights are rounded to 1 um, floors snapped to
        # the layer grid; quantised models make whole plateaus tie) would give that band zero
        # thickness inside its region: top and floor touching along an edge, a non-manifold pinch.
        # Those cells go to the band below: 0.5 um down, so the contour still crosses a third of
        # the way along the next edge (no slivers).
        for key, t in tops.items():
            for lo in lo_mm[1:]:
                t = np.where(np.abs(t - lo) < 2.5e-4, lo - 5e-4, t)
            tops[key] = t
        Sb_mm = tops[BAND0]
        progress(0.1, "Cutting elevation contours")
        for k in range(nb - 1, 0, -1):
            part.step(flat(Sb_mm), lo_mm[k], above=BAND0 + k)
            for lab in labs:
                part.step(flat(tops[lab]), lo_mm[k], above=firsts[lab] + k, only=lab)
        part.finish(BAND0)
        default = BAND0
        label_color = {BAND0 + k: bands[k] for k in range(nb)}
        for lab in labs:
            label_color.update({lab: bands[0]} | {firsts[lab] + k: bands[k] for k in range(1, nb)})

        def band_labels(lab: int, k: int) -> set:
            """Labels of the pieces at band ``k`` or above (band 0 of a structure is its own label)."""
            return {lab if j == 0 else firsts[lab] + j for j in range(k, nb)}
        present_w = [lab for lab in (WATER, RIVER)
                     if water_sep and ((part.tri_label == lab).any() or (part.p_label == lab).any())]
        if present_w:
            d = opts.color_depth_mm
            for lab in present_w:
                nm = "Water" if lab == WATER else "Rivers"
                label_color[lab] = col.water
                if d > 0:
                    body = FieldSpec(flat(_terraced(wtop, d, base, lg)))
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
            defs.append(("band", lbl, bands[k], band_labels(BAND0, k),
                         FieldSpec(flat(Sb_mm), hi=hi), FieldSpec(lo_mm[k])))
            for lab, what in ((BUILDING, "buildings"), (BRIDGE, "bridges")):
                if lab in labs:
                    defs.append(("band", f"{lbl}: {what}", bands[k], band_labels(lab, k),
                                 FieldSpec(flat(tops[lab]), hi=hi), FieldSpec(lo_mm[k])))
            if k > 0:
                color_changes.append({"z_mm": round(lo_mm[k], 2), "color": bands[k],
                                      "elevation_m": round(float(th[k - 1]), 1)})
        if col.base.upper() != bands[0].upper():
            color_changes.insert(0, {"z_mm": round(base, 2), "color": bands[0], "elevation_m": None})
    if opts.frame:
        frame_top = float(lg.snap(base + opts.frame_height_mm)) if opts.frame_height_mm > 0 else base
        has_label = False
        if opts.label:
            field = _label_field(opts, nx, ny, pad, frame_cells, cell_mm, mm_per_m, R, C, notes)
            if field is not None:
                outline.step(field.ravel(), 0.0, above=LABEL, only=FRAME)
                has_label = (outline.tri_label == LABEL).any() or (outline.p_label == LABEL).any()
        defs.append(("frame", "Frame", col.frame, {FRAME}, FieldSpec(frame_top), FieldSpec(0.0)))
        if has_label:
            ltop = float(lg.snap(frame_top + opts.label_emboss_mm))
            defs.append(("label", "Label", col.label, {LABEL}, FieldSpec(ltop), FieldSpec(frame_top)))
            defs.append(("lfill", "Label (body)", col.frame, {LABEL}, FieldSpec(frame_top), FieldSpec(0.0)))
        label_color[FRAME] = col.frame
    elif opts.label:
        notes.append("Labels need the frame (enable Frame / border).")

    # --- base plate as its own print: drop it from the model, which then starts on top of it
    base_defs: list = []
    base_src = outline
    if opts.base_separate:
        defs = [dd for dd in defs if dd[0] != "base"]
        if use_lip:
            # tray: the base reaches past the model (+frame) by clearance + lip, and a raised lip
            # around it locates the model without pins or glue jigs
            basep = Partition(R, C, diag)
            basep.step(shape_field(opts.shape, R, C, nx, ny, corner_cells, grow=lip_out), 0.0, below=DROPPED)
            basep.step(shape_field(opts.shape, R, C, nx, ny, corner_cells, grow=lip_in), 0.0, below=LIP)
            basep.finish(GROUND)
            base_src = basep
            lip_top = float(LayerGrid(0.0, fl, lh).snap(base + opts.lip_height_mm))
            base_defs.append(("base", "Base plate", col.base, {GROUND}, FieldSpec(base), FieldSpec(0.0)))
            base_defs.append(("base", "Base lip", col.base, {LIP}, FieldSpec(lip_top), FieldSpec(0.0)))
        else:
            base_defs.append(("base", "Base plate", col.base,
                              {GROUND, FRAME, LABEL} if opts.frame else {GROUND}, FieldSpec(base), FieldSpec(0.0)))
        sep = []
        for cat, name, colr, labels, top, bottom in defs:
            if cat in ("model", "frame", "lfill"):  # these used to reach down to the bed
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

    def staggered(e: np.ndarray, size: int) -> np.ndarray:
        # base seams half a tile off the model seams: every model seam sits mid-base-tile
        if len(e) <= 2:
            return e
        return np.concatenate([[0], ((e[:-1] + e[1:]) / 2).round().astype(int), [size]])

    bre_, bce_ = staggered(re_, R), staggered(ce_, C)
    ntiles = opts.tiles_x * opts.tiles_y
    multi = ntiles > 1
    bmulti = (len(bre_) - 1) * (len(bce_) - 1) > 1
    done = 0
    total = ntiles * len(defs) + (len(bre_) - 1) * (len(bce_) - 1) * len(base_defs)
    sources = {"base": base_src, "body": land, "frame": outline, "label": outline, "lfill": outline}
    jobs = [(parts, defs, re_, ce_, multi, "Tile"), (base_parts, base_defs, bre_, bce_, bmulti, "Base tile")]
    tasks = []
    for target, dlist, rows_e, cols_e, is_multi, tword in jobs:
        nty, ntx = len(rows_e) - 1, len(cols_e) - 1
        for ty in range(nty):
            for tx in range(ntx):
                win = (rows_e[ty], rows_e[ty + 1], cols_e[tx], cols_e[tx + 1])
                tname = f"{tword} {ty * ntx + tx + 1}" if is_multi else opts.name
                for cat, name, color, labels, top, bottom in dlist:
                    tasks.append((target, is_multi, tname, cat, name, color, sources.get(cat, part),
                                  (labels, top, bottom, win, cell_mm, tx * opts.tile_gap_mm, ty * opts.tile_gap_mm)))
    # parts are independent and only read the partitions; NumPy's sorts / searches release the GIL,
    # so they mesh in parallel (bounded: each part's temporaries are large). Results keep task order.
    from concurrent.futures import ThreadPoolExecutor, as_completed
    meshes: list = [None] * len(tasks)
    with ThreadPoolExecutor(max_workers=max(1, min(PART_WORKERS, len(tasks)))) as ex:
        futs = {ex.submit(t[6].build_part, *t[7], simplify=opts.simplify_mm): i for i, t in enumerate(tasks)}
        for fut in as_completed(futs):
            i = futs[fut]
            meshes[i] = fut.result()
            done += 1
            _, is_multi, tname, _, name = tasks[i][:5]
            progress(0.2 + 0.8 * done / total, f"Meshing {name}" + (f" ({tname})" if is_multi else ""))
    for (target, is_multi, tname, cat, name, color, _src, _args), mesh in zip(tasks, meshes):
        if mesh is not None and len(mesh.triangles):
            if base_defs and target is parts:
                mesh = mesh.translate(0, 0, -base)  # sits on the bed; the base is its own print
            target.append(Part(name if not is_multi else f"{tname} - {name}", color.upper(), mesh, tname, cat))

    # --- filaments
    colors_order: list[str] = []
    for p in parts:
        if p.color not in colors_order:
            colors_order.append(p.color)
    # light -> dark: slicers that change filament in slot order then flush light into dark,
    # which needs the least purge
    colors_order.sort(key=_luminance, reverse=True)
    filaments = [{"slot": i + 1, "color": c,
                  "used_by": sorted({p.name.split(" - ")[-1] for p in parts if p.color == c})}
                 for i, c in enumerate(colors_order)]

    vmax = max(float(p.mesh.vertices[:, 2].max()) for p in parts)
    swaps, max_colors = _layer_color_stats(parts, vmax)
    stats = {
        "size_mm": _extent(parts),
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
    cl = part.cell_labels(default)
    rgb = np.full((R, C, 3), 255, np.uint8)
    for lab, colr in label_color.items():
        rgb[cl == lab] = _hex_rgb(colr)
    rgb[cl == DROPPED] = 255
    zthumb = S_mm.copy()
    if opts.frame:
        zthumb = np.where(f_model >= 0, zthumb, base + opts.frame_height_mm)
    stats["layer_mm"] = [fl, lh]
    tips = [f"Slice with first layer {fl:g} mm and layer height {lh:g} mm: colour boundaries, water "
            "surfaces and band heights are aligned to that layer grid.",
            "Filament slots are ordered light to dark; when the slicer changes in slot order it flushes "
            "light into dark, which needs the least purge.",
            "Enable 'Flush into objects' / 'Flush into infill' (Bambu Studio, OrcaSlicer) or 'Purge into "
            "infill' (PrusaSlicer) so purged filament ends up inside the model instead of the waste chute."]
    if base_parts:
        stats["base_plate_mm"] = _extent(base_parts)
        notes.append(f"Base plate ({base:g} mm, {col.base}) is a separate file; the model starts on top of it.")
        if use_lip:
            tips.append(f"Base plate: the {opts.lip_height_mm:g} mm lip locates the model with "
                        f"{opts.lip_clearance_mm:g} mm clearance - print both flat, drop the model in, glue.")
        if len(bre_) > 2 or len(bce_) > 2:
            tips.append("Base tiles are split half a tile off the model tiles, so every model seam is bridged "
                        "by a base tile - glue the base first, then the model tiles on top.")
    return BuildResult(parts, stats, filaments, color_changes, notes, cell_rgb=rgb, z_mm=zthumb,
                       base_parts=base_parts or None, tips=tips)
