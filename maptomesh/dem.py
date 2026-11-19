"""Elevation-model (DEM) providers and the best-available composite.

A provider puts one elevation model (a national LiDAR DTM, a regional service, a global DEM) onto
the AOI vertex grid. ``best_dems`` asks every provider that covers the area, highest resolution
first, and fills only what is still missing: a 0.5 m national DTM where it exists, a 1-5 m regional
model around it, Copernicus 30 m everywhere else. Each fallback is shifted by its median offset
to what is already filled (vertical datums differ by up to a metre or two), so seams don't step.

Providers live in ``maptomesh.providers`` (one module per region) and register themselves with
``register``. A provider returns a DTM (bare earth) and, where it has one, a DSM (first surface).
DSM-only models (Copernicus, ArcticDEM) set ``dsm_only``: their heights include trees and
buildings and are used as ground.
"""

from __future__ import annotations

import dataclasses
import math
import threading
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .geo import AOI, Grid

ProgressFn = Callable[[float, str], None]


@dataclass
class DemArrays:
    dtm: np.ndarray                 # vertex grid, NaN = no data
    dsm: np.ndarray | None = None   # same, None if the provider has no surface model
    detail: str = ""                # e.g. the items / tiles used, for the job log
    # licence restrictions of the data actually used (e.g. one collection of a provider is
    # non-commercial); shown to the user as a warning with the model
    licence_notes: list[str] = field(default_factory=list)


class DemProvider:
    """Base class. Subclasses set the attributes and implement ``fetch``."""
    key: str = ""
    label: str = ""              # shown to users, e.g. "IGN RGE ALTI 1 m (France)"
    res_m: float = 30.0          # native ground resolution
    has_dsm: bool = False
    dsm_only: bool = False       # heights are a surface model (trees / buildings included)
    attribution: str = ""        # required credit line
    licence: str = ""            # licence name + URL
    regions: tuple = ()          # lon/lat boxes (w, s, e, n); empty = global
    max_area_km2: float = 2500.0  # beyond this the provider is skipped (next fallback is used)
    sample: tuple = ()           # (lat, lon, size_m) for tests/probe_sources.py
    licence_note: str = ""       # restriction that applies whenever this provider's data is used

    def covers(self, bbox: tuple[float, float, float, float]) -> bool:
        if not self.regions:
            return True
        w, s, e, n = bbox
        return any(w <= rx1 and e >= rx0 and s <= ry1 and n >= ry0 for rx0, ry0, rx1, ry1 in self.regions)

    def fetch(self, aoi: AOI, grid: Grid, progress: ProgressFn, cancel: threading.Event) -> DemArrays:
        raise NotImplementedError


PROVIDERS: list[DemProvider] = []


def register(p: DemProvider) -> DemProvider:
    PROVIDERS.append(p)
    return p


def ordered(providers=None) -> list[DemProvider]:
    """Highest resolution first; DTM+DSM before DTM-only before surface-only at equal resolution."""
    ps = providers if providers is not None else PROVIDERS
    return sorted(ps, key=lambda p: (p.res_m, p.dsm_only, not p.has_dsm))


@dataclass
class Composite:
    dtm: np.ndarray
    dsm: np.ndarray
    source: np.ndarray            # index into ``used`` per vertex, -1 = none
    used: list[dict] = field(default_factory=list)   # key, label, res_m, share, attribution, licence, offset_m
    errors: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


def _offset(ref: np.ndarray, vals: np.ndarray, step: float, surface_only: bool = False, min_n: int = 30) -> float:
    """Median height difference ``ref - vals`` where both exist (datum shift of a fallback).

    A surface-only fallback (Copernicus, ArcticDEM) sits on top of trees and buildings, so against
    bare-earth ``ref`` a plain median is biased by the canopy (-2.6 m in Hong Kong where the datum
    shift is +1.3 m). There, only cells where the fallback is at its own local minimum (open
    ground within ~90 m) are compared."""
    from scipy import ndimage

    both = np.isfinite(ref) & np.isfinite(vals)
    if both.sum() < min_n:
        return 0.0
    sel = both
    if surface_only:
        # open, flat ground: the fallback is at its own local minimum and the reference is level
        k = max(3, int(round(90.0 / max(step, 1e-6))) | 1)
        low = vals - ndimage.minimum_filter(np.where(np.isfinite(vals), vals, np.inf), size=k, mode="nearest")
        gy, gx = np.gradient(np.where(np.isfinite(ref), ref, 0.0), step)
        flat = np.hypot(gx, gy) < 0.08
        ground = both & (low < 1.0) & flat
        sel = ground if ground.sum() >= min_n else both
    d = ref[sel] - vals[sel]
    if surface_only and sel is both:   # no open ground found: canopy biases the median low
        off = float(np.percentile(d, 75))
    else:
        off = float(np.median(d))
    if abs(off) >= 50:   # a huge offset means a mismatch, not a datum shift
        return 0.0
    if surface_only:
        # in dense cities even the low points of a 30 m surface model are rooftops (Hong Kong
        # gave +4.4 m for a ~+1.2 m datum shift); real datum shifts to EGM2008 sea level are
        # within ~1.5 m, so a surface-only fallback (mostly filling sea) is never shifted more
        off = float(np.clip(off, -2.0, 2.0))
    return off


_WANT_DSM: ContextVar[bool] = ContextVar("maptomesh_want_dsm", default=True)


def want_dsm() -> bool:
    """False while filling a ground-only job (bare terrain not coloured by land cover): providers
    then skip their surface model (DSM), often half of the download. Read it in ``fetch`` itself:
    context variables don't follow into worker threads."""
    return _WANT_DSM.get()


def best_dems(aoi: AOI, grid: Grid, need: np.ndarray | None = None, ref: np.ndarray | None = None,
              progress: ProgressFn = lambda f, m: None, cancel: threading.Event | None = None,
              providers=None, min_share: float = 0.0005, dsm: bool = True) -> Composite:
    """Fill the vertices in ``need`` (all if None) from the best providers covering the AOI.

    ``ref`` is higher-priority elevation already known elsewhere (e.g. LiDAR ground): fallbacks
    are offset to it where both exist. ``dsm=False``: providers skip surface models."""
    token = _WANT_DSM.set(dsm)
    try:
        return _best_dems(aoi, grid, need, ref, progress, cancel, providers, min_share)
    finally:
        _WANT_DSM.reset(token)


def _gap_blocks(remaining: np.ndarray, step: float, block_m: float = 256.0, max_rects: int = 48) -> tuple:
    """Rectangles (r0, r1, c0, c1, vertex indices, half-open) covering the ``remaining`` cells in
    blocks of ``block_m`` plus one block of margin (so the fill overlaps data already placed, for
    the datum offset). () when they'd cover most of the grid anyway, or be too many."""
    from scipy import ndimage
    H, W = remaining.shape
    b = max(8, int(round(block_m / step)))
    nby, nbx = -(-H // b), -(-W // b)
    pad = np.zeros((nby * b, nbx * b), bool)
    pad[:H, :W] = remaining
    blk = ndimage.binary_dilation(pad.reshape(nby, b, nbx, b).any(axis=(1, 3)), structure=np.ones((3, 3), bool))
    if blk.mean() > 0.5:
        return ()
    rects, open_runs = [], {}
    for by in range(nby + 1):   # runs of blocks per row, merged downwards while the span repeats
        runs = set()
        if by < nby:
            row = np.concatenate([[False], blk[by], [False]])
            d = np.flatnonzero(np.diff(row.astype(np.int8)))
            runs = {(int(a), int(e)) for a, e in zip(d[::2], d[1::2])}
        for span in [sp for sp in open_runs if sp not in runs]:
            rects.append((open_runs.pop(span), by) + span)
        for span in runs:
            open_runs.setdefault(span, by)
        if len(rects) > max_rects:
            return ()
    out = []
    for rb0, rb1, cb0, cb1 in rects:
        r1, c1 = min(rb1 * b + 1, H), min(cb1 * b + 1, W)
        out.append((min(rb0 * b, r1 - 2), r1, min(cb0 * b, c1 - 2), c1))   # at least 2 x 2 vertices
    return tuple(out)


def _best_dems(aoi: AOI, grid: Grid, need, ref, progress: ProgressFn, cancel, providers,
               min_share: float) -> Composite:
    from .pointcloud import Cancelled

    cancel = cancel or threading.Event()
    shape = grid.vshape
    need = np.ones(shape, bool) if need is None else need.astype(bool)
    dtm = np.full(shape, np.nan)
    dsm = np.full(shape, np.nan)
    source = np.full(shape, -1, np.int16)
    ref_all = np.full(shape, np.nan) if ref is None else ref.astype(np.float64).copy()
    out = Composite(dtm, dsm, source)
    bbox = aoi.lonlat_bbox(margin=grid.step * 2)
    area_km2 = aoi.width_m * aoi.height_m / 1e6
    cands = [p for p in ordered(providers) if p.covers(bbox)]
    for k, p in enumerate(cands):
        remaining = need & np.isnan(dtm)
        if remaining.mean() < min_share or not remaining.any():
            break
        if cancel.is_set():
            raise Cancelled()
        if area_km2 > p.max_area_km2:
            out.skipped.append(f"{p.label}: area over {p.max_area_km2:g} km2")
            continue
        progress(k / max(len(cands), 1), f"Elevation model: {p.label}")
        t0 = time.time()
        blocks = _gap_blocks(remaining, grid.step)
        try:
            arr = p.fetch(aoi, dataclasses.replace(grid, read_blocks=blocks) if blocks else grid,
                          lambda f, m: progress((k + f) / max(len(cands), 1), m), cancel)
        except Cancelled:
            raise
        except Exception as e:  # noqa: BLE001 - try the next fallback
            if cancel.is_set():
                raise Cancelled() from e
            out.errors.append(f"{p.label}: {type(e).__name__}: {str(e)[:200]}")
            continue
        d = arr.dtm
        if blocks:   # nothing from outside the rectangles (a provider may fill holes there: no data)
            inside = np.zeros(d.shape, bool)
            for r0, r1, c0, c1 in blocks:
                inside[r0:r1, c0:c1] = True
            d = np.where(inside, d, np.nan)
            if arr.dsm is not None:
                arr.dsm = np.where(inside, arr.dsm, np.nan)
        valid = np.isfinite(d) & (np.abs(d) < 9000)
        d = np.where(valid, d, np.nan)
        take = remaining & valid
        if not take.any():
            continue
        # datum shift to everything better already placed (LiDAR ground, earlier providers)
        placed = np.where(np.isfinite(dtm), dtm, ref_all)
        off = _offset(placed, d, grid.step, surface_only=p.dsm_only)
        idx = len(out.used)
        dtm[take] = d[take] + off
        if arr.dsm is not None:
            s = np.where(np.isfinite(arr.dsm) & (np.abs(arr.dsm) < 9000), arr.dsm, np.nan)
            dsm[take] = s[take] + off
        source[take] = idx
        notes = ([p.licence_note] if p.licence_note else []) + list(arr.licence_notes or [])
        out.used.append({"key": p.key, "label": p.label, "res_m": p.res_m, "attribution": p.attribution,
                         "licence": p.licence, "offset_m": round(off, 2), "dsm": arr.dsm is not None,
                         "surface_only": p.dsm_only, "seconds": round(time.time() - t0, 1),
                         "detail": arr.detail, "licence_notes": notes})
    total = max(need.sum(), 1)
    for i, u in enumerate(out.used):
        u["share"] = float((source == i).sum() / total)
    return out


def describe(cands_bbox: tuple[float, float, float, float], area_km2: float, providers=None) -> list[dict]:
    """Providers that would be tried for an area (cheap: region test only)."""
    out = []
    for p in ordered(providers):
        if p.covers(cands_bbox):
            out.append({"key": p.key, "label": p.label, "res_m": p.res_m,
                        "usable": area_km2 <= p.max_area_km2})
    return out


def utm_epsg(lon: float, lat: float) -> int:
    zone = int(math.floor((lon + 180) / 6)) + 1
    return (32600 if lat >= 0 else 32700) + zone
