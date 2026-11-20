"""Resource limits for generation jobs.

Two lines of defence:

1. **Before a job starts** (``check``): the grid a job asks for must fit the memory budget, using
   a cost model measured on real jobs (peak memory grows with the number of grid vertices, and
   more for land-cover models - several coloured parts, roads, buildings - than for elevation
   bands or a single colour). Too large: the job is refused with the largest detail that fits.
2. **While it runs** (``server._run_job``): every job runs in its own process; one that runs past
   ``TIMEOUT_S`` or grows past ``JOB_MEMORY_MB`` is killed and reported with advice for retrying.

Point clouds over large areas don't need a refusal: beyond ``config.MAX_TILES`` tiles a job uses
elevation models instead (fast for any area).

Configuration: ``MAPTOMESH_MEMORY_MB`` (memory the jobs may use in total; default 80% of the
machine's RAM), ``MAPTOMESH_JOB_TIMEOUT_S`` (default 300), ``MAPTOMESH_MAX_QUEUE`` (default 8).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field


def _total_ram_mb() -> int:
    try:
        import psutil
        return int(psutil.virtual_memory().total / 1e6)
    except Exception:  # noqa: BLE001
        return 4096


TIMEOUT_S = float(os.environ.get("MAPTOMESH_JOB_TIMEOUT_S", "300"))
TARGET_S = 120.0
MEMORY_MB = int(os.environ.get("MAPTOMESH_MEMORY_MB", "0")) or int(_total_ram_mb() * 0.8)
SERVER_MB = 450            # the server process itself (FastAPI, LiDAR tile index, caches)
# Jobs in flight: one building while the next downloads its data. Downloading point clouds
# needs memory too (~1 GB per million grid vertices), so small servers run one job at a time.
PIPELINE = int(os.environ.get("MAPTOMESH_PIPELINE", "0")) or (2 if MEMORY_MB - SERVER_MB >= 6000 else 1)
# what one job may use: everything but the server's share, less room for a second job in flight
JOB_MEMORY_MB = int(max(800, (MEMORY_MB - SERVER_MB) / (1.6 if PIPELINE > 1 else 1.0)))
MAX_QUEUE = int(os.environ.get("MAPTOMESH_MAX_QUEUE", "8"))
# Threads meshing a model's parts: 2 is ~20% faster but its peak is ~25% higher (land cover at 3 M
# vertices: 4.1 GB against 3.2 GB); small servers use 1. MAPTOMESH_PART_WORKERS overrides.
PART_WORKERS = int(os.environ.get("MAPTOMESH_PART_WORKERS", "0")) or (2 if JOB_MEMORY_MB >= 6000 else 1)

# Peak memory (MB) ~ BASE_MB + per-vertex cost * million grid vertices, the larger of
# - building the model: by colour mode (several coloured parts cost more) and part workers;
# - downloading point clouds (LiDAR jobs only): with classes (buildings, trees, land cover) or
#   ground only (bare terrain in bands / one colour).
# Measured on St. John's jobs at 600-2400 cells, 2026-11-19 (README "Server limits").
BASE_MB = 300.0
BUILD_MB = {1: {"landcover": 1000.0, "elevation": 650.0, "single": 650.0},
            2: {"landcover": 1240.0, "elevation": 900.0, "single": 670.0}}
FETCH_MB = {"classes": 1060.0, "ground": 330.0}
# build + files, seconds per million vertices on the development machine (downloads extra)
SEC_PER_MVERT = {"landcover": 10.0, "elevation": 6.5, "single": 4.0}


def kind_of(opts) -> str:
    return opts.color_mode if opts.color_mode in SEC_PER_MVERT else "landcover"


def _per_mvert(opts) -> float:
    build = BUILD_MB[min(PART_WORKERS, 2)][kind_of(opts)]
    if opts.source != "lidar":
        return build
    return max(build, FETCH_MB["ground" if opts.ground_only else "classes"])


def grid_shape(opts) -> tuple[int, int]:
    """(nx, ny) cells of the job's grid (as ``geo.Grid.for_aoi``)."""
    long_side = max(opts.width_m, opts.height_m)
    step = long_side / opts.grid_cells
    return max(4, round(opts.width_m / step)), max(4, round(opts.height_m / step))


def estimate(opts) -> dict:
    nx, ny = grid_shape(opts)
    mv = (nx + 1) * (ny + 1) / 1e6
    k = kind_of(opts)
    return {"mvertices": round(mv, 2), "memory_mb": round(BASE_MB + _per_mvert(opts) * mv),
            "build_s": round(SEC_PER_MVERT[k] * mv), "kind": k}


def max_cells(opts) -> int:
    """Largest ``grid_cells`` (long side) whose estimate fits the job memory budget (with 10% to
    spare for the estimate's error), capped at the option's own maximum."""
    aspect = min(opts.width_m, opts.height_m) / max(opts.width_m, opts.height_m)
    mv = max(0.05, (JOB_MEMORY_MB * 0.9 - BASE_MB) / _per_mvert(opts))
    return min(2400, int(math.sqrt(mv * 1e6 / max(aspect, 0.05))))


@dataclass
class LimitError(Exception):
    kind: str
    message: str
    advice: list[str] = field(default_factory=list)
    suggest: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return self.message


def check(opts) -> dict:
    """The estimate for ``opts``; raises ``LimitError`` when it doesn't fit."""
    est = estimate(opts)
    cap = max_cells(opts)
    if opts.grid_cells > cap:
        cap = max(150, cap // 50 * 50)      # a value the Detail slider (steps of 50) can take
        tips = [f"Lower Detail to {cap} cells or less (the most these settings allow here)."]
        if opts.color_mode == "landcover":
            tips.append("Elevation bands or a single colour need less memory than land-cover colours.")
        if opts.source == "lidar" and not opts.ground_only:
            tips.append("Bare terrain (no buildings or trees) reads lighter point clouds.")
        tips.append("A smaller area at the same Detail needs the same memory: memory follows the Detail.")
        raise LimitError(
            "too_large",
            f"{opts.grid_cells} cells of detail would need about {est['memory_mb'] / 1000:.1f} GB of memory; "
            f"a model may use {JOB_MEMORY_MB / 1000:.1f} GB on this server.",
            advice=tips, suggest={"grid_cells": cap})
    return est


# Point clouds, cold (nothing cached): ~0.2 s per CanElevation tile at 1000 cells (Toronto 10 km,
# 375 tiles: 86 s; Ottawa 20 km, 441 tiles: 74 s) and ~1 s per km2 of USGS Entwine (Denver 10 km:
# 96 s); finer detail reads deeper octree levels (data ~ cells^2). Over these, a job uses elevation
# models, which take seconds for any area. Keeps the download within ~1.5 minutes.
LIDAR_TILES_AT_1000 = int(os.environ.get("MAPTOMESH_MAX_LIDAR_TILES", "400"))
EPT_KM2_AT_1000 = 100.0


def lidar_caps(opts) -> tuple[int, float]:
    """(max CanElevation tiles, max USGS Entwine km2) for a job: fewer at finer detail."""
    f = (1000.0 / max(opts.grid_cells, 1)) ** 2
    tiles = int(min(LIDAR_TILES_AT_1000, max(100, LIDAR_TILES_AT_1000 * f)))
    km2 = float(min(150.0, max(25.0, EPT_KM2_AT_1000 * f)))
    return tiles, round(km2)


def public() -> dict:
    """The memory model for the UI, which caps its Detail slider the same way (``max_cells``)."""
    return {"memory_mb": JOB_MEMORY_MB, "timeout_s": TIMEOUT_S, "target_s": TARGET_S, "max_queue": MAX_QUEUE,
            "base_mb": BASE_MB, "build_mb": BUILD_MB[min(PART_WORKERS, 2)], "fetch_mb": FETCH_MB}


def advice_for(kind: str, opts) -> tuple[list[str], dict]:
    """Retry advice (and a suggested setting) after a job was stopped for ``kind``."""
    lower = max(100, int(opts.grid_cells * 0.7) // 50 * 50)
    if kind == "timeout":
        tips = [f"Lower Detail (e.g. {lower} cells): time grows with the square of the detail."]
        if opts.source == "lidar":
            tips.append("For large areas choose 'Elevation models only' as the source: point clouds are "
                        "the slowest data to download.")
        if opts.color_mode == "landcover" or opts.surface != "terrain":
            tips.append("Bare terrain with elevation bands builds much faster than land cover with buildings.")
        tips.append("If the server is busy, retrying a little later can help (downloads are cached).")
        return tips, {"grid_cells": lower}
    if kind == "memory":
        tips = [f"Lower Detail (e.g. {lower} cells): memory grows with the square of the detail.",
                "Elevation bands or a single colour need less memory than land-cover colours."]
        return tips, {"grid_cells": lower}
    if kind == "busy":
        return ["Too many models are queued right now. Try again in a minute or two."], {}
    return [], {}
