"""Building geometry beyond OpenStreetMap footprints, for the parts of an area whose elevation data
is bare earth (no point cloud, no surface model) and so shows no buildings.

* ``plateau``: Project PLATEAU 3D city models (Japan): surveyed roof shapes (LoD2) or blocks at
  the measured height (LoD1).
* ``overture``: Overture Maps buildings: heights where OpenStreetMap has none (Microsoft ML
  heights in North America, ...) and footprints from datasets other than OSM.

``compute_surface`` (``model._bare_earth_buildings``) merges them with the OSM footprints. Where
LiDAR or a surface model covers the ground, the measured surface is used and nothing is fetched.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


class TooLarge(Exception):
    """The area needs more of a source's data than a job may download."""


class Cancelled(RuntimeError):
    """The job was cancelled while a building source was being read (always propagates)."""

    def __init__(self, msg: str = "Cancelled"):
        super().__init__(msg)


MIN_BARE_M2 = 20000.0          # less bare-earth area than this: OSM heights are enough


def fetch_buildings(opts, elev, aoi, grid, progress=lambda f, m: None, cancel=None) -> tuple[dict, list[str], list[str]]:
    """({"plateau": ..., "overture": ..., "building_notes": [...]} to add to the OSM layers,
    attribution lines, warnings)."""
    if not opts.show_buildings:
        return {}, [], []
    known = elev.arrays.get("veg_known")
    if known is None:
        return {}, [], []
    bare = ~np.asarray(known, bool)
    if bare.sum() * grid.step ** 2 < MIN_BARE_M2:
        return {}, [], []
    out: dict = {}
    credits: list[str] = []
    warnings: list[str] = []
    notes: list[str] = []
    try:
        from . import plateau
        r = plateau.fetch(aoi, grid, cancel, lambda f, m: progress(0.6 * f, m))
        if r is not None:
            out["plateau"] = r
            credits.append(r["credit"])
    except TooLarge as e:        # OSM / Overture heights instead
        notes.append(f"PLATEAU 3D buildings not used ({e}): OpenStreetMap heights instead.")
    except Cancelled:
        raise
    except Exception as e:  # noqa: BLE001 - building sources are optional
        if cancel is not None and cancel.is_set():
            raise Cancelled() from e
        log.warning("PLATEAU failed: %r", e)
        warnings.append(f"Warning: PLATEAU 3D buildings unavailable ({e}); using OpenStreetMap heights")
    if "plateau" not in out:
        try:
            from . import overture
            r = overture.fetch(aoi, grid, bare, cancel, lambda f, m: progress(0.6 + 0.4 * f, m))
            if r is not None and r["mask"].any():
                out["overture"] = r
                credits.append(overture.CREDIT)
        except TooLarge as e:
            notes.append(f"Overture building heights not used ({e}).")
        except Cancelled:
            raise
        except Exception as e:  # noqa: BLE001
            if cancel is not None and cancel.is_set():
                raise Cancelled() from e
            log.warning("Overture failed: %r", e)
            warnings.append(f"Warning: Overture buildings unavailable ({e}); using OpenStreetMap heights only")
    if notes:
        out["building_notes"] = notes
    return out, credits, warnings
