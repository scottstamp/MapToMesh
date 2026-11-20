"""Elevation providers by region. Importing this package registers every provider (DEM providers
with ``maptomesh.dem.register``, point-cloud providers with ``maptomesh.pointcloud.register_pointcloud``).

A region module that fails to import (e.g. a missing optional dependency) is skipped with a warning
instead of breaking the app; ``IMPORT_ERRORS`` records why.
"""

from __future__ import annotations

import importlib
import logging

MODULES = ("global_", "north_america", "france_benelux_ch", "uk_ie_south_eu", "central_eu", "nordic",
           "oceania", "asia", "mapterhorn")
IMPORT_ERRORS: dict[str, str] = {}

for _m in MODULES:
    try:
        importlib.import_module(f"{__name__}.{_m}")
    except ModuleNotFoundError as _e:
        if _e.name != f"{__name__}.{_m}":  # the module exists but one of its imports doesn't
            IMPORT_ERRORS[_m] = repr(_e)
            logging.getLogger(__name__).warning("elevation providers %s not loaded: %r", _m, _e)
    except Exception as _e:  # noqa: BLE001
        IMPORT_ERRORS[_m] = repr(_e)
        logging.getLogger(__name__).warning("elevation providers %s not loaded: %r", _m, _e)
