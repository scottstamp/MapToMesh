"""Option validation limits.

    python tests/test_options.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from maptomesh.options import Colors, Options  # noqa: E402


def _bad(**kw) -> bool:
    try:
        Colors(**kw)
    except ValidationError:
        return True
    return False


def test_band_thresholds_limits():
    assert Colors(band_thresholds_m=[float(i) for i in range(11)]).band_thresholds_m
    assert Colors(band_thresholds_m=None).band_thresholds_m is None
    assert _bad(band_thresholds_m=[float(i) for i in range(12)])
    assert _bad(band_thresholds_m=[0.0] * 300_000)
    assert _bad(band_thresholds_m=[float("nan")])
    assert _bad(band_thresholds_m=[1.0, float("inf")])
    Options(lat=1, lon=1, width_m=1000, height_m=1000, colors={"band_thresholds_m": [5, 10.5]})


if __name__ == "__main__":
    test_band_thresholds_limits()
    print("ok")
