"""Offline checks for bridges and OpenStreetMap building heights.

    python tests/test_structures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from maptomesh.model import _osm_bridges, _water_indicator  # noqa: E402
from maptomesh.options import Options  # noqa: E402
from maptomesh.osm import _fill_box  # noqa: E402
from maptomesh.raster import fill_polygon, masked_median  # noqa: E402

OPTS = Options(lat=43.3, lon=-79.8, width_m=1000, height_m=1000)


def test_fill_box():
    """Filling within a polygon's bounding box == filling the whole grid (also partly outside)."""
    rng = np.random.default_rng(1)
    shape = (80, 100)
    for _ in range(500):
        c = rng.uniform(-20, 120, 2)
        n = rng.integers(3, 9)
        ang = np.sort(rng.uniform(0, 2 * np.pi, n))
        rad = rng.uniform(0.3, 15, n)
        parts = [np.column_stack([c[0] + rad * np.cos(ang), c[1] * 0.8 + rad * np.sin(ang)])]
        if rng.random() < 0.3:   # a courtyard
            parts.append(np.column_stack([c[0] + 0.3 * rad * np.cos(ang), c[1] * 0.8 + 0.3 * rad * np.sin(ang)]))
        out = np.zeros(shape, bool)
        box = _fill_box(parts, shape)
        if box is not None:
            out[box[0]] = box[1]
        assert (out == fill_polygon(parts, shape)).all()


def test_masked_median():
    a = np.full((20, 20), 10.0)
    a[10, 10] = 60.0                          # a lamp post
    a[5, 5] = 0.0                             # a gap to the ground
    mask = np.zeros_like(a, bool)
    mask[3:17, 3:17] = True
    out = masked_median(a, mask, mask, 5)
    assert out[10, 10] == 10.0 and out[5, 5] == 10.0
    assert (out[~mask] == a[~mask]).all()


def test_osm_bridge_margins():
    """An OpenStreetMap bridge drawn wider than the classified deck: the margin takes the deck's
    height, not the towers / cables (unclassified returns) or the ground beside it."""
    shape = (40, 60)
    Gf = np.full(shape, 75.0)
    W = np.zeros(shape, bool)
    W[:, 20:40] = True                        # a canal under the bridge
    BRm = np.zeros(shape, bool)
    BRm[18:22, 5:55] = True                   # classified deck, 4 cells wide
    BRv = np.where(BRm, 81.0, np.nan)
    osm_br = np.zeros(shape, bool)
    osm_br[16:24, 5:55] = True                # OSM: 8 cells wide
    U = np.full(shape, np.nan)
    U[16, 5:55:2] = 136.0                     # lift towers / cables along one side
    U[23, 30:40] = 76.0                       # the water / ground next to the deck
    m, v = _osm_bridges(osm_br, BRm, BRv, U, Gf, W, OPTS, [], 2.5)
    assert m[16:24, 5:55].all()
    assert np.allclose(v[16:24, 5:55], 81.0), np.unique(v[16:24, 5:55])


def test_osm_bridge_without_class():
    """Elevation models: deck from the surface heights on the bridge, posts removed, ends on land."""
    shape = (30, 60)
    Gf = np.full(shape, 10.0)
    W = np.zeros(shape, bool)
    W[:, 15:45] = True
    osm_br = np.zeros(shape, bool)
    osm_br[13:17, 5:55] = True
    U = np.full(shape, np.nan)
    U[13:17, 15:45] = 30.0                    # the deck over the water
    U[13, 20] = U[16, 33] = 45.0              # lamp posts
    notes = []
    m, v = _osm_bridges(osm_br, np.zeros(shape, bool), np.full(shape, np.nan), U, Gf, W, OPTS, notes, 2.5)
    assert m[13:17, 5:55].all() and notes
    assert np.allclose(v[13:17, 18:42], 30.0)
    assert v[14, 5] < 15.0                    # the bridge's ends meet the ground


def test_osm_bridge_tower():
    """A lift bridge without a bridge class: its towers (returns well above the deck, a couple of
    cells across) are not the deck - the deck is the lowest surface above the water."""
    shape = (30, 60)
    Gf = np.full(shape, 75.0)
    W = np.zeros(shape, bool)
    W[:, 15:45] = True
    osm_br = np.zeros(shape, bool)
    osm_br[12:17, 5:55] = True
    U = np.full(shape, np.nan)
    U[12:17, 15:45] = 87.0                    # the deck over the canal
    U[12:14, 28:30] = U[15:17, 28:30] = 119.0  # its two towers
    m, v = _osm_bridges(osm_br, np.zeros(shape, bool), np.full(shape, np.nan), U, Gf, W, OPTS, [], 8.0)
    assert np.allclose(v[12:17, 18:42], 87.0), np.unique(np.round(v[12:17, 18:42]))


def test_pier_fragments():
    """A pier about one cell wide leaves no dashed line of specks in the water; a small island and
    a printable breakwater stay."""
    shape = (60, 80)
    wet = np.zeros(shape, bool)
    wet[:, 20:] = True                        # the sea east of the shore
    pier = np.zeros(shape, bool)
    pier[30, 20:70] = True                    # 1 cell wide ...
    for c in range(24, 68, 8):
        pier[29:32, c:c + 3] = True           # ... with wider spots every 8 cells
    wet &= ~pier
    wet[10:13, 20:60] = False                 # a breakwater, 3 cells wide
    wet[48:52, 60:64] = False                 # a small island
    land = _water_indicator(wet, np.zeros(shape, bool), 4, 1.2) < 0.5
    assert not land[28:33, 26:70].any()
    assert land[11, 21:58].all()
    assert land[49:51, 61:63].all()


if __name__ == "__main__":
    for fn in (test_fill_box, test_masked_median, test_osm_bridge_margins, test_osm_bridge_without_class,
               test_osm_bridge_tower, test_pier_fragments):
        fn()
        print(f"{fn.__name__}: ok")
    print("all passed")
