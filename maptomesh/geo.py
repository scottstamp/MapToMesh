"""Area of interest, local projection and the vertex grid used everywhere.

Heights live on a *vertex* grid of shape (ny + 1, nx + 1); row 0 is the
southern edge (y grows north) so it maps directly onto mesh coordinates.
Masks/labels live on the *cell* grid (ny, nx).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from pyproj import CRS, Transformer

SHAPES = ("rectangle", "rounded", "circle", "hexagon")


@dataclass(frozen=True)
class AOI:
    lat: float
    lon: float
    width_m: float
    height_m: float

    def key(self) -> str:
        return f"{self.lat:.6f},{self.lon:.6f},{self.width_m:.1f},{self.height_m:.1f}"

    @cached_property
    def crs(self) -> CRS:
        # Local transverse Mercator: true scale at the AOI centre.
        return CRS.from_proj4(
            f"+proj=tmerc +lat_0={self.lat} +lon_0={self.lon} +k=1 +x_0=0 +y_0=0 "
            "+ellps=GRS80 +units=m +no_defs"
        )

    @cached_property
    def to_lonlat(self) -> Transformer:
        return Transformer.from_crs(self.crs, CRS.from_epsg(4617), always_xy=True)

    def outline_local(self, margin: float = 0.0, n: int = 16) -> np.ndarray:
        """Densified rectangle boundary in local metres (n points per side)."""
        hw, hh = self.width_m / 2 + margin, self.height_m / 2 + margin
        t = np.linspace(-1, 1, n, endpoint=False)
        xs = np.concatenate([t * hw, np.full(n, hw), -t * hw, np.full(n, -hw)])
        ys = np.concatenate([np.full(n, -hh), t * hh, np.full(n, hh), -t * hh])
        return np.column_stack([xs, ys])

    def lonlat_bbox(self, margin: float = 0.0) -> tuple[float, float, float, float]:
        pts = self.outline_local(margin)
        lon, lat = self.to_lonlat.transform(pts[:, 0], pts[:, 1])
        return float(np.min(lon)), float(np.min(lat)), float(np.max(lon)), float(np.max(lat))


@dataclass(frozen=True)
class Grid:
    """Vertex grid in local AOI metres."""
    nx: int  # cells along x
    ny: int  # cells along y
    step: float  # metres per cell
    x0: float  # x of vertex column 0
    y0: float  # y of vertex row 0

    @classmethod
    def for_aoi(cls, aoi: AOI, cells_long_side: int) -> "Grid":
        step = max(aoi.width_m, aoi.height_m) / cells_long_side
        nx = max(4, int(round(aoi.width_m / step)))
        ny = max(4, int(round(aoi.height_m / step)))
        return cls(nx, ny, step, -nx * step / 2, -ny * step / 2)

    @property
    def vshape(self) -> tuple[int, int]:
        return self.ny + 1, self.nx + 1

    @property
    def width_m(self) -> float:
        return self.nx * self.step

    @property
    def height_m(self) -> float:
        return self.ny * self.step

    def key(self) -> str:
        return f"{self.nx}x{self.ny}@{self.step:.4f}"


# ---------------------------------------------------------------- shapes

def shape_mask(shape: str, ny: int, nx: int, w: float, h: float,
               corner_r: float = 0.0, pad: int = 0, grow: float = 0.0) -> np.ndarray:
    """Cell mask (ny + 2*pad, nx + 2*pad) of a centred shape.

    ``w``/``h`` are the shape's size in cell units, ``grow`` offsets the outline
    outward (cells) - used to build the frame ring.
    """
    Y, X = np.mgrid[0:ny + 2 * pad, 0:nx + 2 * pad].astype(np.float64)
    cx, cy = (nx + 2 * pad) / 2, (ny + 2 * pad) / 2
    x = X + 0.5 - cx
    y = Y + 0.5 - cy
    hw, hh = w / 2 + grow, h / 2 + grow
    if shape == "circle":
        return (x / hw) ** 2 + (y / hh) ** 2 <= 1.0
    if shape == "hexagon":
        # Flat-topped hexagon: vertices at (+-hw0, 0) and (+-hw0/2, +-hh0) before growth.
        hw0, hh0 = w / 2, h / 2
        # offset of the slanted edges: distance grows by grow / cos(theta)
        n = math.hypot(hh0, hw0 / 2)  # slanted edge normal scale
        ax, ay = np.abs(x), np.abs(y)
        slanted = hh0 * ax + (hw0 / 2) * ay <= hh0 * hw0 + grow * n
        return (ay <= hh0 + grow) & slanted
    inside = (np.abs(x) <= hw) & (np.abs(y) <= hh)
    if shape == "rounded" and corner_r > 0:
        r = min(corner_r + grow, hw, hh)
        dx = np.maximum(np.abs(x) - (hw - r), 0)
        dy = np.maximum(np.abs(y) - (hh - r), 0)
        inside &= dx * dx + dy * dy <= r * r
    return inside


def lonlat_outline(aoi: AOI, shape: str, corner_r_frac: float = 0.0, n: int = 128) -> list[list[float]]:
    """Outline polygon of the shape in lon/lat (for echoing back to the map)."""
    hw, hh = aoi.width_m / 2, aoi.height_m / 2
    if shape == "circle":
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        pts = np.column_stack([hw * np.cos(t), hh * np.sin(t)])
    elif shape == "hexagon":
        pts = np.array([[hw, 0], [hw / 2, hh], [-hw / 2, hh], [-hw, 0], [-hw / 2, -hh], [hw / 2, -hh]])
    else:
        pts = aoi.outline_local(0, 4)
    lon, lat = aoi.to_lonlat.transform(pts[:, 0], pts[:, 1])
    return [[float(a), float(b)] for a, b in zip(lon, lat)]
