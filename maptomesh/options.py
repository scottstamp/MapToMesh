"""User-facing generation options (validated by pydantic)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

HEX = r"^#[0-9a-fA-F]{6}$"


class Colors(BaseModel):
    single: str = Field("#E9E4D8", pattern=HEX)
    base: str = Field("#2B2B2B", pattern=HEX)
    frame: str = Field("#111111", pattern=HEX)
    ground: str = Field("#E9E4D8", pattern=HEX)
    water: str = Field("#2F8FD8", pattern=HEX)
    buildings: str = Field("#C8553D", pattern=HEX)
    roads: str = Field("#FAFAFA", pattern=HEX)
    vegetation: str = Field("#4F8A3C", pattern=HEX)
    bands: list[str] = Field(default_factory=lambda: ["#2F8FD8", "#6FA35A", "#C9B26B", "#8B5A2B", "#F4F4F4"])
    band_thresholds_m: list[float] | None = None
    band_mode: Literal["equal", "quantile"] = "equal"

    @field_validator("bands")
    @classmethod
    def _bands(cls, v: list[str]) -> list[str]:
        import re
        if not 1 <= len(v) <= 12:
            raise ValueError("1-12 band colours")
        for c in v:
            if not re.match(HEX, c):
                raise ValueError(f"bad colour {c}")
        return v


class Options(BaseModel):
    # area (centre + ground size in metres, local transverse Mercator)
    lat: float = Field(..., ge=41.0, le=84.0)
    lon: float = Field(..., ge=-142.0, le=-50.0)
    width_m: float = Field(..., gt=20, le=200_000)
    height_m: float = Field(..., gt=20, le=200_000)
    shape: Literal["rectangle", "rounded", "circle", "hexagon"] = "rectangle"
    corner_radius_mm: float = Field(12.0, ge=0, le=500)

    # data
    source: Literal["lidar", "hrdem"] = "lidar"
    surface: Literal["terrain", "buildings", "full"] = "buildings"
    grid_cells: int = Field(600, ge=50, le=2400)
    smoothing: float = Field(0.0, ge=0, le=10)

    # print geometry
    size_mm: float = Field(200.0, gt=10, le=2000)
    base_mm: float = Field(3.0, ge=0.4, le=50)
    floor_mm: float = Field(1.5, ge=0.2, le=50)
    z_exaggeration: float = Field(1.0, gt=0, le=50)
    max_height_mm: float | None = Field(None, gt=0, le=500)
    frame: bool = False
    frame_width_mm: float = Field(6.0, gt=0, le=100)
    frame_height_mm: float = Field(2.0, ge=0, le=200)
    tiles_x: int = Field(1, ge=1, le=8)
    tiles_y: int = Field(1, ge=1, le=8)
    tile_gap_mm: float = Field(10.0, ge=0, le=200)

    # water
    flatten_water: bool = True
    water_recess_mm: float = Field(0.6, ge=0, le=20)
    voids_as_water: bool = True
    osm_water: bool = True

    # roads
    osm_roads: Literal["none", "major", "medium", "all"] = "none"
    road_min_width_mm: float = Field(0.6, ge=0.1, le=10)
    road_width_scale: float = Field(1.0, gt=0, le=10)
    road_emboss_mm: float = Field(0.0, ge=-5, le=5)

    # colour
    color_mode: Literal["single", "landcover", "elevation"] = "landcover"
    colors: Colors = Field(default_factory=Colors)
    min_region_mm2: float = Field(0.4, ge=0, le=100)

    # output
    layout: Literal["parts", "objects"] = "parts"
    name: str = Field("MapToMesh", max_length=80)

    def elevation_key(self) -> str:
        return f"v3|{self.source}|{self.lat:.6f}|{self.lon:.6f}|{self.width_m:.1f}|{self.height_m:.1f}|{self.grid_cells}"
