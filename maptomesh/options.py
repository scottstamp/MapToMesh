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
    label: str = Field("#FAFAFA", pattern=HEX)
    bands: list[str] = Field(default_factory=lambda: ["#6FA35A", "#C9B26B", "#B07A3E", "#7A4E2A", "#F4F4F4"])
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
    cliff_smoothing: float = Field(1.0, ge=0, le=4)  # extra smoothing on steep slopes (cells)

    # print geometry
    size_mm: float = Field(200.0, gt=10, le=2000)
    base_mm: float = Field(3.0, ge=0.4, le=50)
    base_separate: bool = False  # base plate as its own print (frees a filament slot)
    base_lip: bool = True        # separate base: raised lip that locates the model (tray)
    lip_height_mm: float = Field(1.2, ge=0.2, le=10)
    lip_width_mm: float = Field(1.6, ge=0.4, le=10)
    lip_clearance_mm: float = Field(0.2, ge=0, le=2)
    # printer: colour boundaries / flat tops are aligned to this layer grid
    layer_height_mm: float = Field(0.2, ge=0.04, le=1.0)
    first_layer_mm: float = Field(0.2, ge=0.04, le=1.0)
    min_feature_mm: float = Field(0.5, ge=0, le=5)   # smallest printable building / patch width
    simplify_mm: float = Field(0.05, ge=0, le=0.5)   # max surface error when merging flat-ish cells
    # labels on the frame
    label: bool = False
    label_text: str = Field("", max_length=60)
    label_scale_bar: bool = True
    label_north: bool = True
    label_emboss_mm: float = Field(0.6, ge=0.2, le=5)
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
    osm_streams: Literal["none", "rivers", "all"] = "all"  # OSM river/stream centrelines as water
    river_min_width_mm: float = Field(0.8, ge=0.2, le=10)

    # roads
    osm_roads: Literal["none", "major", "medium", "all"] = "none"
    road_min_width_mm: float = Field(0.6, ge=0.1, le=10)
    road_width_scale: float = Field(1.0, gt=0, le=10)
    road_emboss_mm: float = Field(0.0, ge=-5, le=5)

    # colour
    color_mode: Literal["single", "landcover", "elevation"] = "landcover"
    colors: Colors = Field(default_factory=Colors)
    min_region_mm2: float = Field(1.0, ge=0, le=100)
    # colours are thin skins of this depth on a ground-coloured body (0 = full-height colour columns)
    color_depth_mm: float = Field(1.0, ge=0, le=50)
    # recolour small patches that alone force their colour into extra layers (total mm2, 0 = off)
    swap_patch_budget_mm2: float = Field(50.0, ge=0, le=5000)
    elevation_water: bool = True  # elevation-band mode: flat water in its own colour

    # output
    layout: Literal["parts", "objects"] = "parts"
    name: str = Field("MapToMesh", max_length=80)

    def elevation_key(self) -> str:
        return f"v4|{self.source}|{self.lat:.6f}|{self.lon:.6f}|{self.width_m:.1f}|{self.height_m:.1f}|{self.grid_cells}"
