# MapToMesh

Generate 3D-printable, multi-colour **3MF** topographic / city models of anywhere in Canada from
**NRCan CanElevation LiDAR point clouds** (COPC), with an OpenStreetMap-based area picker.

- Pick an area on an OpenStreetMap map (rectangle, rounded rectangle, circle, hexagon); the
  LiDAR coverage is shown as an overlay.
- Elevation comes straight from the classified point clouds (ground, buildings, bridges, trees,
  water) – or from the HRDEM 1 m mosaics for large areas.
- Colour by **land cover** (ground / water / buildings / roads & bridges / trees), by
  **elevation bands**, or single colour. Optional frame, base thickness, vertical exaggeration,
  max height, physical size / scale, and splitting into tiles for small print beds.
- Output: one 3MF with one watertight part per colour (filament slots pre-assigned for
  Bambu Studio / OrcaSlicer), an STL-per-colour zip, a top-view thumbnail and an in-browser 3D
  preview. Elevation-band models also list the Z heights for manual filament changes on
  single-extruder printers.

## Quick start

Requires Python 3.11+ (tested on 3.14, Windows).

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt     # Linux/macOS: .venv/bin/pip
.venv/Scripts/python run.py                       # opens http://127.0.0.1:8000/
```

On first start the app downloads NRCan's LiDAR tile index (~230 MB, once) and converts it into
`cache/tile_index.npz` (~22 MB). The status line at the top of the sidebar shows progress.

## Using it

1. **Area** – search a place, then *Draw on map* (drag a box; for circle/hexagon drag from the
   centre) or type the width/height. Drag the red outline to move it. The info box lists how many
   LiDAR tiles / which acquisition projects cover the area.
2. **Elevation data** – *LiDAR point clouds* (default) or *HRDEM 1 m mosaic*. Surface:
   bare terrain, terrain + buildings & bridges, or full surface including trees. *Detail* is the
   number of grid cells along the long side; the hint shows metres and millimetres per cell.
3. **Print** – long-side size in mm (scale is shown), base thickness, minimum terrain thickness,
   vertical exaggeration, optional frame and tiling.
4. **Water & roads** – lakes/sea are flattened (and optionally recessed); roads come from
   OpenStreetMap with a minimum printable width.
5. **Colours** – presets or individual colour pickers. Islands smaller than the given area are
   merged into their surroundings to avoid pointless filament swaps.
6. **Generate** – progress and log stream in the sidebar. A finished job can be reopened with
   `?job=<id>` while the server is running.

### In the slicer

- **Bambu Studio / OrcaSlicer**: open the 3MF; it loads as one object whose parts are already
  mapped to filament slots 1..N (listed in the result panel). Set those slots to the matching
  colours.
- **PrusaSlicer** and others: the parts are named after their colour (e.g. `Water`), assign
  extruders per part. If your slicer asks whether to load as a single object with multiple
  parts, answer *yes*. With the *Separate objects* layout every colour is its own object.
- **Single extruder, elevation bands**: print as one colour and insert filament changes at the
  listed Z heights (round to your layer height).

## How it works

```
web/ (Leaflet + three.js)  ──HTTP──>  maptomesh/server.py (FastAPI, background jobs)
                                       ├─ lidar_index.py  NRCan tile/project indexes -> npz / GeoJSON
                                       ├─ pointcloud.py   COPC range-reads at the needed LOD, binning
                                       ├─ hrdem.py        STAC search + COG overview reads (rasterio)
                                       ├─ osm.py          OpenMapTiles vector tiles (OpenFreeMap) -> masks
                                       ├─ model.py        surfaces, water levels, colour regions, parts
                                       ├─ mesh.py         watertight heightfield solids
                                       └─ export.py       3MF / STL / preview / thumbnail
```

- **Point clouds**: each COPC tile is read remotely with HTTP range requests, only for octree
  nodes inside the area and only down to the level of detail the grid needs, decompressing just
  XYZ + classification + flags. Points are re-projected from each tile's CRS (UTM zone + CGVD2013
  etc.) into a local transverse Mercator grid centred on the area. Projects are merged oldest to
  newest.
- **Inconsistent classification across projects** is handled explicitly: per project the app
  detects whether buildings (6) and vegetation (3–5) are classified (remembered in
  `cache/project_classes.json`) and routes class 1 accordingly – e.g. Ottawa 2020 leaves
  buildings in class 1, Hamilton 2021 and St. John's 2020 leave trees in class 1. Where buildings
  are not classified, OpenStreetMap footprints (or a roof-flatness heuristic) separate them from
  trees.
- **Water**: LiDAR water class, large voids in the point cloud and OSM water polygons. Near-level
  water is flattened per water body; sloped rivers keep their profile; open sea outside the LiDAR
  tiles is filled with the sea level.
- **Meshes**: every colour region becomes its own closed, consistently oriented 2-manifold
  (checked in tests). Flat areas (bases, frames, water, clamped band tops) are strip-triangulated
  without T-junctions, which keeps files small.

## Tests

```bash
.venv/Scripts/python tests/test_mesh_unit.py                    # offline mesh/raster checks
.venv/Scripts/python run.py --no-browser --port 8765            # in another terminal
.venv/Scripts/python tests/integration.py                       # end-to-end, needs internet
.venv/Scripts/python tests/validate_3mf.py some_model.3mf       # independent 3MF checker
```

The integration suite focuses on **St_Johns_2020_2** (downtown + harbour, a 10 km / 121-tile
elevation-band model split into 4 tiles, Cape Spear running off the eastern edge of the data,
Signal Hill full surface at high detail, and the HRDEM comparison), plus Ottawa and Hamilton
regressions. Each output is re-parsed and every part is checked for open edges, duplicate edges,
degenerate triangles and positive volume. Thumbnails land in `cache/integration/`.

## Configuration

| Environment variable | Default |
|---|---|
| `MAPTOMESH_CACHE` | `./cache` (tile index, elevation/OSM caches, outputs) |
| `MAPTOMESH_VECTOR_TILES` | `https://tiles.openfreemap.org/planet` (any OpenMapTiles TileJSON) |

## Limits and known behaviour

- Up to 2,500 LiDAR tiles per request (≈ 2,500 km²); use HRDEM for anything larger.
- Areas outside LiDAR coverage are interpolated and reported in the result notes.
- HRDEM has no classification: buildings come from OSM footprints, other raised objects count as trees.
- Multi-colour parts are full-height columns above the base, so colours show on the side walls.

## Data and licences

- LiDAR point clouds and HRDEM: Natural Resources Canada, CanElevation series –
  [Open Government Licence – Canada](https://open.canada.ca/en/open-government-licence-canada).
- Map tiles, roads, water and building footprints: © OpenStreetMap contributors (ODbL), served by
  [OpenFreeMap](https://openfreemap.org/); place search by Nominatim.
- Please respect the tile and Nominatim usage policies; everything is cached locally.
