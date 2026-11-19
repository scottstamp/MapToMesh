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
   centre) or type the width/height. Drag the red outline to move it, or the white handles on its
   corners/edges to resize it (the opposite side stays put). The info box lists how many
   LiDAR tiles / which acquisition projects cover the area.
2. **Elevation data** – *LiDAR point clouds* (default) or *HRDEM 1 m mosaic*. Surface:
   bare terrain, terrain + buildings & bridges, or full surface including trees. *Detail* is the
   number of grid cells along the long side; the hint shows metres and millimetres per cell.
   *Cliff smoothing* only affects steep slopes; raise it if sea cliffs or bluffs still look rough.
3. **Print** – long-side size in mm (scale is shown), base thickness, minimum terrain thickness,
   vertical exaggeration, optional frame (with an embossed label: name, scale bar, north arrow)
   and tiling. *Printer* settings: layer / first-layer height (colour boundaries, flat water and
   band heights are aligned to that layer grid, so no half-layer colour slivers), minimum feature
   width (smaller buildings are enlarged, narrower gaps closed, thinner tree slivers dropped) and a
   simplification tolerance for near-flat surfaces. *Print base plate separately*: the model starts on top of the
   base (z = 0 on the bed) without the base colour, freeing a filament slot; the base plate is a
   separate single-colour 3MF download (also in the STL zip) with a raised locating lip around
   the model (a tray: drop the model in, glue). With tiling, base tiles are split half a tile off
   the model tiles so every model seam is bridged by a base tile.
4. **Water & roads** – every lake / sea / harbour is printed as a perfectly flat, slightly
   recessed surface with a crisp shoreline; roads come from OpenStreetMap with a minimum
   printable width. Rivers and streams use the OpenStreetMap centrelines (LiDAR only sees
   scattered water returns on narrow streams), widened to at least *Min river width* so they
   stay continuous; their surface follows the channel downhill and roads stay on top where they
   cross. They are a separate "Rivers" part in the water colour. *Max shore slope* (default
   70 degrees, 90 = vertical walls): cliffs and banks descend to the water at most this steep and
   end in one even edge (the water recess) along the shoreline.
5. **Colours** – presets or individual colour pickers. Islands smaller than the given area are
   merged into their surroundings to avoid pointless filament swaps. *Colour depth*: water, roads,
   trees and buildings are thin colour skins on one ground-coloured body, so a layer only carries
   the colours whose visible surface is at that height (plus building walls) - nothing is coloured
   through the model's full height. *Drop swap-only patches*: small patches that alone force their
   colour into extra layers (a lone pond or hilltop house) are recoloured to ground, best savings
   first, within an area budget. The result panel shows the estimated number of colour changes. In elevation-band mode water can get its own colour
   (needs a multi-material printer) or stay part of the lowest band (single-extruder friendly).
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

### Filament changes and purge

The result shows an estimated number of filament changes. Slots are ordered light to dark (slicers
that change in slot order then flush light into dark, which needs the least purge) and the result
lists slicer tips (layer heights to use, flushing into infill/objects).

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
- **Ground on steep terrain**: each grid vertex's ground height is a weighted least-squares
  plane fitted through the ground points of its cell and neighbours, evaluated at the vertex. A
  plain average depends on where in the cell the points happen to fall, which on a 60 m sea cliff
  produces saw-tooth faces. Slopes steeper than ~20 degrees are additionally despiked and smoothed
  (*Cliff smoothing*, 0 = off); gentler terrain keeps full detail.
- **Shorelines**: lake and sea outlines are smoothed at about the minimum feature size (LiDAR
  water edges wobble from cell to cell; rivers keep a tight outline). A grid sample next to a
  cliff that drops straight into the sea can be anywhere on the face, so the land used to end in
  a wall whose height jumped from vertex to vertex (jagged cliff tops). The land near water is
  now capped by a ramp that rises from the exact water outline at *Max shore slope*: cliffs
  descend smoothly with contours parallel to the shore, the cell diagonals follow the ramp, and
  the shore edge is the same height everywhere.
- **Inconsistent classification across projects** is handled explicitly: per project the app
  detects whether buildings (6) and vegetation (3–5) are classified (remembered in
  `cache/project_classes.json`) and routes class 1 accordingly – e.g. Ottawa 2020 leaves
  buildings in class 1, Hamilton 2021 and St. John's 2020 leave trees in class 1. Where buildings
  are not classified, OpenStreetMap footprints (or a roof-flatness heuristic) separate them from
  trees.
- **Water**: LiDAR water class, large voids in the point cloud and OSM water polygons. Each water
  body gets one constant level (median of the LiDAR water returns), so noisy scans over the sea
  print perfectly flat; only rivers whose measured surface drops more than 3 m keep a (smoothed)
  profile. Open sea outside the LiDAR tiles takes the sea level.
- **Smooth, exact region outlines**: the terrain grid is triangulated and every triangle is cut
  exactly along iso-lines – the shape outline (true circles/hexagons), smoothed class boundaries
  and elevation contours – instead of assigning whole grid cells, so colour edges and band
  contours are smooth lines rather than pixel staircases (`maptomesh/partition.py`).
- **Per-colour surfaces**: each colour part has its own top surface (flat water, roofs, terrain,
  bridge decks), giving crisp vertical shorelines and building walls in the ground colour; water
  colour only ever appears on the flat water itself.
- **Meshes**: every part is a closed, consistently oriented 2-manifold (checked in tests). Flat
  areas (bases, frames, water, band tops) are strip-triangulated, and near-flat terrain is merged
  into larger triangles (quadtree, within the simplification tolerance) - both without
  T-junctions. Most remaining triangles sit along colour boundaries.
- **Caching NRCan data**: CanElevation is a free public service, so every byte read from it -
  COPC point-cloud nodes and HRDEM COG tiles - is kept in an on-disk block cache
  (`cache/nrcan_range_cache.sqlite`, 64 KiB blocks, LRU, default 10 GB via
  `MAPTOMESH_DATA_CACHE_GB`, ETags re-checked after 30 days). Overlapping areas, other detail
  levels and re-generations only download blocks never seen before; the job log reports MB
  downloaded vs. served from cache. STAC searches are cached for a day, OSM tiles for 30 days.

## Tests

```bash
.venv/Scripts/python tests/test_mesh_unit.py                    # offline mesh/raster checks
.venv/Scripts/python tests/test_partition.py                    # offline iso-line partition checks
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
| `MAPTOMESH_DATA_CACHE_GB` | `10` (size limit of the NRCan range cache) |

## Limits and known behaviour

- Up to 2,500 LiDAR tiles per request (≈ 2,500 km²); use HRDEM for anything larger.
- Areas outside LiDAR coverage are interpolated and reported in the result notes.
- HRDEM has no classification: buildings come from OSM footprints, other raised objects count as trees.
- On hilly terrain with trees/roads/buildings at every elevation, most layers still need every
  colour: a colour must be printed wherever its surface is visible. Fewer colours (e.g. trees in
  the ground colour) is the main lever there; the estimate in the result panel shows the effect.

## Data and licences

- LiDAR point clouds and HRDEM: Natural Resources Canada, CanElevation series –
  [Open Government Licence – Canada](https://open.canada.ca/en/open-government-licence-canada).
- Map tiles, roads, water and building footprints: © OpenStreetMap contributors (ODbL), served by
  [OpenFreeMap](https://openfreemap.org/); place search by Nominatim.
- Please respect the tile and Nominatim usage policies; everything is cached locally.
