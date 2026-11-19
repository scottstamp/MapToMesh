# MapToMesh

Generate 3D-printable, multi-colour **3MF** topographic / city models of **anywhere on Earth**,
with an OpenStreetMap-based area picker. Each area uses the best elevation data available:
classified **LiDAR point clouds** where they exist (Canada, the US, France, Switzerland, Poland,
Slovenia, ...), 0.5–10 m national **elevation models** across much of Europe, Japan, New Zealand
and parts of Australia, and the **Copernicus 30 m** model everywhere else (see
[Elevation sources](#elevation-sources)).

- Pick an area on an OpenStreetMap map (rectangle, rounded rectangle, circle, hexagon); the
  LiDAR coverage is shown as an overlay.
- Elevation comes straight from classified point clouds (ground, buildings, bridges, trees,
  water) where available; gaps and other areas are filled from the highest-resolution elevation
  model that covers them.
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
2. **Elevation data** – *Best available* (default: point clouds, then elevation models, highest
   resolution first) or *Elevation models only* (faster for big areas). The info box lists the
   point-cloud projects and the elevation models that will fill the rest. *US LiDAR source*: USGS
   3DEP on AWS (default; every project, kept current) or Microsoft Planetary Computer (a COPC
   snapshot to about 2022 with gaps). Surface:
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

### Reviews and problem reports

After a model is generated (or a job fails) a short form asks for a 1-5 star score, problem
categories (terrain, missing data, buildings/trees, water, roads, colours, slicing/printing, slow or
failed generation, other) and an optional comment. Ticking *I'd like a follow-up* reveals optional
name and email fields; contact details are only stored then.

Reviews go to SQLite at `data/reviews.sqlite` (`MAPTOMESH_DATA_DIR`), one row per review with the
job's full settings and a snapshot of its result (stats, data sources, notes, errors, the end of the
job log), so reported problems can be reproduced. No IP addresses are stored; a hidden honeypot field
and a per-address limit (20 per hour, 10 per job) keep spam out. The `data/` folder is git-ignored:
it can contain personal data. Rows have `status` / `admin_note` columns for the planned admin report
page (`maptomesh.reviews.ReviewStore.recent()` lists them; escape all text when rendering it).

### Filament changes and purge

The result shows an estimated number of filament changes. Slots are ordered light to dark (slicers
that change in slot order then flush light into dark, which needs the least purge) and the result
lists slicer tips (layer heights to use, flushing into infill/objects).

## Elevation sources

For every area MapToMesh combines sources, best first:

1. **Classified LiDAR point clouds** (COPC / Entwine, read remotely: only the part of the octree
   inside the area, at the detail the grid needs).
2. **Elevation models** for everything the point clouds don't cover, highest resolution first.
   Each one only fills what is still missing, and is shifted by its median height difference to
   what is already placed (national height datums differ from each other by up to a metre or two),
   so seams don't step. A provider whose download would be too large for the area is skipped in
   favour of the next one.
3. **Copernicus DEM 30 m** everywhere else (also sea and lakes outside the other sources).

Elevation-model areas go through the same surface builder as point clouds: the terrain model is
the ground and a surface model, where there is one, supplies "unclassified" heights that become
buildings (OSM footprints) and trees. Terrain-only sources give bare terrain plus OSM features.
The result panel lists the share of each source and the **credits** their licences require
(also written into the 3MF description). The sources below allow commercial use of derived
products (prints) with attribution; the few restricted collections that are used as a last resort
come with a licence warning on the model (see *Limits*).

| Region | Source | Data | Resolution |
|---|---|---|---|
| Global | Copernicus DEM 30 m (global) | surface model | 30 m |
| Canada, US | NRCan CanElevation (Canada) | classified point cloud (COPC) | point cloud |
| Canada, US | USGS 3DEP (US): AWS Entwine or Planetary Computer | classified point cloud | point cloud |
| Canada, US | NRCan HRDEM 1-2 m (Canada) | terrain + surface | 1 m |
| Canada, US | USGS 3DEP 10 m (US) | terrain | 10 m |
| France, Benelux, Switzerland | IGN LiDAR HD (France) | classified point cloud (COPC) | point cloud |
| France, Benelux, Switzerland | swisstopo swissSURFACE3D (Switzerland) | classified point cloud (COPC) | point cloud |
| France, Benelux, Switzerland | IGN LiDAR HD 0.5 m (France) | terrain + surface | 0.5 m |
| France, Benelux, Switzerland | swisstopo swissALTI3D 0.5 m (Switzerland) | terrain + surface | 0.5 m |
| France, Benelux, Switzerland | AHN4 0.5 m (Netherlands) | terrain + surface | 0.5 m |
| France, Benelux, Switzerland | Luxembourg LiDAR 2024 0.5 m | terrain + surface | 0.5 m |
| France, Benelux, Switzerland | Digitaal Hoogtemodel Vlaanderen II 1 m (Flanders) | terrain + surface | 1 m |
| France, Benelux, Switzerland | IGN RGE ALTI 1 m (France) | terrain | 1 m |
| UK, Ireland, Spain, Italy | Scottish Remote Sensing Portal LiDAR 0.5-1 m (Scotland) | terrain + surface | 0.5 m |
| UK, Ireland, Spain, Italy | South Tyrol LiDAR DTM/DSM 0.5 m (Italy, valleys) | terrain + surface | 0.5 m |
| UK, Ireland, Spain, Italy | Welsh Government LiDAR 1 m 2020-23 (Wales) | terrain + surface | 1 m |
| UK, Ireland, Spain, Italy | Environment Agency LiDAR 1 m (England) | terrain + surface | 1 m |
| UK, Ireland, Spain, Italy | GSI / OPW open LiDAR 1-2 m (Ireland, partial) | terrain + surface | 1 m |
| UK, Ireland, Spain, Italy | South Tyrol DTM/DSM 2.5 m (Italy) | terrain + surface | 2.5 m |
| UK, Ireland, Spain, Italy | IGN MDT05 5 m (Spain) | terrain | 5 m |
| Central Europe | Slovenia LiDAR (GURS CLSS / ARSO LSS, COPC mirror) | classified point cloud (COPC) | point cloud |
| Central Europe | Poland LiDAR (GUGiK, COPC mirror) | classified point cloud (COPC) | point cloud |
| Central Europe | Wallonia LiDAR 2021-2022 (SPW, COPC mirror) | classified point cloud (COPC) | point cloud |
| Central Europe | Saxony LiDAR (GeoSN, COPC mirror) | classified point cloud (COPC) | point cloud |
| Central Europe | BEV ALS 1 m (Austria) | terrain + surface | 1 m |
| Central Europe | LGLN DGM1 1 m (Lower Saxony) | terrain + surface | 1 m |
| Central Europe | Geobasis NRW DGM1/DOM1 1 m (North Rhine-Westphalia) | terrain + surface | 1 m |
| Central Europe | LVermGeo DGM1/DOM1 1 m (Rhineland-Palatinate) | terrain + surface | 1 m |
| Central Europe | LAiV DGM1/DOM1 1 m (Mecklenburg-Western Pomerania) | terrain + surface | 1 m |
| Central Europe | Bayerische Vermessungsverwaltung DGM1 1 m (Bavaria) | terrain | 1 m |
| Central Europe | LGB DGM1 1 m (Brandenburg, Berlin) | terrain | 1 m |
| Central Europe | GeoSN DGM1 1 m (Saxony) | terrain | 1 m |
| Nordics, Baltics, Arctic | Maa- ja Ruumiamet 1 m (Estonia) | terrain | 1 m |
| Nordics, Baltics, Arctic | NLS 2 m (Finland) | terrain | 2 m |
| Nordics, Baltics, Arctic | ArcticDEM 2 m (Arctic, surface model) | surface model | 2 m |
| Nordics, Baltics, Arctic | ÍslandsDEM 10 m (Iceland) | surface model | 10 m |
| Australia, New Zealand | LINZ LiDAR 1 m (New Zealand) | terrain + surface | 1 m |
| Australia, New Zealand | Queensland LiDAR 1 m | terrain | 1 m |
| Australia, New Zealand | Geoscience Australia LiDAR 5 m | terrain | 5 m |
| Australia, New Zealand | LINZ contour DEM 8 m (New Zealand) | terrain | 8 m |
| Japan, Hong Kong | CEDD LiDAR 2020 0.5 m (Hong Kong) | terrain + surface | 0.5 m |
| Japan, Hong Kong | GSI 1-10 m (Japan) | terrain | 1 m |

`tests/probe_sources.py` smoke-tests every provider on a sample area
(`python tests/probe_sources.py --list` lists them). Research notes, endpoints and the sources
that were left out (portals, logins, non-commercial licences, unreachable services) are in
[docs/plans/international-sources.md](docs/plans/international-sources.md).

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
- **USGS 3DEP (US), AWS**: USGS / Hobu publish every 3DEP project as an Entwine Point Tile
  octree in `s3://usgs-lidar-public` (Web Mercator, one small LAZ file per node). Projects
  intersecting the area come from Hobu's `resources.geojson` footprints (also the coverage
  overlay); each octree is walked down to the depth whose point spacing matches the grid and the
  intersecting nodes are fetched whole (in parallel, through the range cache). No token needed.
- **USGS 3DEP (US), Planetary Computer**: tiles are found with a STAC search of the
  `3dep-lidar-copc` collection (cached for a day) and read with the same remote COPC reader.
  Blob URLs need a SAS token, fetched anonymously and added per request by the cache, so cached
  blocks outlive the tokens. Heights are converted to metres from each file's CRS (both US
  sources agree to a few cm on a shared project). 3DEP projects
  classify ground / water / bridges but usually leave buildings and trees in class 1, which goes
  through the same OSM-footprint routing as unclassified Canadian projects; where a project does
  classify both, leftover class-1 points inside OSM footprints still count as buildings
  (classifiers often skip the tallest towers - Chicago 2017 leaves Willis Tower in class 1).
- **Caching LiDAR data**: CanElevation and 3DEP are free public services, so every byte read from
  them - COPC point-cloud nodes and HRDEM COG tiles - is kept in an on-disk block cache
  (`cache/nrcan_range_cache.sqlite`, 64 KiB blocks, LRU, default 10 GB via
  `MAPTOMESH_DATA_CACHE_GB`, ETags re-checked after 30 days). Overlapping areas, other detail
  levels and re-generations only download blocks never seen before; the job log reports MB
  downloaded vs. served from cache. STAC searches are cached for a day, OSM tiles for 30 days.

## Tests

```bash
.venv/Scripts/python tests/test_mesh_unit.py                    # offline mesh/raster checks
.venv/Scripts/python tests/test_partition.py                    # offline iso-line partition checks
.venv/Scripts/python tests/probe_sources.py                    # every elevation provider, needs internet
.venv/Scripts/python run.py --no-browser --port 8765            # in another terminal
.venv/Scripts/python tests/integration.py                       # end-to-end, needs internet
.venv/Scripts/python tests/validate_3mf.py some_model.3mf       # independent 3MF checker
```

The integration suite focuses on **St_Johns_2020_2** (downtown + harbour, a 10 km / 121-tile
elevation-band model split into 4 tiles, Cape Spear running off the eastern edge of the data,
Signal Hill full surface at high detail, and the HRDEM comparison), plus Ottawa and Hamilton
regressions, the Chicago Loop (Planetary Computer), Niagara Falls (NRCan + USGS in one model),
Seattle and the Grand Canyon (USGS on AWS), and one area per international source family (Paris,
Amsterdam, the Matterhorn, London, Snowdon, Hallstatt, Cologne, Krakow, Wellington, Brisbane, Fuji,
Hong Kong, Lofoten, Teide, Everest) - each checks that the expected source supplied the data. Each output is re-parsed and every part is checked for open edges, duplicate edges,
degenerate triangles and positive volume. Thumbnails land in `cache/integration/`.

## Configuration

| Environment variable | Default |
|---|---|
| `MAPTOMESH_CACHE` | `./cache` (tile index, elevation/OSM caches, outputs) |
| `MAPTOMESH_VECTOR_TILES` | `https://tiles.openfreemap.org/planet` (any OpenMapTiles TileJSON) |
| `MAPTOMESH_DATA_CACHE_GB` | `10` (size limit of the LiDAR / DEM range cache) |
| `MAPTOMESH_SERVICE_CACHE_GB` | `2` (size limit of cached elevation-service responses) |
| `MAPTOMESH_DATA_DIR` | `./data` (user reviews database; not a cache, keep it private) |
| `MAPTOMESH_PUBLIC_URL` | `https://maptomesh.com` (base of the absolute URLs in link previews; empty = the URL each request came in on) |

**Link previews**: the page carries Open Graph / Twitter-card metadata, so links posted to Discord,
Slack, X, iMessage etc. show a preview. The server fills in absolute URLs (from
`MAPTOMESH_PUBLIC_URL`, default `https://maptomesh.com`). The default image is `web/og-image.jpg` (1200 x 630); a shared
`?job=<id>` link previews that model's top view instead. To regenerate the example image, generate
San Francisco (centre 37.7870, -122.4400, 10 x 8.5 km, 1000 cells, land cover, bare terrain,
z x2) and screenshot the 3D view alone with headless Chrome at 1200 x 630:
`/?job=<id>&embed=1&az=180&el=40&zoom=1.25` (`embed=1` hides everything but the 3D view).

## Limits and known behaviour

- Up to 2,500 LiDAR tiles per request (≈ 2,500 km²); larger areas use elevation models. Each
  elevation model has its own area limit (e.g. Hong Kong 25 km², German 1 m state tiles
  20–120 km²); beyond it the next source is used.
- Not covered at high resolution (Copernicus 30 m there): southern Norway (Kartverket was not
  reachable from North America when this was built), Sweden and Denmark (need an account / API
  token), most of Asia, Africa and South America.
- **Licence warnings**: a few collections carry restrictions - Scotland's LiDAR phases II/III
  (the survey contractor retains the commercial copyright) and Historic Environment Scotland
  sets (Non-Commercial Government Licence). They are only used where no unrestricted data
  exists, and a model that uses them shows a licence warning (result panel, notes and the 3MF
  description).
- Spain's national 5 m model comes in whole metres (stepped at large scales).
- USGS 3DEP on Planetary Computer is a snapshot (projects up to about 2022) with gaps - e.g.
  Seattle, Miami, Manhattan, the Grand Canyon have no tiles there (the default AWS source has
  them). With Planetary Computer selected the map shows its tiles only when zoomed in. HRDEM is
  Canada only.
- USGS on AWS reads whole octree nodes: the coarse levels near the root cover a whole project,
  so a first request in a new project downloads more than COPC would (Seattle 2.4 x 1.8 km:
  ~28 MB); everything is cached afterwards.
- Waterfalls are not modelled specially: a river and the pool below the falls form one water
  body whose level is smoothed between the two.
- On hilly terrain with trees/roads/buildings at every elevation, most layers still need every
  colour: a colour must be printed wherever its surface is visible. Fewer colours (e.g. trees in
  the ground colour) is the main lever there; the estimate in the result panel shows the effect.

## Data and licences

- LiDAR point clouds and HRDEM: Natural Resources Canada, CanElevation series –
  [Open Government Licence – Canada](https://open.canada.ca/en/open-government-licence-canada).
- US LiDAR: USGS 3D Elevation Program (public domain), served from the
  [AWS Open Data registry](https://registry.opendata.aws/usgs-lidar/) (Entwine, footprints by
  Hobu) or [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/dataset/3dep-lidar-copc)
  (subject to its terms of use).
- Elsewhere: national and regional mapping agencies and Copernicus (see
  [Elevation sources](#elevation-sources)); each model lists the exact credit lines.
- Map tiles, roads, water and building footprints: © OpenStreetMap contributors (ODbL), served by
  [OpenFreeMap](https://openfreemap.org/); place search by Nominatim.
- Please respect the tile and Nominatim usage policies; everything is cached locally.
- Selling prints? Every source used allows it with credit: copy the **Credits** shown with the
  model (also in the 3MF description) plus "© OpenStreetMap contributors" into the listing or a
  label.

## License

[PolyForm Noncommercial 1.0.0](LICENSE.md) with an additional permission for generated models:

- **Free for personal and other noncommercial use**: run, modify and share the software.
- **Generated models and prints are yours to use and sell**, including commercially. The
  licensor claims no rights in them (third-party data credits above still apply).
- **Not allowed without permission**: selling or bundling the software itself, or offering it
  (or a modified version) as a commercial product or paid service.

This is a source-available licence, not an OSI open-source licence. See [LICENSE.md](LICENSE.md)
for the full terms.
