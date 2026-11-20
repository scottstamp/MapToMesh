# MapToMesh

Generate 3D-printable, multi-colour **3MF** topographic / city models of **anywhere on Earth**,
with an OpenStreetMap-based area picker. Each area uses the best elevation data available:
classified **LiDAR point clouds** where they exist (Canada, the US, France, Switzerland, Poland,
Slovenia, ...), 0.25–10 m national **elevation models** across most of Europe, North America,
Japan, New Zealand and parts of Australia (from the agencies and from
[Mapterhorn](https://mapterhorn.com)'s shared terrain tiles), and the **Copernicus 30 m** model
everywhere else (see [Elevation sources](#elevation-sources)).

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

### Optional native extension

`native/` is a small Rust extension (PyO3) for the CPU-heavy steps, all on every core:

- writing the 3MF (and the STL zip): a 6.9 M-triangle model takes 0.8 s instead of 12 s, with
  byte-identical content;
- meshing the parts (`Partition.build_part`): about 5x faster, same vertices and triangles in the
  same order as the NumPy version;
- simplifying the 3D preview of large models (meshoptimizer);
- fitting the ground planes of the point clouds (3.5 s -> 0.03 s on a 3 M-cell grid).

Without it the pure-Python versions are used automatically (the preview of a very large model is
then thinned instead of simplified). To build it (needs a Rust toolchain and a C++ compiler; on
Windows the MSVC build tools):

```bash
.venv/Scripts/pip install maturin
.venv/Scripts/maturin develop --release -m native/Cargo.toml   # builds and installs into the venv
```

For deployment, `maturin build --release -m native/Cargo.toml` makes a wheel (abi3: one wheel per
OS / CPU for every Python >= 3.10). `MAPTOMESH_NATIVE=0` switches back to the Python code.

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
   bare terrain, terrain + bridges (road and rail decks, no buildings: at coarse detail LiDAR
   buildings come out lumpy), terrain + buildings & bridges, or full surface including trees.
   *Detail* is the
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

The panel opens in **Simple** mode: area, surface, detail, size, exaggeration, frame and label,
roads and rivers, colours and name. **Advanced** shows everything else (data source, smoothing,
base and printer settings, tiles, water tuning, colour depth, 3MF layout). Hidden advanced
settings keep applying; Simple mode lists any that differ from the defaults, with a reset.

On a phone held upright the settings panel becomes a bottom sheet under a full-width map: tap or
swipe its header to open it; the header keeps the *Generate* button, progress and the download
link in view. The area can be moved, resized and drawn with a finger. The ☾ / ☀ button next to
*About / Donate* switches between the light and dark theme (the default follows the system).

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
2. **Elevation models** for everything the point clouds don't cover, most detailed first. Detail
   finer than half a grid cell can't reach the model, so it counts as equal (a 0.5 m and a 2 m
   model tie on a 5 m grid); ties go to a model with a surface model when the job needs one
   (buildings, trees), then to Mapterhorn's shared tiles (below), then to the finer native
   resolution. Each one only fills what is still missing, and is shifted by its median height
   difference to what is already placed (national height datums differ from each other by up to
   a metre or two), so seams don't step. A provider whose download would be too large for the
   area is skipped in favour of the next one.
3. **30 m global models** everywhere else: FABDEM (Copernicus with trees and buildings removed;
   non-commercial, so models using it carry a licence warning) on land, the **Copernicus DEM**
   for sea and lakes and wherever FABDEM can't be read.

**Bare-terrain jobs** (no land-cover colours) skip the point clouds where Mapterhorn's models are
at least as fine as the grid cells over all the land of the area: a few MB of tiles instead of
often hundreds of MB of point clouds from the agencies' servers, for a ground that differs by a
few centimetres (median) - hundredths of a millimetre on the print.

### Mapterhorn

[Mapterhorn](https://mapterhorn.com) merges ~150 open national and regional terrain models
(0.25–20 m, bare earth) into one global set of tiles (PMTiles archives on Cloudflare, read with
range requests and cached like everything else). It fills areas no agency service here covers -
Denmark 0.4 m, Norway and Sweden 1 m, Portugal and Romania 0.5 m, Spain 0.5–2 m, Czechia, Slovakia,
Italian regions, Tasmania, the German states without a service, Cyprus, Latvia, Taiwan, Israel,
Greenland... - and, at equal detail, it is used before the agencies' own services to spare them.
The **Elevation models** setting (advanced) switches to "agencies' own services first" (they can
have newer releases); Mapterhorn then only fills gaps.

How it is used:

- Its coverage index gives each source's footprint; the area gets one candidate per resolution
  tier, limited to the footprints, so its global Copernicus filler never stands in for real data.
- Where two sources meet (and along coasts) Mapterhorn blends them over ~150 m; measured against
  the agencies' models the blend affects the first ~50 m only. That band ranks like 25 m data:
  an agency model fills it where there is one, otherwise it still beats Copernicus.
- Heights are Terrarium-encoded with a zoom-dependent rounding of 3 cm at 1 m pixels (z16), 6 cm
  at 2 m (z15): far below what a print shows.
- No surface models: buildings come from OSM heights (below) and trees are left out (or another
  source with a surface model ranks first for those jobs).

Compared with the agencies' own models on the same grid (2026-11-19; residual after the median
datum offset):

| Area (grid) | Compared with | RMSE | p99 | Mapterhorn / agency time (cold) |
|---|---|---|---|---|
| St. John's (5 m) | NRCan HRDEM | 0.12 m | 0.47 m | 1.9 s / 1.0 s |
| Zermatt (5 m) | swissALTI3D | 0.16 m | 0.61 m | 1.8 s / 8.7 s |
| Lake District (5 m) | Environment Agency 1 m | 0.12 m | 0.43 m | 3.4 s / 8.2 s |
| Chamonix (2 m) | IGN LiDAR HD 0.5 m | 0.18 m | 0.34 m | 2.6 s / 74 s |
| Kamakura (5 m) | GSI tiles | 0.33 m | 0.05 m | 0.4 s / 0.6 s |
| Hallstatt (5 m) | BEV ALS 2025 | 0.30 m | 1.18 m | 3.8 s / 4.0 s |
| St. John's harbour (1.6 m) | LiDAR ground (CanElevation) | 0.16 m | 0.62 m | 197 MB of point clouds vs 5 MB of tiles |
| Boulder (6.7 m) | LiDAR ground (USGS) | 0.39 m | 1.47 m | 17 s / 51 MB vs 1.6 s / 3.5 MB |

Larger differences were all the agencies' other datasets: RGE ALTI's older mountain data
(Chamonix, 5.6 m RMSE; Mapterhorn has LiDAR HD there), Spain's 5 m model against Mapterhorn's 2 m,
and BEV's newer 2025 release (+0.23 m offset, flat ground equal within 2 cm).

Elevation-model areas go through the same surface builder as point clouds: the terrain model is
the ground and a surface model, where there is one, supplies "unclassified" heights that become
buildings (OSM footprints) and trees. Terrain-only sources give bare terrain plus OSM features;
there, OSM building footprints are raised to their mapped heights as flat-topped blocks
(OpenMapTiles `render_height`: the tagged height, else floors x ~3.7 m, else a default; parts that
start above the ground count only over a building, so canopies and skybridges don't wall off
streets). On surface-only fallbacks (Copernicus 30 m), whose heights already hold blurred
buildings, the blocks stand on the ground under those bumps (a ~150 m grey opening).

**Building sources** for those bare-earth areas (`maptomesh/buildings/`; nothing is fetched where
LiDAR or a surface model covers the ground):

- **Japan: Project PLATEAU** 3D city models (MLIT; ~300 municipalities, ~250 with LoD2 roof
  shapes, LoD1 blocks at the surveyed height elsewhere). PLATEAU VIEW's nationwide tilesets list
  each city's 3D Tiles; the leaves under the area are read (b3dm / glb with Draco-compressed
  glTF, decoded by the Rust extension's pure-Rust Draco decoder, so it also works in the
  browser). Heights are ellipsoidal, so each building is kept relative to its own lowest point
  and stood on the model's lowest ground under it; roofs are rasterised onto the grid (highest
  surface per vertex; hairline cracks between roof polygons are filled). OSM footprints that
  overlap a PLATEAU building give way to it. Only each tile's header and glTF part are read (the
  per-building attribute tables, 25-75% of a tile, are skipped); still ~30 MB per km² in central
  Tokyo, so a job takes at most 25 km² and 250 MB of geometry (else OSM heights).
- **Elsewhere: Overture Maps** buildings (zoom-14 PMTiles of the latest release, found through its
  STAC catalogue): where OSM's height is the 5 m default, Overture's height or floors (OSM tags
  again in most places, Microsoft ML heights in the US and Canada); footprints from other
  datasets (Microsoft / Google / Esri) where OSM has no building. Up to 100 tiles (~15 x 15 km).

GHS-OBAT (per-building heights averaged from a 100 m raster) and GlobalBuildingAtlas
(non-commercial heights, terabyte-scale downloads) were considered and left out.

**Bridges** come from the LiDAR bridge class, or from OSM bridges where the data has none (deck
from the surface heights on the bridge, ends on the ground). OSM bridges are drawn wider than
many decks: cells beside a classified deck take that deck's height (the returns there are
railings, lamp posts, lift-bridge towers and cables), and a low percentile over ~30 m along each
OSM-only deck removes posts and towers (the deck is the lowest surface above the ground). Every
deck is smoothed (a ~10 m median, then small steps such as the Skyway's twin decks softened into
a ramp; larger steps between bridges stay).

**Clean walls**: in every colouring, buildings and bridge decks are regions of their own, cut
along smooth outlines, each with its own top (roof / deck height, continued past the outline so
the cut takes it): their side walls are clean vertical cuts. As part of one height surface they
dropped to the ground across one grid cell along the staircase edge of their raster masks -
sloped sawtooth walls that cost print time. In land cover a deck is a region inside the roads
(road colour); with one colour, buildings and bridges are parts of their own in that colour;
with elevation bands each is banded by its own height (walls striped by band).
The result panel lists the share of each source and the **credits** their licences require
(also written into the 3MF description). Nearly all sources allow commercial use of derived
products (prints) with attribution; the few with extra conditions are still used, and a model
that uses them carries a licence warning saying what to check (see *Limits*).

| Region | Source | Data | Resolution |
|---|---|---|---|
| Global | Copernicus DEM 30 m (global) | surface model | 30 m |
| Global | FABDEM 30 m bare earth (global, non-commercial) | terrain | 30 m |
| Global | ALOS World 3D 30 m (JAXA, global) | surface model | 30 m |
| Global (~150 national / regional models) | Mapterhorn terrain tiles | terrain | 0.25–20 m |
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
| UK, Ireland, Spain, Italy | Tuscany LiDAR DTM 1 m 2019-21 (Italy, partial, non-commercial) | terrain | 1 m |
| UK, Ireland, Spain, Italy | South Tyrol DTM/DSM 2.5 m (Italy) | terrain + surface | 2.5 m |
| UK, Ireland, Spain, Italy | IGN MDT05 5 m (Spain) | terrain | 5 m |
| Central Europe | Slovenia LiDAR (GURS CLSS / ARSO LSS, COPC mirror) | classified point cloud (COPC) | point cloud |
| Central Europe | Poland LiDAR (GUGiK, COPC mirror) | classified point cloud (COPC) | point cloud |
| Central Europe | Wallonia LiDAR 2021-2022 (SPW, COPC mirror) | classified point cloud (COPC) | point cloud |
| Central Europe | Saxony LiDAR (GeoSN, COPC mirror) | classified point cloud (COPC) | point cloud |
| Central Europe | Flanders LiDAR DHMV II 2013-2015 (Digitaal Vlaanderen, COPC mirror) | point cloud (COPC, ground / water classes; areas up to ~1.2 km) | point cloud |
| Central Europe | BEV ALS 1 m (Austria) | terrain + surface | 1 m |
| Central Europe | LGLN DGM1 1 m (Lower Saxony) | terrain + surface | 1 m |
| Central Europe | Geobasis NRW DGM1/DOM1 1 m (North Rhine-Westphalia) | terrain + surface | 1 m |
| Central Europe | LVermGeo DGM1/DOM1 1 m (Rhineland-Palatinate) | terrain + surface | 1 m |
| Central Europe | LAiV DGM1/DOM1 1 m (Mecklenburg-Western Pomerania) | terrain + surface | 1 m |
| Central Europe | Bayerische Vermessungsverwaltung DGM1 1 m (Bavaria) | terrain | 1 m |
| Central Europe | LGB DGM1 1 m (Brandenburg, Berlin) | terrain | 1 m |
| Central Europe | GeoSN DGM1 1 m (Saxony) | terrain | 1 m |
| Nordics, Baltics, Arctic | Maa- ja Ruumiamet 1 m (Estonia) | terrain | 1 m |
| Nordics, Baltics, Arctic | DTM-LT-1m (Lithuania, non-commercial) | terrain | 1 m |
| Nordics, Baltics, Arctic | NLS 2 m (Finland) | terrain | 2 m |
| Nordics, Baltics, Arctic | ArcticDEM 2 m (Arctic, surface model) | surface model | 2 m |
| Nordics, Baltics, Arctic | ÍslandsDEM 10 m (Iceland) | surface model | 10 m |
| Australia, New Zealand | LINZ LiDAR 1 m (New Zealand) | terrain + surface | 1 m |
| Australia, New Zealand | Queensland LiDAR 1 m | terrain | 1 m |
| Australia, New Zealand | MDBA river-corridor LiDAR 0.5-5 m (Murray-Darling Basin) | terrain | 0.5–5 m |
| Australia, New Zealand | Geoscience Australia LiDAR 5 m | terrain | 5 m |
| Australia, New Zealand | LINZ contour DEM 8 m (New Zealand) | terrain | 8 m |
| Japan, Hong Kong, Philippines | CEDD LiDAR 2020 0.5 m (Hong Kong) | terrain + surface | 0.5 m |
| Japan, Hong Kong, Philippines | GSI 1-10 m (Japan) | terrain | 1 m |
| Japan, Hong Kong, Philippines | Taal Volcano LiDAR 1 m 2014-17 (Philippines, CC BY-SA) | terrain + surface | 1 m |

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
  water edges wobble from cell to cell; rivers keep a tight outline). Land narrower than that
  (a pier or breakwater about one cell wide) is left out whole: the smoothing kept only its wider
  spots, a dashed line of specks into the water. A grid sample next to a
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
- **Ground-only jobs** (bare terrain coloured by a single colour or elevation bands) use only
  the ground: LiDAR is read one octree level coarser (3x fewer points, ~0.03 mm height difference
  on the printed model) and elevation models skip their surface model (DSM). On a 3.2 x 2.4 km
  model at full detail that halves the download. Land-cover models keep the finer level: their
  tree outlines need it.
- **Filling gaps**: where point clouds cover most of the area, the elevation models that fill the
  gaps are read only in 256 m blocks around them (plus a block of overlap for the datum offset),
  not over the whole area - on the St. John's harbour model, 52 MB instead of 211 MB. Cloud-
  optimized GeoTIFFs are read by fetching the tiles the warp needs in parallel first.
- **Jobs**: each runs in its own worker process (see [Server limits](#server-limits)). On servers
  with room for it (~7 GB and more) two run at once - one building its model (CPU and memory
  heavy, one at a time) while the next downloads its data - and a model's parts are meshed in two
  threads (~20% faster, ~25% more memory); smaller servers run one job at a time with one meshing
  thread (`MAPTOMESH_PIPELINE`, `MAPTOMESH_PART_WORKERS` override both).
- **3D preview**: models over 2 M triangles are simplified for display (600 k on phones and
  low-memory devices), quantised to 16-bit positions and gzip-compressed (a 6.9 M-triangle model:
  14 MB instead of 125 MB); the preview is cached with the job's files and by browsers. The
  downloads always have full detail.
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
.venv/Scripts/python tests/test_native_writer.py                # native 3MF writer == Python writer
.venv/Scripts/python tests/test_mapterhorn.py                   # offline PMTiles / MVT / ranking checks
.venv/Scripts/python tests/test_limits.py                       # offline job-limit model checks (4 GB server)
.venv/Scripts/python tests/test_browser_io.py                   # offline GDAL-free GeoTIFF reader / resampling / proxy allowlist
.venv/Scripts/python tests/test_shares.py                       # offline share-link store
.venv/Scripts/python tests/test_device_failures.py              # offline device-failure reports
.venv/Scripts/python tests/test_server_security.py              # offline API hardening (bounds, body caps, ids, limits)
.venv/Scripts/python tests/test_structures.py                   # offline bridge / OSM building-height checks
.venv/Scripts/python tests/test_buildings.py                    # offline PLATEAU / Overture building checks
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
Hong Kong, Khibiny, Teide, Everest (FABDEM); these pin "agencies' own services first") and Mapterhorn
(Lofoten, Møns Klint, Geiranger, and the St. John's 10 km bands, which skip the point clouds) -
each checks that the expected source supplied the data. Each output is re-parsed and every part is checked for open edges, duplicate edges,
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
| `MAPTOMESH_NATIVE` | `1` (use the native extension when it is installed; `0` = pure Python) |
| `MAPTOMESH_MEMORY_MB` | 80% of the machine's RAM (memory the server and its jobs may use; sets the limits below) |
| `MAPTOMESH_JOB_TIMEOUT_S` | `300` (a job running longer is stopped; time waiting for another model's build doesn't count) |
| `MAPTOMESH_MAX_QUEUE` | `8` (jobs queued or running; more are refused as "server busy") |
| `MAPTOMESH_KEEP_JOBS` | `10` (finished jobs whose files are kept; older ones are deleted) |
| `MAPTOMESH_MAX_LIDAR_TILES` | `400` (CanElevation tiles per job at 1000 cells, fewer at finer detail; over it: elevation models) |
| `MAPTOMESH_PIPELINE` | `2` with ~7 GB or more for jobs, else `1` (jobs in flight: one building while the next downloads) |
| `MAPTOMESH_PART_WORKERS` | `2` with ~6 GB or more per job, else `1` (threads meshing a model's parts; more threads, more memory) |
| `MAPTOMESH_MAPTERHORN` | `1` (`0` = don't use Mapterhorn's tiles at all) |
| `MAPTOMESH_MAPTERHORN_FIRST` | `1` (`0` = bare-terrain jobs always read the point clouds) |
| `MAPTOMESH_MAPTERHORN_URL` | `https://download.mapterhorn.com/` (or a mirror with the same files) |
| `MAPTOMESH_PROXY_GB_PER_HOUR` | `4` (data each client may pull through `/api/proxy` per hour, for models built in the browser) |
| `MAPTOMESH_PROXY_MAX_MB` | `1024` (largest single file the proxy forwards) |
| `MAPTOMESH_PROXY_CONCURRENT` | `64` (proxied downloads in flight across all clients; 16 per client) |
| `MAPTOMESH_KEEP_SHARES` | `200000` (share links kept; the least recently used go first) |
| `MAPTOMESH_SHARES_PER_HOUR` | `60` (new share links per client address per hour) |
| `MAPTOMESH_ESTIMATES_PER_HOUR` | `120` (download estimates per client address per hour; each reads file headers) |
| `MAPTOMESH_ESTIMATES_CONCURRENT` | `2` (download estimates computed at once, one per client; more are refused at once with 503 / 429) |
| `MAPTOMESH_COVERAGE_PER_HOUR` | `1200` (coverage checks per client per hour: `/api/coverage`, `/api/coverage/usgs`, `/api/lidar/tiles`) |
| `MAPTOMESH_COVERAGE_CONCURRENT` | `6` (coverage checks computed at once, two per client) |
| `MAPTOMESH_JOBS_PER_CLIENT` | `2` (server jobs one client may have queued or running) |
| `MAPTOMESH_KEEP_REVIEWS` | `20000` (reviews kept; the oldest go first) |
| `MAPTOMESH_REVIEWS_MAX_MB` | `200` (size budget of the stored reviews; the oldest go first) |
| `MAPTOMESH_FORWARDED_ALLOW_IPS` | `127.0.0.1` (reverse proxies trusted to set `X-Forwarded-For`; see below) |

**Deployment behind a reverse proxy.** Per-client limits (jobs, estimates, coverage, shares,
reviews, the data proxy) count against the client's address (IPv6: its /64). Behind nginx, Caddy
or a tunnel, that address comes from `X-Forwarded-For`, which uvicorn only believes from the
addresses in `MAPTOMESH_FORWARDED_ALLOW_IPS` (comma-separated IPs / CIDR ranges; `run.py` passes it
on). Set it to the proxy's address, make the proxy overwrite (not pass through) the header, and
never use `*` on a server that is reachable directly. Started some other way than `run.py`, give
uvicorn the same `--proxy-headers --forwarded-allow-ips`. The API has no public docs or schema
(`/docs`, `/openapi.json`), request bodies are capped (413), and error responses carry an error id
instead of internals: the details go to the server log.

**Link previews**: the page carries Open Graph / Twitter-card metadata, so links posted to Discord,
Slack, X, iMessage etc. show a preview. The server fills in absolute URLs (from
`MAPTOMESH_PUBLIC_URL`, default `https://maptomesh.com`). The default image is `web/og-image.jpg` (1200 x 630); a shared
`?job=<id>` link previews that model's top view instead. To regenerate the example image, generate
San Francisco (centre 37.7870, -122.4400, 10 x 8.5 km, 1000 cells, land cover, bare terrain,
z x2) and screenshot the 3D view alone with headless Chrome at 1200 x 630:
`/?job=<id>&embed=1&az=180&el=40&zoom=1.25` (`embed=1` hides everything but the 3D view).

## Server limits

A public server must survive anyone's largest request. Three safeguards
(`maptomesh/limits.py`, `maptomesh/worker.py`):

1. **Detail cap before a job starts.** Peak memory follows the number of grid vertices, not the
   area, so the server refuses a Detail whose estimated peak exceeds the memory a job may use,
   and tells the user the largest Detail that fits. The UI caps its Detail slider the same way.
2. **Point clouds only where they arrive in time.** Their download grows with the number of tiles
   and with the Detail; over the cap a job uses elevation models (seconds for any area).
3. **A worker process per job**, watched by the server: past `MAPTOMESH_JOB_TIMEOUT_S` (5 min) or
   its memory budget it is killed, and the user sees why in a dialog with retry advice (and a
   one-click retry at a lower Detail). A job killed by the operating system for lack of memory
   (Linux OOM killer; workers volunteer first) is reported the same way, and the server carries
   on. Finished jobs live on disk (`cache/output/<job>`, the last `MAPTOMESH_KEEP_JOBS` = 10),
   not in the server's memory; their links keep working across restarts.

Measured peaks (2026-11-19, St. John's; peak memory ≈ 0.3 GB + per-vertex cost × million
vertices; a 4:3 area at 1400 cells has 1.5 M vertices, at 2400 cells 4.3 M):

| | per million vertices | 2400 cells (4:3), dev machine |
|---|---|---|
| Land cover, building the model (1 thread / 2 threads) | 1.0 / 1.24 GB | 5.6 GB, 48 s |
| Elevation bands (1 / 2 threads) | 0.65 / 0.90 GB | 5.4 GB, 49 s (10 km circle, 5.8 M vertices) |
| Single colour | 0.67 GB | 3.1 GB, 17-24 s |
| Point-cloud download with classes (buildings, trees, land cover) | 1.06 GB | 3.2 GB at 2000 cells |
| Point-cloud download, ground only (bare terrain) | 0.33 GB | 1.6 GB at 2400 cells |
| Print tiles | none: 4 x 4 tiles peak lower than one piece | |

Cold point-cloud downloads at 1000 cells: Toronto 10 km (375 tiles) 86 s, Ottawa 20 km
(441 tiles) 74 s, Denver 10 km (USGS) 96 s - hence ~400 tiles / ~100 km² at 1000 cells, fewer at
finer detail (data grows with the square of the Detail). Cached re-runs take seconds.

**A 4 GB server** (`MAPTOMESH_MEMORY_MB` defaults to 3.3 GB): one job at a time with one meshing
thread, up to 2.8 GB per job. Largest Detail (cells on the long side):

| | square area | 4:3 area |
|---|---|---|
| Land cover, or any colours with buildings / trees (point clouds) | 1450 | 1680 |
| Land cover from elevation models only | 1500 | 1730 |
| Bare terrain, elevation bands or one colour | 1850 | 2140 |

At those sizes a job takes 20-40 s on the development machine once its data is cached, ~1-2 min
with a cold download (a slower server CPU adds to the build part); the 5-minute limit catches the
rest. The default Detail (600 cells) needs 0.5-0.7 GB.

## Building in the browser

Models are built in the visitor's browser: the same Python package runs as WebAssembly (Pyodide 314.0.8) in a Web Worker, with
the Rust extension and the LAZ decoder built for it. The server then only serves files, answers
NRCan tile lookups (`/api/lidar/tiles`) and streams data from hosts without CORS headers
(`/api/proxy`, allowlisted hosts only, per-client limits: `maptomesh/proxy.py`). The device's
memory sets the largest Detail (WebAssembly stops at 4 GB; phones get much less). The server
builds a model only for browsers that can't (no WebAssembly / module workers, under 2 GB of RAM),
or when a build on the device fails: the failure dialog then offers *Build on the server* for that
model (when the server accepts its Detail), and the failure is reported to the server
(`/api/device-failures`, `data/device_failures.sqlite`: the error, the model's settings and
browser facts, no IP addresses). Review them with `python -m maptomesh.device_failures [N]`. Plan
and measurements: [docs/plans/browser-wasm.md](docs/plans/browser-wasm.md); a bare test page with
timings: `/static/browser.html`.

- **Data cache**: downloaded data (GeoTIFF / point-cloud blocks and the files the code caches
  under `/cache`) is kept in the browser's private file system (`web/opfs-cache.js`, up to 2 GB on
  desktops / 512 MB on phones, at most 30% of the browser's quota; oldest data dropped first), so
  building the same area again downloads little. One tab uses it at a time; another tab building
  meanwhile works without it. Files replaced upstream are detected (ETag / Last-Modified) as on
  the server.
- **Python cache**: Python (Pyodide, ~23 MB with its packages) is kept in the browser's Cache
  Storage (`maptomesh-runtime-pyodide-<version>`, `web/browser-worker.js`) rather than left to the
  HTTP cache; our wheels and the package come from content-versioned URLs
  (`/api/browser/files/<hash>/<wheel>`, `/api/browser/package.zip?v=<hash>`), so they're kept until
  they change. Only the first build on a device downloads it. Starting it still takes a few
  seconds of CPU (more on phones), so once it's on the device the page starts it in the background
  when idle, for people who build there. Browsers keep Python and data only for secure pages
  (https, or localhost): over plain http (e.g. a phone reaching a development server by its LAN
  address) everything downloads again on every visit, and the log and the phone dialog say so.
- **Phones** see the download before building on the device: a dialog with the estimated size
  (`/api/estimate`, `maptomesh/estimate.py`: LiDAR exact from the COPC octree, elevation models
  and OpenStreetMap estimated) plus Python on the first build; *Download and build* or *Cancel*.
  Not asked again for an area already built there, or after *Don't ask again*.
- **Share links** (the result's *Share link*): only the model's settings are stored on the server
  (`/api/shares`, `data/shares.sqlite`, `maptomesh/shares.py`), never model files, so nobody can
  hand others a crafted 3MF through the site. Opening `/?s=<id>` loads the settings and builds the
  model right away. Server-built results keep their `/?job=<id>` links while their files last.

- GeoTIFFs are read by `maptomesh/geotiff.py` (no GDAL in the browser): LZW, Deflate, ZSTD,
  PackBits, predictors 2 / 3, BigTIFF, overviews. LERC isn't supported, so New Zealand comes from
  Mapterhorn (which carries LINZ's 1 m model).
- Wheels for Pyodide live in `web/pyodide/`; rebuild them after changing `native/` (Linux, or WSL
  on Windows; installs its toolchain under `$HOME` on first run, ~2 min, ~2.7 GB):

  ```bash
  wsl -d Ubuntu -- bash /mnt/c/path/to/MapToMesh/native/build_pyodide.sh        # maptomesh_native
  wsl -d Ubuntu -- bash /mnt/c/path/to/MapToMesh/native/build_lazrs_pyodide.sh  # lazrs (LAZ), pinned
  ```

  laspy is pure Python, pinned from PyPI into the same folder (check its sha256 against PyPI):
  `pip download laspy==2.7.0 --no-deps --only-binary=:all: -d web/pyodide`.

  SciPy is trimmed to the parts maptomesh uses (`ndimage`, `spatial` and what they import: 6.6 of
  14 MB; a first visit downloads ~23 MB instead of ~30): `python native/trim_scipy.py` (checks the
  sha256 from Pyodide's lock file). A build that needs a part left out retries by itself with
  Pyodide's full SciPy, which that device then keeps using, and is reported to the device-failure
  log: add the subpackage to `KEEP` in the script and run it again.

## Limits and known behaviour

- Point clouds are read for up to 400 CanElevation tiles / 150 km² of USGS data at normal detail,
  fewer at fine detail (see [Server limits](#server-limits)); larger areas use elevation models.
  Each elevation model has its own area limit (e.g. Hong Kong 25 km², German 1 m state tiles
  20–120 km²); beyond it the next source is used.
- Not covered at high resolution (30 m there: FABDEM bare earth on land, with its non-commercial
  licence warning; Copernicus over water or where FABDEM can't be read): most of Asia, Africa and South America,
  and the parts of Europe and Australia no agency or Mapterhorn source covers (Mapterhorn's
  [coverage map](https://mapterhorn.com/coverage) shows them).
- **Trees** are drawn only when at least 90% of the land in the area has data that shows them
  (LiDAR, or elevation models with a surface model). Where most of it comes from bare-earth models
  (FABDEM, Mapterhorn; e.g. large areas beyond the LiDAR projects) the ground stays plain, with a
  note, instead of a forest patch ending at a data boundary (`model.TREES_MIN_KNOWN`).
- **Licence warnings**: some sources carry restrictions and are still used; respecting them is up
  to whoever uses the model. At equal detail they rank after unrestricted data, and a model that
  uses them shows a licence warning naming the dataset, what to check and where (result panel,
  notes and the 3MF description). They are:
  - Scotland's LiDAR phases II/III (the survey contractor retains the commercial copyright) and
    Historic Environment Scotland sets (Non-Commercial Government Licence);
  - Japan's GSI DEM through Mapterhorn (GSI's raw survey data: publishing products made from it
    may need the user's own approval under Japan's Survey Act; GSI's own tiles, used first, don't);
  - FABDEM 30 m (CC BY-NC-SA, non-commercial). Being bare earth it ranks before the Copernicus
    surface model, so land with nothing finer (most of Asia, Africa, South America) gets this
    warning; sea still comes from Copernicus;
  - ALOS AW3D30 (JAXA asks commercial users to notify it; only used where Copernicus can't be read);
  - Tuscany's 1 m LiDAR DTM 2019-21 (CC BY-NC-SA; Mapterhorn's open 10 m model is used first on
    grids of 20 m and more);
  - Lithuania's DTM-LT-1m (HNIT-Baltic's service terms: non-commercial use inside Esri ArcGIS only);
  - Flanders' DHMV II point clouds (public-interest use only; offered for areas up to ~1.2 km,
    where they are finer than the open DHMV II 1 m models that cover everything larger);
  - the Murray-Darling Basin Authority's river-corridor LiDAR DEMs (most state no licence; the
    warning names the surveys used);
  - Taal Volcano LiDAR (CC BY-SA 4.0, ShareAlike).
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
  [Elevation sources](#elevation-sources)), directly or through
  [Mapterhorn](https://mapterhorn.com/attribution) (code BSD-3; the data keeps each source's
  licence); each model lists the exact credit lines.
- Map tiles, roads, water and building footprints: © OpenStreetMap contributors (ODbL), served by
  [OpenFreeMap](https://openfreemap.org/); place search by Nominatim.
- Please respect the tile and Nominatim usage policies; everything is cached locally.
- Selling prints? Copy the **Credits** shown with the model (also in the 3MF description) plus
  "© OpenStreetMap contributors" into the listing or a label. If the model shows a **licence
  warning**, the data it names has extra conditions: check them first.

## License

[PolyForm Noncommercial 1.0.0](LICENSE.md) with an additional permission for generated models:

- **Free for personal and other noncommercial use**: run, modify and share the software.
- **Generated models and prints are yours to use and sell**, including commercially. The
  licensor claims no rights in them (third-party data credits above still apply).
- **Not allowed without permission**: selling or bundling the software itself, or offering it
  (or a modified version) as a commercial product or paid service.

This is a source-available licence, not an OSI open-source licence. See [LICENSE.md](LICENSE.md)
for the full terms.
