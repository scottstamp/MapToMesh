# Plan: USGS 3DEP from the AWS Entwine (EPT) collection

Status: **implemented** (2026-11-19) in `maptomesh/usgs_ept.py` as the default *US LiDAR
source*. Planetary Computer stays selectable. Differences from the sketch below: there is no
separate point-source class; every COPC tile and every octree node is one job in
`pointcloud.fetch_lidar`. The user picks one US provider, so there is no dedupe step. Still
open: the US equivalent of HRDEM (last section).

## Why

Microsoft Planetary Computer's `3dep-lidar-copc` collection only contains projects up to about
2022, and many areas are missing entirely. All of these had **no** tiles there (checked
2026-11-19), but **are** in the AWS EPT collection:

| Place | EPT project(s) |
|---|---|
| Seattle | `WA_KingCo_1_2021` |
| Miami | `USGS_LPC_FL_Southeast_B2_2018_LAS_2019` |
| Manhattan | `NY_NewYorkCity` |
| Grand Canyon | `AZ_GrandCanyonNP_1_2019` |
| Yosemite | `CA_SierraNevada_11_B22` |
| Portland, ME | `ME_SouthCoastal_1_2020` |

Other advantages: no SAS token, kept up to date by USGS / Hobu, 2,284 projects.

## The dataset

- Bucket `s3://usgs-lidar-public` (us-west-2), public, no auth. HTTP:
  `https://s3-us-west-2.amazonaws.com/usgs-lidar-public/<PROJECT>/ept.json`
- Project footprints: `resources.geojson` (8.8 MB, properties `name`, `id`, `count`, `url`) at
  `https://raw.githubusercontent.com/hobuinc/usgs-lidar/master/boundaries/resources.geojson`
  (also `https://usgs.entwine.io/boundaries/resources.geojson`).
- Format: [Entwine Point Tile](https://entwine.io/en/latest/entwine-point-tile.html) 1.1.
  - `ept.json`: `bounds` (cube) / `boundsConforming`, `span` 128, `schema` (X/Y/Z scale 0.001 +
    offsets), `dataType: laszip`, `hierarchyType: json`, `srs` EPSG:3857 (Web Mercator, metres;
    **no vertical CRS** - check Z against the Planetary Computer copy of a shared project before
    trusting it as NAVD88 metres).
  - Hierarchy: `ept-hierarchy/D-X-Y-Z.json` maps node keys to point counts; `-1` means the
    subtree continues in its own hierarchy file.
  - Data: `ept-data/D-X-Y-Z.laz`, one small LAZ file per octree node (root node of
    `NY_3County_2019` is 351 KB). Whole-file GETs, not range reads.

## Implementation sketch

1. `maptomesh/usgs_ept.py`
   - Download and cache `resources.geojson` (refresh monthly). Query the AOI against the
     project polygons, not just their bounding boxes.
   - Per project: read `ept.json` and the hierarchy, cached on disk.
   - Node selection: node bounds = root cube / 2^D. Node point spacing is about
     `cube_width / span / 2^D`. Descend while the node intersects the AOI (in EPSG:3857) and
     its spacing is coarser than `max(grid.step * 0.7, 0.25)`, the same rule as the COPC
     `resolution`.
   - Fetch the selected nodes through `RANGE_CACHE` (whole file = ranges from 0 to size), decode
     with `laspy.read(io.BytesIO(...))` (lazrs). Filter to the AOI.
2. Refactor `pointcloud._process_tile` into a point-source interface yielding
   `(x, y, z, classification, withheld)` in a known CRS, plus a footprint polygon. COPC tiles
   (NRCan, Planetary Computer) and EPT projects then share the `_Acc` / plane-fit / merge path.
   For EPT the footprint is the project polygon from `resources.geojson`.
3. Provider choice: prefer EPT for any project it has (a superset of Planetary Computer, no
   token). Keep Planetary Computer as the fallback if AWS is unreachable, or drop it.
   Project names differ slightly between the two (`USGS_LPC_...` prefixes), so dedupe by
   normalized name and year rather than exact string.
4. Coverage map: draw `resources.geojson` (simplified, like `projects.geojson` for NRCan)
   instead of the per-view STAC search. That also removes the "zoom in to see tiles" limit.
5. Tests: add Seattle (`WA_KingCo_1_2021`) and Grand Canyon (steep terrain) integration
   cases. Also compare one shared project read from EPT and from Planetary Computer: heights
   within ~0.1 m means the vertical datum matches.

Cost notes: EPT nodes are whole files, so an AOI needs more requests than COPC range reads, but
nodes are small and cache well. Use 6-8 parallel fetches, like the COPC path.

## Related idea: US equivalent of HRDEM

Planetary Computer `3dep-seamless` (1/3 arc-second ~10 m and 1 m DEM COGs) would give a fast,
classification-free source for big US areas. It would plug into the existing HRDEM code path:
STAC search + rasterio COG reads through `CachedHttpFile`.
