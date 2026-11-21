# Generating models in the browser (WebAssembly)

Status (2026-11-20): **phases 0-5 done: models are built in the browser by default.** "Build on:
This device" (next to Generate) builds any model in the browser: point clouds (NRCan, USGS Entwine
/ Planetary Computer, IGN), every elevation provider except LINZ (LERC), OpenStreetMap layers, the
3D preview and downloads, with downloaded data kept between visits. Tested in Chrome and Firefox.
The server builds for browsers that can't, and when chosen. Share links carry settings only.
Results of each round are at the end.

## Goal

Move the heavy work (data download, surface, meshing, 3MF writing) from the server to the user's
browser, so the server's job is serving the page, streaming data the browser can't fetch itself,
and storing reviews. Server memory and CPU then no longer limit how many people can use the site
at once, and the job size limits that protect a 4 GB server become per-device limits.

## Approach: run our Python as WebAssembly (Pyodide)

[Pyodide](https://pyodide.org) is CPython compiled to WebAssembly, with numpy, scipy, pyproj,
pydantic and Pillow prebuilt. The existing pipeline (`maptomesh/`) runs in a Web Worker almost
unchanged; a small runtime layer (`maptomesh/browser.py`) swaps the parts that assume a server:

| Server | Browser |
|---|---|
| `requests` + SQLite range cache (`httpcache.RANGE_CACHE`) | `fetch()` from the worker; in-memory block cache with the same interface |
| Thread pools for parallel downloads | Tasks run one at a time, but downloads overlap: batches of concurrent `fetch()` calls (JavaScript Promise Integration, `run_sync`), and *hints* (`httpcache.hint`, `pointcloud._hint_sources`, GeoTIFF URLs found in queued tasks) start downloads in the background; at most 6 requests per host. Serial synchronous requests where Promise Integration is missing |
| GDAL (rasterio) to decode and reproject rasters | `maptomesh/geotiff.py` (GeoTIFF / COG reader; tiles decoded by the Rust extension), Pillow for WebP / PNG tiles, pyproj + scipy resampling approximating GDAL's (`dem_readers._reproject_numpy`) |
| Rust extension (`native/`) | The same crate built for Pyodide (`native/build_pyodide.sh`, WSL / Linux), plus `lazrs` (`native/build_lazrs_pyodide.sh`) |
| NRCan tile index in memory | Asked of the server: `/api/lidar/tiles` (the index is 22 MB) |
| Results written to `cache/output/<job>` | Same files in the worker's in-memory file system, handed to the page as downloads |
| Disk caches under `cache/` (range blocks in SQLite, tiles, indexes, elevation grids) | The browser's private file system (OPFS, `web/opfs-cache.js`): range blocks and the files under `/cache`, in four pack files read and written synchronously |
| Process per job, memory watchdog | Each job in its own worker; the browser tab is the limit |

A full rewrite in Rust would be faster (multi-threaded, ~3 MB download instead of ~30 MB) but
means re-implementing ~9,000 lines including GDAL's warping and the scipy image filters. Pyodide
first, with the hot spots in Rust (the server's extension, built for WebAssembly).

## What the investigation found

**Data access (CORS).** Probed one real file per host (`_scratch/wasm/cors.py`) and fetched byte
ranges from Chrome 157.

- Direct from the browser: NRCan (HRDEM, LiDAR), USGS 1/3" and Entwine, Planetary Computer, FABDEM,
  ArcticDEM, NZ, swisstopo, IGN, Austria, Scotland, NRW, Finland, PDOK, Japan GSI, Environment
  Agency, the ArcGIS / WMS / WCS services, Mapterhorn, OpenFreeMap, Nominatim.
- Mapterhorn, USGS Entwine, NRW, Finland and PDOK reject the CORS preflight for `Range`, but Chrome
  sends a single `bytes=a-b` range without one, so they work. Firefox / Safari untested: fall back
  to the proxy when a direct fetch fails.
- No CORS (must go through the proxy): Copernicus on AWS, the Wales COG, Taal, Iceland, Estonia,
  the Irish and Hong Kong zips, Bavaria, Saxony. Copernicus matters least: Mapterhorn carries
  Copernicus GLO-30 as its filler.

**Compute.** Surface → parts → 3MF → previews on identical St. John's input
(`_scratch/wasm/compute_bench.py`, `node/run.mjs`):

| Detail | Server (Python + Rust) | Server, Python only | Pyodide | Pyodide memory |
|---|---|---|---|---|
| 1000 cells | 2.1 s | 6.2 s | 10.6 s | 0.8 GB |
| 1600 cells | 5.1 s | 15.5 s | 26.4 s | 1.7 GB |

Memory matches the server's cost model (`limits.py`). Output differs by ~10 triangles in 1.5 M
(floating-point differences). WebAssembly memory is capped at 4 GB; phones kill tabs far sooner.

**Gaps in Pyodide.**

- Its GDAL decodes only LZW. Deflate (FABDEM, Copernicus, PDOK, Wales, geoid grid), LERC (NZ),
  ZSTD (Iceland) and WebP (Mapterhorn) fail, so GeoTIFF sources need our own decoder (Phase 1).
- `lazrs` (LAZ decompression) isn't packaged (Phase 2).
- wasm32: `np.intp` is 32-bit (one `np.repeat` failed); no threads (thread pools run inline).
- First download: ~30 MB without rasterio (runtime 13 MB, scipy 14 MB, pyproj 4 MB, numpy
  3 MB, ...; rasterio + GDAL would add ~15 MB), cached by the browser afterwards.

## Server role afterwards

1. Static files: the page, the Python package (`/api/browser/package.zip`), the Pyodide wheels;
   NRCan tile lookups (`/api/lidar/tiles`).
2. **Streaming proxy** for hosts without CORS: `GET /api/proxy?url=...` passes the request
   (with `Range`) through and streams the response without buffering it. It must only fetch URLs
   on an allowlist of data hosts (scheme `https`, exact host match, no redirects to other hosts),
   or it becomes an open proxy that anyone can use to reach arbitrary or internal addresses.
   Rate- and size-limited per client.
3. Reviews and share links. Share links store the **settings only** (`/api/shares`,
   `maptomesh/shares.py`); opening one sets up the model, built on the visitor's device when they
   press Generate. Uploading
   finished models was ruled out: the site would then serve 3MF files made by anyone to everyone,
   and slicers have had 3MF parsing vulnerabilities, so a crafted file could reach other users
   under the site's name.
4. Optional fallback generation for devices that can't (old browsers, low memory), with the
   existing limits.

## Phases

0. **Prototype** - done: Mapterhorn + OpenStreetMap, `/static/browser.html` (timings page).
1. **Data layer** - done: `geotiff.py` (LZW, Deflate, ZSTD, PackBits, predictors, BigTIFF,
   overviews; tiles decoded in Rust), the allowlisted streaming proxy with automatic fallback, all
   providers. Left: LERC (only LINZ NZ uses it; Mapterhorn carries LINZ 1 m).
2. **Point clouds** - done: `lazrs` wheel, COPC (NRCan, Planetary Computer, IGN) and USGS Entwine
   through the browser range cache, NRCan tile lookups on the server. Left: per-device download
   budgets (a LiDAR job can pull 100+ MB onto a phone; the tile caps of `limits.lidar_caps` apply).
3. **UI** - done: "Build on" next to Generate; progress, log, failure dialog (device-specific
   "too large" before starting, out-of-memory after), 3D preview and downloads from blob URLs, the
   Detail cap from the device's memory (`deviceMemoryMb` in `web/app.js`: half the RAM on desktops,
   a third on phones, at most 3.6 / 1.6 GB). Share links: settings only, built on open (`/?s=<id>`).
4. **Speed and reach** - done: Rust extension (2-2.4x on the build); persistent data cache (OPFS:
   2 GB on desktops, 512 MB on phones, at most 30% of the quota; one tab at a time, others build
   without it); Firefox confirmed by the user. Left: Safari (its path without Promise Integration
   runs one request at a time, simulated in Chrome with `?nojspi`).
5. **Switch over** - done: models are built on the device; no "Build on" choice. The server
   builds where the browser can't (no WebAssembly / module workers / gzip streams, under 2 GB of
   RAM), and a model whose device build failed when the user accepts the failure dialog's offer;
   every device failure is reported (`maptomesh/device_failures.py`) for review. Left: per-device download budgets for phones (a LiDAR job can pull 100+ MB).

## GPU (WebGL / WebGPU)?

- **3D preview:** already on the GPU (three.js, WebGL). It now draws only when something changes
  (camera, resize, a new model) instead of every frame, so an idle view costs no GPU time or
  battery. WebGPU would not make one static mesh of at most a few million triangles faster, and
  isn't available everywhere yet. The preview's remaining CPU cost is decoding and computing
  creased normals (`toCreasedNormals`) on the page; those could move into the worker if previews
  feel slow on phones.
- **Building the model:** not worth it now. With the Rust extension the CPU part of a build is
  5-12 s (1000-1600 cells); what's left in Python is mostly colour regions, the partition into
  printable parts and iso-lines: branchy, graph-like work that suits a GPU poorly. The parts that
  would suit it (resampling, smoothing filters, binning points into the grid) take 1-3 s. Pyodide
  has no WebGPU binding, so each would be rewritten in WGSL with a CPU twin for devices without
  WebGPU and kept equal to the server's results. LiDAR jobs are dominated by downloads, which the
  data cache addresses.
- **The bigger lever is threads:** Pyodide runs on one core. The Rust extension could use several
  with WebAssembly threads (rayon), which needs `SharedArrayBuffer` and so cross-origin isolation
  (COOP / COEP headers): every cross-origin resource (CDN, map tiles, data hosts) must then allow
  it (CORP / CORS, or COEP `credentialless`, which Safari lacks). A separate investigation if build
  time becomes the complaint.

## Risks and open questions

- **Agency load.** Each browser downloads its own data; the server's shared cache no longer
  absorbs repeat requests. Mapterhorn-first keeps most jobs off agency servers; the proxy can cache
  what passes through it; OPFS caches per user.
- **Phones.** Memory (tabs killed around 1-2 GB, not measured), CPU and mobile data. Smaller caps,
  server fallback.
- **JavaScript Promise Integration** (parallel downloads from synchronous Python) ships in Chrome
  and Edge; elsewhere downloads run one at a time until Phase 4.
- **Determinism.** Results differ slightly from the server's; tests compare with tolerances.
- **Licences.** Unchanged: the same warnings and attribution go into the 3MF.

## Round 5: smaller first download (trimmed SciPy)

Pyodide has no tree shaking (Python imports are decided at run time; pyodide-pack, the project's
own bundler, is unmaintained). SciPy was the biggest file (13.9 of ~30 MB) and maptomesh uses only
`scipy.ndimage` and `scipy.spatial.cKDTree`. Running the integration cases and offline tests with
every import recorded showed the same 8 subpackages each time, all imported when those two load
(a static import closure would keep 12.7 MB: `_lib` refers to nearly everything lazily).
`native/trim_scipy.py` keeps them and drops the rest and the test suites: **6.6 MB**. A first
visit now downloads **23.6 MB** (Firefox, fresh profile; ~30 MB before). Checked: Firefox and
Chrome builds (LiDAR land cover with shores, Mapterhorn), and an over-trimmed wheel without
`spatial`: the build failed on the missing module, the worker restarted with the full SciPy, the
model finished (28 s), the device keeps the full SciPy from then on and the failure was reported.

Not done: replacing `cKDTree` (exact distances to the shore iso-line, `model._iso_distance`)
would drop `spatial`, `sparse` and `constants` too (~2 MB), but needs an exact replacement (e.g.
in the Rust extension) to keep the shore shapes unchanged.

## Round 4 results (Python cache, phone download warning)

- **Python cache.** Firefox didn't keep Python between visits (the HTTP cache), so the worker keeps
  it in Cache Storage itself: a cache-first `fetch` for the versioned Pyodide CDN path (Pyodide
  fetches its wasm, standard library and packages through the global `fetch`; `pyodide.asm.mjs` is
  loaded the same way and handed over as `createPyodideModule`), and for our wheels and package
  under content-versioned URLs. laspy comes from our server too (pinned wheel), not PyPI on every
  start. Chrome and Firefox 160 (headless, WebDriver BiDi: `_scratch/wasm/ff_runtime.py`): after
  the first build nothing is downloaded. Starting Python still takes ~5 s of CPU on this PC (more
  on phones), which looked like a download: the log now says "Starting Python (kept on this
  device)", and the page starts it in the background once it's kept (a build after a reload then
  starts at once: 4 s for a small Mapterhorn model). Plain http pages (a phone on a LAN address)
  get no Cache Storage or OPFS at all; the log and the phone dialog say so.
- **Phone download warning** (`maptomesh/estimate.py`, `/api/estimate`): measured against real
  builds on an emulated phone, fresh data each time:

| Area | Source | Estimate | Downloaded |
|---|---|---|---|
| St. John's 2 x 1.5 km, land cover, 500 | NRCan LiDAR (COPC octree, exact) | 81 MB | 82.9 MB |
| Seattle 2 x 1.5 km, land cover, 500 | USGS Entwine (points x 12 B) | 37.7 MB | 38.3 MB |
| Halifax 4 x 3 km, bands, 600 | HRDEM (COG headers, every overlapping item) | 68 MB | 70.2 MB |
| Zermatt 6 x 6 km, bands, 600 | swissALTI3D (COG headers) | 66 MB | 68.3-74.9 MB |
| Bath 3 x 3 km, bands, 600 | EA WCS (15 B per cell and layer) | 11.2 MB | 10.9 MB |
| Mt Fuji 8 x 8 km, bands, 600 | GSI tiles (formula) | 14.1 MB | 11.2 MB |
| Denver 5 x 4 km, bare terrain, 500 | Mapterhorn (whole tiles counted) | 3.9 MB | 4.9 MB |
| Valparaiso 6 x 4 km, bands, 600 | FABDEM + Copernicus (COG headers) | 3.4 MB | 6.8 MB |

  Region boxes cross borders (IGN's France box covers Zermatt, where IGN has no data): when the
  first choice is known only by its region, a provider of the same detail measured from its files
  is taken if larger, so the warning doesn't fall short. Not asked again for an area already built
  on the device.

## Round 3 results (default switch, data cache, share links)

Chrome 157, main page, one fresh profile kept across reloads (`_scratch/wasm/ui_e2e.py`); St. John's
2 x 1.5 km, land cover from NRCan LiDAR, 500 cells. Cache logic tested on its own with mocked OPFS
files and Pyodide's real file system (`_scratch/wasm/node/cache_test.mjs`: rotation, torn
records, a second tab, storage full, Python file operations across a reload).

| Run | Time | Downloaded | From the data cache |
|---|---|---|---|
| First build (empty cache) | 33 s | 82.9 MB | - |
| Same model after a page reload (new worker and Python) | 15 s | 0 MB | 2.5 MB (the cached elevation grid; the point clouds weren't needed) |
| Cancelled mid-run, built again (the cache's files released) | 9 s | 0 MB | 0.4 MB |
| Share link opened with other saved settings | 14 s | 0 MB | 2.5 MB |

- "Build on" defaulted to this device; the share link's settings replaced the saved ones, the
  About dialog stayed closed, the build started by itself; unknown ids answer 404.
- 3D view: 0 frames drawn in 3 s idle (every frame before), redraws while dragged, the damping
  eases out in ~3.6 s (`_scratch/wasm/viewer_idle.py`).
- `/api/browser/package.zip` now has an ETag: returning visitors get a 304 instead of the zip.

## Round 2 results (phases 1-4)

Chrome 157, this development PC, a fresh browser profile per run (nothing cached), "This device".

**Rust extension in Pyodide** (`compute_bench.py`; byte-identical 3MF entries to the native build):

| Detail | Pyodide + Rust | Pyodide, Python only | Server (all cores) |
|---|---|---|---|
| 1000 cells | 5.3 s, 0.58 GB | 10.5 s, 0.79 GB | 2.2 s |
| 1600 cells | 11.7 s, 1.34 GB | 28.0 s, 1.69 GB | 5.3 s |

**Whole jobs** (download included; elevation bands at 600 cells unless noted):

| Area | Source | Job | Downloaded |
|---|---|---|---|
| St. John's 2 x 1.5 km, land cover, 600 | NRCan LiDAR (COPC), 4.8 M points | 25 s | 83 MB |
| Seattle 2 x 1.5 km, land cover, 600 | USGS Entwine, 104 octree nodes | 20 s | 38 MB |
| St. John's 4 x 3 km | HRDEM COGs (LZW) | 7 s | 38 MB |
| Chamonix 6 x 4.5 km | IGN LiDAR HD (COPC) | 106 s | 37 MB |
| Zermatt 6 x 6 km | swissALTI3D COGs + Mapterhorn (Italy) | 49 s | 26 MB |
| Bath 3 x 3 km | Environment Agency WCS | 8 s | 5 MB |
| Mt Fuji 8 x 8 km | GSI PNG tiles (112) | 8 s | 11 MB |
| Valparaiso 6 x 4 km | FABDEM + Copernicus (through the proxy) | 7 s | 5 MB |

- Downloads dominate the slow ones (IGN's server: 98 s of waiting for 37 MB; the server took 74 s
  for the same area). Hints cut Fuji from 77 s to 8 s (the 112 tiles were fetched one by one).
- IGN answers HTTP 429 to a dozen simultaneous requests: at most 6 per host now, with backoff
  honouring Retry-After.
- The GeoTIFF reader matches GDAL pixel for pixel on real files from FABDEM, Taal, Copernicus,
  ArcticDEM (BigTIFF), USGS 1/3", swisstopo (`_scratch/wasm/tiff_check.py`) and on synthetic files
  in every codec / layout (`tests/test_browser_io.py`).
- Main page (`_scratch/wasm/ui_device.py`): a St. John's land-cover model built on the device in
  30 s, preview and blob downloads working, the device cap (1750 cells for land cover on an 8 GB
  machine) refusing 2400 cells with the dialog and a one-click retry.
- The server's own path is unchanged: the integration cases of the touched providers pass with the
  same triangle counts.

## Prototype results

`/static/browser.html` (Chrome 157, this development PC, St. John's 4 x 3 km). Script:
`_scratch/wasm/proto.py` drives the page headless; `server_job.py` runs the same job on the server.

**Setup** (Python runtime + numpy, scipy, pyproj, pydantic, Pillow, requests, laspy: 30 MB):
2.2 s runtime + 4.5-17 s packages + 2 s our code on a first visit (once 72 s, a cold CDN edge);
**4.9 s** for a returning visitor (browser cache).

**Jobs** (Mapterhorn data in the browser's HTTP cache after the first run; the server had the
elevation cached):

| Job | Browser | Server | Browser memory | Server memory |
|---|---|---|---|---|
| Elevation bands, 1000 cells | 13.4-15.7 s | 3.3 s | 0.80 GB | 0.73 GB |
| Elevation bands, 1600 cells | 26.0-27.2 s | 6.7 s ¹ | 1.76 GB | 1.96 GB ¹ |
| Land cover + OSM water / roads, 1000 cells | 30.1-30.4 s | 12.8 s ² | 1.00 GB | 0.88 GB ² |
| Land cover + OSM water / roads, 1600 cells | 50.5 s | 26.4 s ² | 2.27 GB | 2.49 GB ² |

¹ The server used HRDEM 1 m (finer than Mapterhorn's 2 m at this detail).
² The server used HRDEM's terrain + surface models (trees, buildings: 40-50% more triangles);
the browser has Mapterhorn's bare earth only.

- **Where the time goes** (land cover, 1000 cells, 30 s): elevation 4.3 s (2.4 MB of tiles),
  OpenStreetMap 4 s, surface 2 s, colour regions + meshing 8 s, 3MF 7 s, previews 2 s, STL zip
  2 s. The server's Rust extension writes the 3MF in 0.1 s and meshes ~4x faster: building it for
  Pyodide should roughly halve browser jobs.
- **Downloads per job:** 2.6-3.3 MB (Mapterhorn + OpenStreetMap), 9-18 requests.
- **Accuracy:** the GDAL-free resampling (pyproj + scipy, GDAL's widened bilinear kernel
  approximated) differs from GDAL by 1-2 cm RMSE at St. John's and 0.16-0.18 m around the
  Matterhorn (8 m and 20 m cells); triangle counts within 1%. On identical input, browser and
  native Python differ by 10-22 triangles in 0.8-1.4 M.
- **Without Promise Integration** (simulated with `?nojspi`): the same land-cover job took 30.1 s.
  Mapterhorn jobs make few requests, so one at a time costs little; point clouds will need the
  concurrent path.

**Code** (all inert on the server: each fallback runs only where rasterio is missing):

- `maptomesh/browser.py`: runtime layer: inline thread pools, `requests` and the range cache
  over the worker's `fetch`, provider filter, `run_job` (reuses `worker._pipeline`).
- `web/browser.html`, `web/browser-worker.js`: page and worker (Pyodide 314.0.8 from jsDelivr).
- `/api/browser/package.zip`: the package sources (server-only modules left out). This publishes
  the Python code to anyone who asks; fine for a free service, but worth knowing.
- `dem_readers.reproject_array`: pyproj + scipy fallback without GDAL.
- `providers/mapterhorn.py`: coverage polygons via `raster.fill_polygon`, WebP via Pillow,
  `affine.Affine` instead of rasterio's re-export.
- `partition.py`, `mesh.py`, `raster.py`: `np.repeat` counts as `np.intp` (32-bit in wasm).
- `osm._prefetch_tiles`: hook to fetch a job's tiles in one batch (no-op on the server).

**Not in the prototype:** other elevation sources (so no trees / buildings from surface models),
the proxy, the 3D preview, a per-device Detail cap (land cover at 2400 cells would need ~4.5 GB,
over the 4 GB WebAssembly limit), sharing, browsers other than Chrome.

**Suggested order from here:** Rust extension for Pyodide (speed, Phase 4 item first), then the
COG decoder + proxy (Phase 1: brings back HRDEM, USGS, IGN, ... and their surface models), then UI
integration with device caps (Phase 3), point clouds last (Phase 2: biggest downloads).
