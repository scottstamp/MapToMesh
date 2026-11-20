# Plan: elevation sources outside Canada and the US

Status: **implemented** (2026-11-19) in `maptomesh/providers/` (one module per region) on top of
`maptomesh/dem.py` (best-available composite) and `maptomesh/dem_readers.py`; the README's
"Elevation sources" table lists what is live. Not implemented directly: Sweden / Denmark
(accounts), Norway (unreachable from here), AHN6 COPC (undocumented), Czech ImageServer (slow),
the heavy plain-LAZ downloads, ELVIS.

**Update (2026-11-19): Mapterhorn** (`maptomesh/providers/mapterhorn.py`, https://mapterhorn.com)
serves ~150 open DTMs as one set of PMTiles: Denmark, Norway, Sweden, Portugal, Romania, Czechia,
Slovakia, Spain 0.5-2 m, Italian regions, Tasmania, the remaining German states and more are now
covered through it, and at equal detail it is preferred over the agencies' own services (see the
README's *Mapterhorn* section for the comparison). Japan's data there is GSI's raw 基盤地図情報
(licence warning; GSI's tiles stay first). Research notes below are kept for reference.

**Update (2026-11-19): restricted licences.** The collections under *Licence traps* are now used
where they can be reached anonymously, with a licence warning on every model that uses them and
ranked after unrestricted data of equal detail: FABDEM, ALOS AW3D30, Tuscany 1 m, Lithuania
DTM-LT-1m, Flanders DHMV II point clouds, the MDBA river-corridor DEMs and Taal. Skipped: TanDEM-X
90 m and Sweden's laser data (accounts required). See the table at the end.

Every source was probed on 2026-11-19 from St. John's, NL.
**Verified** means a request returned the stated result: an HTTP status, a successful range read, or a parsed
file. Everything else is from the source's documentation.

## How sources fit MapToMesh

| Tier | What it means | Work needed |
|---|---|---|
| **1** | Cloud-optimized data (COPC / EPT / COG) with a queryable index (STAC, WFS, listing, predictable names) | Plugs into the existing readers |
| **2** | Direct per-tile files, or an elevation service that cuts the area out server-side | A small new reader, or whole-file downloads |
| **3** | Portal, order, email, login, or bulk archives only | Not automatable |

A DEM-only source works like HRDEM: it has no classification, so buildings and trees come from OpenStreetMap.
Where both a DSM and a DTM exist, DSM minus DTM gives the raised objects.

### Cross-cutting needs

- **CRS overrides per source.** These files carry no CRS:
  - Luxembourg national COG → EPSG:2169
  - Flai COPC mirror → per dataset, e.g. Slovenia 3794, Poland 2180, Saxony 25833, Wallonia 3812
  - Shizuoka LAS → JGD2011 zone VIII
  - Emilia-Romagna WCS → EPSG:7791
- **Generic DEM-COG source.** Many tier-1 sources are just COGs behind a STAC search or a filename template. One
  configurable reader (index type + URL template + CRS + licence/attribution text) covers most of them.
- **Licence check per collection, not per country.** See *Licence traps* below.
- **Geo-blocking.** Norway (Kartverket / Geonorge) and Poland (GUGiK) were unreachable from Canada. Re-test both
  from the production server's location before building anything for them.

## Recommended order

1. **Global fallback: Copernicus DEM GLO-30.**
   - Format: COGs at `s3://copernicus-dem-30m`. The tile key is computable from latitude/longitude. **Verified**
     window read.
   - Gap: 25 tiles in Armenia/Azerbaijan (N38–N41, E43–E50) are missing at 30 m. Use GLO-90 there.
   - It is a surface model, so trees and buildings are included.
   - Licence: free, worldwide, any purpose. The required notice is: "produced using Copernicus WorldDEM-30 © DLR
     e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union
     and ESA; all rights reserved".
2. **France: IGN LiDAR HD** (classified COPC, Etalab Licence Ouverte 2.0, CORS `*`).
   - Classes 1–6, 9 and 17 are present. Coverage is about 85–90%, with full coverage targeted for end 2026.
   - Index: WFS `https://data.geopf.fr/wfs/ows`, layer `IGNF_LIDAR-HD_METADONNEE:metadata`, JSON output, bbox
     query. Each feature carries `url_npl` (the COPC), plus `url_mnt`, `url_mns` and `url_mnh` rasters.
     **Verified.**
   - The old `...:dalle` layers were withdrawn and now return HTTP 400.
   - Alternative: the national EPT at
     `https://data.geopf.fr/chunk/telechargement/download/lidarhd_fxx_ept/vpc-128/index.vpc`. It is a STAC
     FeatureCollection of 168 EPT blocks, which fits the existing EPT reader. **Verified.**
   - Nationwide DEM fallback: WMS-R `https://data.geopf.fr/wms-r`, layer `ELEVATION.ELEVATIONGRIDCOVERAGE.HIGHRES`
     (RGE ALTI 1 m), which returns float32 GeoTIFF. Requests are limited to 5010 px.
3. **New Zealand: LINZ.**
   - Data: national 1 m DEM and DSM COGs at `https://nz-elevation.s3.ap-southeast-2.amazonaws.com/`, under
     `new-zealand/new-zealand/{dem_1m,dsm_1m}/2193/<sheet>.tiff`.
   - Index: a static STAC (`catalog.json`). Sheets are NZTopo50 sheets, computable from NZTM coordinates.
   - Format: LERC-compressed COG. **Verified** that it decodes with the bundled GDAL.
   - Licence: CC BY 4.0. Credit the licensor named in each collection.
   - Extras:
     - The `nz-coastal` bucket adds seabed data down to about 25 m.
     - Classified point clouds exist only as plain LAZ on OpenTopography (tier 2).
4. **Switzerland: swisstopo STAC** at `https://data.geo.admin.ch/api/stac/v0.9/`. **Verified.**
   - `ch.swisstopo.swissalti3d`: DTM COG, 0.5 m and 2 m.
   - `ch.swisstopo.swisssurface3d-raster`: DSM COG.
   - `ch.swisstopo.swisssurface3d`: classified point cloud. Tiles from 2024 on are COPC; older tiles are `las.zip`.
   - Licence: commercial use allowed with mandatory "© swisstopo". The STAC says "proprietary", but the OGD
     conditions page allows it.
5. **Netherlands.**
   - **AHN4 DTM/DSM 0.5 m:** true COGs, CC0, nationwide. Index: the Atom feed
     `https://service.pdok.nl/rws/actueel-hoogtebestand-nederland/atom/dtm_05m.xml` (1373 tiles with bbox).
     **Verified.**
   - **AHN6 classified COPC:** `https://fsn1.your-objectstorage.com/hwh-ahn/AHN6/01_LAZ/AHN6_{year}_C_{x}_{y}.COPC.LAZ`,
     covering about two thirds of the country (complete in 2027). CC BY 4.0, "Bron: AHN". **Verified.**
     **Undocumented:** the index only lists `.LAZ`. Confirm with AHN before relying on the COPC files.
6. **Austria: BEV national 1 m DTM and DSM COGs.**
   - URL pattern: `https://data.bev.gv.at/download/ALS/DTM/20250915/ALS_DTM_CRS3035RES50000mN2800000E4600000.tif`.
   - 50 km tiles in EPSG:3035. The date folder varies.
   - Licence: CC BY 4.0. **Verified** with a 1 km window read in 3.2 s.
7. **UK.**
   - **Wales:** a single all-Wales 1 m COG each for DTM and DSM, OGL. **Verified.**
     - `https://dmwproductionblob.blob.core.windows.net/cogs/lidar/wales_dtm_32bit_cog.tif`
     - `https://dmwproductionblob.blob.core.windows.net/cogs/lidar/wales_dsm_32bit_cog.tif`
   - **Scotland:** a search API `POST https://api.remotesensing.data.gov.scot/search/product` over S3
     `srsp-open-data`. The 50 cm DTM/DSM are COGs. LAZ comes in 1 km tiles of about 58 MB. Coverage is partial;
     the national programme runs 2025–27. **Exclude the Non-Commercial Government Licence collections.**
8. **Nordic and Baltic DEMs, no auth.**
   - **Finland 2 m DEM:** Paituli STAC
     `https://paituli.csc.fi/geoserver/ogc/stac/v1/search?collections=nls_digital_elevation_model_2m_at_paituli`
     → Funet GeoTIFFs. They are tiled, without overviews, which is fine at 2 m. CC BY 4.0.
   - **Estonia 1 m DTM:** COGs per 5×5 km map sheet from geoportaal.maaruum.ee.
   - **ArcticDEM v4.1 2 m:** PGC STAC `https://stac.pgc.umn.edu/api/v1/search`, collection
     `arcticdem-mosaics-v4.1-2m`, COG, CC BY 4.0. Covers Norway, Sweden's mountains, Iceland, Greenland and the
     Faroes. It is a satellite DSM, so noisier than lidar.
9. **Luxembourg:** one national 0.5 m DTM COG, CC0. A matching DSM exists. The CRS tag is missing; use 2169.
   - `https://download.data.public.lu/resources/bd-l-lidar2024-releve-3d-du-territoire-luxembourgeois/20241223-093912/MNT_Lidar2024.tif`
10. **Lower Saxony:** DGM1 COG STAC `https://dgm.stac.lgln.niedersachsen.de/search?collections=dgm1`, CC BY 4.0.
    The STAC host's certificate was reported **revoked** by Windows; check it before use.

### Second phase: new small readers

- **WCS (GetCoverage → float32 GeoTIFF):**
  - England EA 1 m DTM/DSM composite (OGL, about 4 MB per km²):
    `https://environment.data.gov.uk/spatialdata/lidar-composite-digital-terrain-model-dtm-1m/wcs`
  - Australia GA 5 m LiDAR DEM (CC BY 4.0): works with `crs=EPSG:4283` only.
    `https://services.ga.gov.au/gis/services/DEM_LiDAR_5m_2025/MapServer/WCSServer`
  - Flanders 1 m DTM/DSM: use WCS 1.0.0.
    `https://geo.api.vlaanderen.be/DHMV/wcs`
  - Mecklenburg-Vorpommern: `geodaten-mv.de/dienste/dgm_wcs`
  - South Tyrol 0.5 m (CC0): `geoservices9.civis.bz.it/geoserver/ows`
  - Emilia-Romagna 0.5 m (CC BY 4.0)
  - Spain IGN 5 m: CC BY 4.0, but **int16 whole metres**, so it steps visibly on small prints.
- **ArcGIS ImageServer `exportImage`:**
  - **Queensland QldDem 0.5–1 m (CC BY 3.0 AU):** the best Australian source.
    `https://spatial-img.information.qld.gov.au/arcgis/rest/services/Elevation/QldDem/ImageServer/exportImage`
  - Czechia DMR5G: about 23 s per km².
  - NSW 5 m: derived from photogrammetry, slow.
- **XYZ elevation tiles (PNG, RGB-encoded):**
  - **Japan GSI:** 1 m nationwide.
    - Fallback chain: `dem1a_png` → `dem5a_png` → `dem5b_png`/`dem5c_png` → `dem_png`.
    - URL: `https://cyberjapandata.gsi.go.jp/xyz/<layer>/{z}/{x}/{y}.png`.
    - Decode: x = R·2¹⁶ + G·2⁸ + B. Height is 0.01·x m, or 0.01·(x − 2²⁴) when x > 2²³; x = 2²³ means no data.
    - Licence: category 2, attribution only ("国土地理院"). Commercial use is allowed and no Survey Act approval is
      needed for the tiles.
  - **Hong Kong 2020 0.5 m DTM/DSM:** zipped GeoTIFF sheets of about 5.7 MB, found via an Esri FeatureServer
    query. Free for commercial use.
- **Credentials (free, self-service), stored in server config:**
  - **Sweden, Lantmäteriet:** STAC `https://api.lantmateriet.se/stac-hojd/v1` with `dtm-cog` (1 m) and
    `dsm-skoglig-copc` (COPC, ground/water/bridge only, about 75% coverage). Uses HTTP Basic auth with a Geotorget
    account plus a per-product permission. CC BY 4.0.
  - **Denmark:** Dataforsyningen token for the DHM 0.4 m WCS. Point clouds need a Datafordeler key.
  - **Finland:** NLS API key, needed only for the official API. The Funet route needs none.
  - **Portugal:** DGT account for its STAC (2024 LiDAR, CC BY 4.0).

### Heavy, only if there's demand: plain LAZ downloaded whole

| Source | Size per tile |
|---|---|
| NZ OpenTopography | 0.1–45 MB per 480×720 m tile |
| Estonia | about 46 MB per km² |
| Finland 0.5 p/m² | about 18 MB per 9 km² |
| NRW | median 80 MB per km² |
| Bavaria | about 105 MB per km² |
| RLP | about 340 MB per km² |
| Latvia | uncompressed LAS, about 163 MB per km² |
| Catalonia | about 440 MB per km² |
| Castilla y León | about 54 MB per 4 km² |
| Japanese prefectures (Shizuoka, Tokyo, Yamanashi) | 26–243 MB per sheet |

**Flai `open-lidar-data` mirror:** a third-party COPC copy of Slovenia, Poland, Saxony, Wallonia, Flanders and
Luxembourg, with shapefile footprints. It works with the COPC reader as is. It is the only route to Polish data
from outside Poland, and the only one to Wallonia without 48–271 GB archives. It has no SLA and no CRS in the
headers.

## Not automatable (tier 3)

- **Australia ELVIS:**
  - The API `https://api.elevation.fsdf.org.au/elevation/downloadables?polygon=...` exists, but it returns 403 to
    non-browser clients.
  - The files sit in requester-pays S3 buckets (`nsw-elvis`, `ga-elvis`), so anonymous requests are refused.
  - Do not work around this. Only revisit it if Geoscience Australia offers a real API or written permission.
- **Paid or token-only:** Victoria (Vicmap), Western Australia (Landgate), BKG national DGM1, Hungary DDM5.
- **Portals or giant archives:**
  - Spain (CNIG basket)
  - Italy national (MASE)
  - Wallonia (official), Hesse, Berlin, Saarland, Saxony-Anhalt, Hamburg, Bremen
  - Slovakia (one 184 GB zip)
  - Slovenia CLSS 2025 (official)
  - Croatia, Lithuania, Northern Ireland
- **Asia, login or government-only:** Korea, Taiwan (1 m/5 m), India (CartoDEM), Indonesia (DEMNAS), Philippines
  (LiPAD).

## Licence traps (collections that look open but can't be used for sold prints)

**Update (2026-11-19): restricted sources are now included.** MapToMesh is run as a free service,
so the licence conditions fall to whoever uses a model: restricted data is used, ranked after
unrestricted data of equal effective resolution (`prefer = 2`), and every model that uses it
carries a licence warning (`licence_note` / `DemArrays.licence_notes`) naming the dataset, the
restriction, what to check before selling prints, and the terms URL, plus the required credit
line. Only sources reachable anonymously are used: no accounts, credentials, login walls,
browser-only checks or requester-pays buckets. Probed 2026-11-19 with `tests/probe_sources.py`.

| Collection | Problem | Status |
|---|---|---|
| Scotland Phase II/III, several Historic Environment Scotland sets | Non-Commercial Government Licence (HES); Fugro keeps the commercial copyright (II/III) | **Included** (already): DTM/DSM COGs, used after every unrestricted collection, per-collection warnings. Phase II LAZ point clouds not used (plain 58 MB/km² LAZ, no COPC). |
| Tuscany 2019–21 1 m DTM (Italian Remote Sensing) | CC BY-NC-SA, distributed for public bodies | **Included**: `rt_tos_dtm1m_irs`, Geoscopio WMS `wmsmorfologia` GetMap `image/tiff` (real float32, EPSG:25832, 0 = outside the surveys, 4096 px limit). Monte Amiata 1663–1731 m. Mapterhorn's open TINITALY 10 m ranks first on grids ≥ 20 m. The 2021 photogrammetric DSM (`iddsm2021.1m.rt`, Regione Toscana, no licence stated) is not used. |
| Flanders DHMV II point cloud | Official metadata: raw LiDAR for public-interest tasks ("taken van algemeen belang") only; the Flai mirror lists the free open data licence v1.2 | **Included**: `flai_be_vl`, Flai COPC mirror `BE/EODaS/LiDAR_DHMV_II-2013-2015` (EPSG:31370, 78 809 strip tiles, classes 1/2/9). Point clouds outrank all elevation models and have no preference order, so it is only offered for areas up to ~1.2 km (cells < 2 m at 600 cells), where it is finer than the open DHMV II 1 m DTM/DSM; ground matches that DTM to +2 cm (median) at Kemmelberg. |
| FABDEM V1-2 | CC BY-NC-SA 4.0 (non-commercial; commercial licences from Fathom) | **Included**: `fabdem`, Fondazione LINKS COG copy on Hugging Face (`huggingface.co/buckets/links-ads/fabdem/resolve/tiles/<10° folder>/<tile>_FABDEM_V1-2.tif`, anonymous, redirect to a CDN, range requests; STAC `collection.json` lists the 19 011 tiles). Sea is stored as 0 m and treated as no data. Bare earth, so it ranks **before** Copernicus (surface model) at 30 m: land with nothing finer gets the non-commercial warning. |
| TanDEM-X 90 m | Scientific use only, licence accepted at registration | **Skipped**: downloads need an EOC Geoservice account (registration + login); the public WMS would bypass the licence acceptance. 90 m is coarser than Copernicus anyway. |
| Lithuania DTM-LT-1m on ArcGIS Online | HNIT-Baltic service terms: non-commercial, inside Esri ArcGIS only; no distribution or reproduction without written consent | **Included**: `hnit_dtm_lt_1m`, the public item's ImageServer `utility.arcgis.com/usrsvcs/servers/fef66dec83c14b0295180ecafa662aa0/rest/services/DTM_LT2020/ImageServer` (`exportImage`, F32, EPSG:3346, 15000 × 4100 px limit). Note the "ArcGIS only" condition binds the service's user, i.e. MapToMesh itself, not just the buyer of a print. Only source for Lithuania (Mapterhorn excludes it). The Flai mirror also holds NŽT point clouds (`LT/NZT/2017`, `2019-2022`, `2025-2026` COPC) and Flai-derived 0.5 m DEMs with no licence stated; not used. |
| ALOS AW3D30 | Commercial use allowed, but JAXA asks to be **notified in advance**; credit ©JAXA | **Included**: `jaxa_aw3d30`, Microsoft Planetary Computer `alos-dem` COGs (`ALPSMLC30_N035E138_DSM.tif`, anonymous SAS token per container). Surface model, ranks after Copernicus: a fallback when Copernicus can't be read. JAXA's EORC site was down for maintenance during this check; the notification clause is from the earlier research. |
| Philippines Taal LiDAR | Provisional CC BY-SA 4.0 (ShareAlike; the FAQ asks for derived products to stay open and free) | **Included**: `phl_taal_lidar_1m`, public S3 `phil-lidar-taal-s3` (`Taal_DTM/E{x}N{y}_DTM.tif`, `Taal_DSM/...`, 1 km UTM 51N tiles, 4.8 MB uncompressed) with the tile list from the project's map page (`taal-20km-grid.js`, 1 275 tiles with DTM/DSM). Taal Lake has no data. The Entwine point cloud of the volcano island is not used. |
| Japan raw 基盤地図情報 DEM downloads, smartmaps/japan-geotiff-dem | Survey Act approval needed. GSI *tiles* are fine. | Used through Mapterhorn with a warning (GSI's tiles first). |
| MDBA river-corridor DEMs (Australia) | No licence stated on most services; Chowilla 2003 has contractor conditions (use only for the commissioned purpose); some state CC BY 3.0 AU | **Included**: `mdba_lidar`, 24 ImageServers under `gis.mdba.gov.au/arcgis/rest/services/ELEVATION` (0.5–5 m, 2003–2021, `exportImage` F32 with `noData=-9999`; the SRTM DEM-H service is not used). Finest first, CC BY surveys before unlicensed ones at equal resolution; the warning names the unlicensed surveys actually used. GA 5 m / Queensland 1 m rank first at equal detail. |
| Sweden laser data | Commercial use allowed, but the licensee takes on GDPR obligations | **Skipped**: Lantmäteriet's STAC (`api.lantmateriet.se/stac-hojd/v1`) is public, but its COPC / COG assets answer 401 (HTTP Basic, Geotorget account). The Flai mirror's `SE/LANTMATERIET/Laserdata_Skog` is empty. Sweden's open 1 m DEM (Markhöjdmodell, CC0) is already used through Mapterhorn. |
