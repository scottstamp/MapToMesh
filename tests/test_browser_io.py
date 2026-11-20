"""Offline checks for the browser build's I/O pieces (docs/plans/browser-wasm.md).

    python tests/test_browser_io.py

- ``geotiff``: GeoTIFFs written by GDAL (rasterio) in every codec / predictor / layout the
  elevation sources use are read back identically (window reads, overviews, nodata, CRS,
  transform), with the Rust tile decoder and with the pure-Python fallback.
- ``dem_readers``: the GDAL-free resampling stays close to GDAL's.
- ``proxy``: the allowlist (hosts, path prefixes, schemes, ports, POST rules).
"""

from __future__ import annotations

import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from maptomesh import geotiff, native, proxy  # noqa: E402

warnings.filterwarnings("ignore")


def _write(path, arr, **kw):
    import rasterio
    from rasterio.transform import from_origin
    prof = dict(driver="GTiff", width=arr.shape[-1], height=arr.shape[-2], count=1 if arr.ndim == 2 else arr.shape[0],
                dtype=arr.dtype, crs="EPSG:32622", transform=from_origin(370000.5, 5270000.25, 2.0, 2.0))
    prof.update(kw)
    with rasterio.open(path, "w", **prof) as w:
        w.write(arr if arr.ndim == 3 else arr[None])
    return path


def _cases():
    rng = np.random.default_rng(1)
    y, x = np.mgrid[0:300, 0:420]
    f32 = (100 + 30 * np.sin(x / 17) + 20 * np.cos(y / 11) + rng.normal(0, 0.1, x.shape)).astype("float32")
    f32[10:20, 30:60] = -9999.0
    i16 = (f32 * 10).astype("int16")
    u8 = np.stack([(f32 % 256).astype("uint8")] * 3)
    yield "deflate f32 pred3 tiled", f32, dict(compress="deflate", predictor=3, tiled=True, blockxsize=128, blockysize=128, nodata=-9999.0)
    yield "lzw f32 pred3 tiled", f32, dict(compress="lzw", predictor=3, tiled=True, blockxsize=256, blockysize=256, nodata=-9999.0)
    yield "lzw f32 striped", f32, dict(compress="lzw", nodata=-9999.0)
    yield "zstd f32 pred3 tiled", f32, dict(compress="zstd", predictor=3, tiled=True, blockxsize=128, blockysize=128)
    yield "deflate i16 pred2 striped", i16, dict(compress="deflate", predictor=2)
    yield "packbits i16", i16, dict(compress="packbits", tiled=True, blockxsize=128, blockysize=128)
    yield "none f32 bigtiff", f32, dict(BIGTIFF="YES", tiled=True, blockxsize=128, blockysize=128)
    yield "deflate u8x3 pixel-interleaved", u8, dict(compress="deflate", predictor=2, interleave="pixel")
    yield "deflate u8x3 band-interleaved", u8, dict(compress="deflate", interleave="band", tiled=True, blockxsize=128, blockysize=128)


def _check_reads(use_native: bool) -> None:
    import rasterio
    saved = native.ext
    if not use_native:
        native.ext = None
    try:
        with tempfile.TemporaryDirectory() as d:
            for k, (name, arr, kw) in enumerate(_cases()):
                p = _write(Path(d) / f"t{k}.tif", arr, **kw)
                if "zstd" in name and not use_native:
                    try:
                        import zstandard  # noqa: F401
                    except ImportError:
                        continue          # the pure-Python fallback needs the zstandard module
                if "bigtiff" in name or k == 0:
                    with rasterio.open(p, "r+") as w:
                        w.build_overviews([2, 4], rasterio.enums.Resampling.average)
                g = geotiff.GeoTiff(geotiff.BytesSource(p.read_bytes()))
                with rasterio.open(p) as r:
                    assert g.crs == 32622, (name, g.crs)
                    assert np.allclose(tuple(g.transform0)[:6], tuple(r.transform)[:6]), name
                    assert g.factors == r.overviews(1), (name, g.factors, r.overviews(1))
                    for band in range(1, r.count + 1):
                        win = (37, 251, 55, 399)
                        a = g.read(0, win, band)
                        from rasterio.windows import Window
                        b = r.read(band, window=Window(win[2], win[0], win[3] - win[2], win[1] - win[0])).astype("float32")
                        if r.nodata is not None:
                            b[b == np.float32(r.nodata)] = np.nan
                        assert np.array_equal(a, b, equal_nan=True), (name, band)
                    for lv in range(1, len(g.factors) + 1):
                        with rasterio.open(p, overview_level=lv - 1) as ro:
                            c = ro.read(1).astype("float32")
                            if ro.nodata is not None:
                                c[c == np.float32(ro.nodata)] = np.nan
                        assert np.array_equal(g.read(lv), c, equal_nan=True), (name, "overview", lv)
    finally:
        native.ext = saved


def test_geotiff_native() -> None:
    if native.ext is None or not hasattr(native.ext, "decode_tile"):
        print("  (Rust extension not built: skipped)")
        return
    _check_reads(True)


def test_geotiff_python() -> None:
    _check_reads(False)


def test_reproject_numpy() -> None:
    from affine import Affine

    from maptomesh import dem_readers
    from maptomesh.geo import AOI, Grid
    aoi = AOI(47.5615, -52.7126, 3000, 2000)
    for cells in (300, 900):
        grid = Grid.for_aoi(aoi, cells)
        x0, y0, x1, y1 = dem_readers.src_bbox(aoi, grid, 3857, 6)
        px = 2.4
        w, h = int((x1 - x0) / px) + 1, int((y1 - y0) / px) + 1
        yy, xx = np.mgrid[0:h, 0:w]
        src = (50 + 20 * np.sin(xx / 40.0) * np.cos(yy / 33.0) + 0.3 * np.sin(xx * 1.7)).astype("float32")
        tf = Affine(px, 0, x0, 0, -px, y1)
        a = dem_readers.reproject_array(src, tf, 3857, aoi, grid)
        b = dem_readers._reproject_numpy(src, tf, 3857, aoi, grid, "bilinear")
        ok = np.isfinite(a) & np.isfinite(b)
        assert ok.mean() > 0.99
        rmse = float(np.sqrt(np.mean((a - b)[ok] ** 2)))
        assert rmse < 0.15, (cells, rmse)


def test_ascii_grid() -> None:
    from maptomesh.dem_readers import _ascii_grid
    txt = (b"ncols 3\nnrows 2\nxllcorner 100.0\nyllcorner 200.0\ncellsize 2.0\nNODATA_value -9999\n"
           b"1 2 3\n4 -9999 6\n")
    g = _ascii_grid(txt)
    assert g.arr.shape == (2, 3) and np.isnan(g.arr[1, 1]) and g.arr[0, 2] == 3
    assert tuple(g.transform)[:6] == (2.0, 0.0, 100.0, 0.0, -2.0, 204.0)


def test_proxy_allowlist() -> None:
    ok = proxy.allowed
    assert ok("https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N47_00_W053_00_DEM/x.tif")
    assert ok("https://s3-us-west-2.amazonaws.com/usgs-lidar-public/CA_X/ept.json")
    assert not ok("https://s3-us-west-2.amazonaws.com/another-bucket/x")          # path prefix
    assert not ok("http://copernicus-dem-30m.s3.amazonaws.com/x")                  # https only
    assert not ok("https://copernicus-dem-30m.s3.amazonaws.com:8443/x")            # default port only
    assert not ok("https://user:pw@copernicus-dem-30m.s3.amazonaws.com/x")         # no credentials
    assert not ok("https://copernicus-dem-30m.s3.amazonaws.com.evil.example/x")    # exact host
    assert not ok("https://169.254.169.254/latest/meta-data/")
    assert not ok("https://localhost/")
    assert not ok("file:///etc/passwd")
    assert not ok("https://download.mapterhorn.com/x", "POST")                      # POST: searches only
    assert ok("https://planetarycomputer.microsoft.com/api/stac/v1/search", "POST")
    assert set(proxy.NO_CORS) <= {h for h, _ in proxy.RULES}


if __name__ == "__main__":
    for fn in (test_geotiff_native, test_geotiff_python, test_reproject_numpy, test_ascii_grid, test_proxy_allowlist):
        fn()
        print(f"{fn.__name__}: ok")
    print("all passed")
