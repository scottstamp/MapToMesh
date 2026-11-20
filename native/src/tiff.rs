//! GeoTIFF / COG tile decoding for the GDAL-free reader (`maptomesh/geotiff.py`), which the
//! browser build uses: decompression (LZW, Deflate, ZSTD, PackBits) and the TIFF predictors.
//! Pure Rust, so it builds for WebAssembly (Pyodide) as well as natively.

use std::io::Read;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

pub const NONE: u16 = 1;
pub const LZW: u16 = 5;
pub const DEFLATE: u16 = 8;
pub const DEFLATE_OLD: u16 = 32946;
pub const PACKBITS: u16 = 32773;
pub const ZSTD: u16 = 50000;

fn decompress(data: &[u8], compression: u16, expected: usize) -> Result<Vec<u8>, String> {
    match compression {
        NONE => Ok(data.to_vec()),
        LZW => {
            let mut d = weezl::decode::Decoder::with_tiff_size_switch(weezl::BitOrder::Msb, 8);
            let mut out = Vec::with_capacity(expected);
            let r = d.into_vec(&mut out).decode(data);
            // a tile's stream may stop without an end code, or carry junk after it: what decoded counts
            if out.is_empty() {
                r.status.map_err(|e| format!("LZW: {e}"))?;
            }
            Ok(out)
        }
        DEFLATE | DEFLATE_OLD => {
            let mut out = Vec::with_capacity(expected);
            flate2::read::ZlibDecoder::new(data)
                .read_to_end(&mut out)
                .map_err(|e| format!("Deflate: {e}"))?;
            Ok(out)
        }
        ZSTD => {
            let mut out = Vec::with_capacity(expected);
            ruzstd::decoding::StreamingDecoder::new(data)
                .map_err(|e| format!("ZSTD: {e}"))?
                .read_to_end(&mut out)
                .map_err(|e| format!("ZSTD: {e}"))?;
            Ok(out)
        }
        PACKBITS => {
            let mut out = Vec::with_capacity(expected);
            let mut i = 0;
            while i < data.len() && out.len() < expected {
                let n = data[i] as i8;
                i += 1;
                if n >= 0 {
                    let k = n as usize + 1;
                    let end = (i + k).min(data.len());
                    out.extend_from_slice(&data[i..end]);
                    i = end;
                } else if n != -128 && i < data.len() {
                    out.extend(std::iter::repeat_n(data[i], 1 - n as isize as usize));
                    i += 1;
                }
            }
            Ok(out)
        }
        other => Err(format!("unsupported TIFF compression {other}")),
    }
}

/// Undo predictor 2 (horizontal differencing of whole samples, in the file's byte order).
fn undo_horizontal(buf: &mut [u8], width: usize, rows: usize, bps: usize, spp: usize, big_endian: bool) {
    let row_len = width * spp * bps;
    for row in buf.chunks_exact_mut(row_len).take(rows) {
        match bps {
            1 => {
                for i in spp..row.len() {
                    row[i] = row[i].wrapping_add(row[i - spp]);
                }
            }
            2 | 4 | 8 => {
                let get = |b: &[u8]| -> u64 {
                    let mut v = 0u64;
                    for k in 0..bps {
                        let byte = if big_endian { b[k] } else { b[bps - 1 - k] };
                        v = (v << 8) | byte as u64;
                    }
                    v
                };
                let mask = if bps == 8 { u64::MAX } else { (1u64 << (8 * bps)) - 1 };
                let stride = spp * bps;
                for i in (stride..row.len()).step_by(bps) {
                    let prev = get(&row[i - stride..i - stride + bps]);
                    let cur = get(&row[i..i + bps]);
                    let v = cur.wrapping_add(prev) & mask;
                    for k in 0..bps {
                        let byte = (v >> (8 * k)) as u8;
                        if big_endian {
                            row[i + bps - 1 - k] = byte;
                        } else {
                            row[i + k] = byte;
                        }
                    }
                }
            }
            _ => {}
        }
    }
}

/// Undo predictor 3 (floating point: byte differencing, then bytes regrouped most significant
/// first). The result is little-endian whatever the file's byte order.
fn undo_float(buf: &mut [u8], width: usize, rows: usize, bps: usize, spp: usize) {
    let wc = width * spp;
    let row_len = wc * bps;
    let mut tmp = vec![0u8; row_len];
    for row in buf.chunks_exact_mut(row_len).take(rows) {
        for i in spp..row_len {
            row[i] = row[i].wrapping_add(row[i - spp]);
        }
        tmp.copy_from_slice(row);
        for k in 0..wc {
            for b in 0..bps {
                row[k * bps + b] = tmp[(bps - 1 - b) * wc + k];
            }
        }
    }
}

pub fn decode(
    data: &[u8],
    compression: u16,
    predictor: u16,
    width: usize,
    rows: usize,
    bps: usize,
    spp: usize,
    big_endian: bool,
) -> Result<Vec<u8>, String> {
    let expected = width * rows * bps * spp;
    let mut out = decompress(data, compression, expected)?;
    out.resize(expected, 0); // short (truncated) tiles: zeros, as libtiff does
    match predictor {
        1 => {}
        2 => undo_horizontal(&mut out, width, rows, bps, spp, big_endian),
        3 => undo_float(&mut out, width, rows, bps, spp),
        p => return Err(format!("unsupported TIFF predictor {p}")),
    }
    Ok(out)
}

/// Decode one TIFF tile or strip to raw pixel bytes (`width` x `rows` x `samples` samples of
/// `bytes_per_sample`). Byte order: the file's, except after predictor 3 (little-endian).
#[pyfunction]
#[pyo3(signature = (data, compression, predictor, width, rows, bytes_per_sample, samples = 1, big_endian = false))]
pub fn decode_tile<'py>(
    py: Python<'py>,
    data: &[u8],
    compression: u16,
    predictor: u16,
    width: usize,
    rows: usize,
    bytes_per_sample: usize,
    samples: usize,
    big_endian: bool,
) -> PyResult<Bound<'py, PyBytes>> {
    let input = data.to_vec();
    let out = py
        .detach(move || decode(&input, compression, predictor, width, rows, bytes_per_sample, samples, big_endian))
        .map_err(PyValueError::new_err)?;
    Ok(PyBytes::new(py, &out))
}
