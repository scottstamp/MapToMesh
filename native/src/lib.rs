//! Optional native speed-ups for MapToMesh.
//!
//! `write_zip` writes a ZIP archive (a 3MF is one) whose entries are lists of segments: raw
//! bytes, or numpy vertex / triangle arrays rendered as 3MF `<vertex>` / `<triangle>` XML lines.
//! The XML is formatted and deflate-compressed in parallel, chunk by chunk: every chunk is
//! compressed independently and ends on a byte boundary (a sync flush), so the chunks simply
//! concatenate into one valid deflate stream (the pigz approach); their CRC-32s are combined.
//!
//! The text is byte-for-byte what the Python writer in `maptomesh/export.py` produces (Rust's
//! `{:.N}` and Python's `{:.Nf}` both round the exact binary value, ties to even), so the two
//! are interchangeable and tests can compare their entries directly.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::Arc;

use crc32fast::Hasher;
use flate2::{Compress, Compression, FlushCompress};
use numpy::{PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyTuple};
use rayon::prelude::*;

/// Rows of vertices / triangles per parallel work item (about 1.5-2 MB of XML).
const ROWS_PER_CHUNK: usize = 32_768;
/// Raw bytes per parallel work item (a large binary entry, e.g. an STL, uses every core).
const RAW_CHUNK: usize = 4 << 20;
const U32_MAX: u64 = 0xFFFF_FFFF;

#[derive(Clone)]
enum Floats {
    F32(Arc<Vec<f32>>),
    F64(Arc<Vec<f64>>),
}

#[derive(Clone)]
enum Ints {
    I32(Arc<Vec<i32>>),
    I64(Arc<Vec<i64>>),
}

/// A unit of parallel work: rows [a, b) of an array, or bytes [a, b) of a raw buffer.
enum Work {
    Raw(Arc<Vec<u8>>, usize, usize),
    Vertices(Floats, usize, usize, usize), // array, rows [a, b), decimals
    Triangles(Ints, usize, usize),
}

fn fmt_vertices<T: Copy + Into<f64>>(v: &[T], a: usize, b: usize, d: usize, out: &mut Vec<u8>) {
    out.reserve((b - a) * 64);
    for r in a..b {
        let (x, y, z): (f64, f64, f64) = (v[3 * r].into(), v[3 * r + 1].into(), v[3 * r + 2].into());
        write!(out, "     <vertex x=\"{x:.d$}\" y=\"{y:.d$}\" z=\"{z:.d$}\"/>\n").expect("write to Vec");
    }
}

fn fmt_triangles<T: Copy + std::fmt::Display>(t: &[T], a: usize, b: usize, out: &mut Vec<u8>) {
    out.reserve((b - a) * 48);
    for r in a..b {
        let (p, q, s) = (t[3 * r], t[3 * r + 1], t[3 * r + 2]);
        write!(out, "     <triangle v1=\"{p}\" v2=\"{q}\" v3=\"{s}\"/>\n").expect("write to Vec");
    }
}

/// Raw deflate of `data` ending in a sync flush (byte-aligned, not final).
fn deflate_sync(data: &[u8], level: u32) -> Vec<u8> {
    let mut c = Compress::new(Compression::new(level), false);
    let mut out: Vec<u8> = Vec::with_capacity(data.len() / 3 + 1024);
    let mut pos = 0usize;
    loop {
        if out.len() == out.capacity() {
            out.reserve(out.capacity() / 2 + 4096);
        }
        let before = c.total_in();
        c.compress_vec(&data[pos..], &mut out, FlushCompress::Sync).expect("deflate");
        pos += (c.total_in() - before) as usize;
        // the flush is complete once all input is consumed and output space was left over
        if pos >= data.len() && out.len() < out.capacity() {
            return out;
        }
    }
}

/// The closing empty final block of a deflate stream.
fn deflate_end(level: u32) -> Vec<u8> {
    let mut c = Compress::new(Compression::new(level), false);
    let mut out = Vec::with_capacity(64);
    c.compress_vec(&[], &mut out, FlushCompress::Finish).expect("deflate");
    out
}

struct Compressed {
    chunks: Vec<Vec<u8>>,
    crc: u32,
    uncompressed: u64,
    compressed: u64,
}

fn compress_entry(work: Vec<Work>, level: u32) -> Compressed {
    let parts: Vec<(Vec<u8>, Hasher, u64)> = work
        .into_par_iter()
        .map(|w| {
            let mut text = Vec::new();
            let data: &[u8] = match &w {
                Work::Raw(b, a, e) => &b[*a..*e],
                Work::Vertices(Floats::F32(v), a, b, d) => { fmt_vertices(v, *a, *b, *d, &mut text); &text }
                Work::Vertices(Floats::F64(v), a, b, d) => { fmt_vertices(v, *a, *b, *d, &mut text); &text }
                Work::Triangles(Ints::I32(t), a, b) => { fmt_triangles(t, *a, *b, &mut text); &text }
                Work::Triangles(Ints::I64(t), a, b) => { fmt_triangles(t, *a, *b, &mut text); &text }
            };
            let mut h = Hasher::new();
            h.update(data);
            (deflate_sync(data, level), h, data.len() as u64)
        })
        .collect();
    let mut crc = Hasher::new();
    let mut uncompressed = 0u64;
    let mut chunks = Vec::with_capacity(parts.len() + 1);
    for (c, h, n) in parts {
        crc.combine(&h);
        uncompressed += n;
        chunks.push(c);
    }
    chunks.push(deflate_end(level));
    let compressed = chunks.iter().map(|c| c.len() as u64).sum();
    Compressed { chunks, crc: crc.finalize(), uncompressed, compressed }
}

struct CentralEntry {
    name: Vec<u8>,
    crc: u32,
    comp: u64,
    uncomp: u64,
    offset: u64,
}

struct ZipOut {
    w: BufWriter<File>,
    pos: u64,
    time: u16,
    date: u16,
    entries: Vec<CentralEntry>,
}

impl ZipOut {
    fn put(&mut self, b: &[u8]) -> std::io::Result<()> {
        self.w.write_all(b)?;
        self.pos += b.len() as u64;
        Ok(())
    }

    /// `force64`: zip64 sizes in the local header even when small (as Python's zipfile does for
    /// streamed entries), so readers and this code path see the same layout at every size.
    fn entry(&mut self, name: &str, c: &Compressed, force64: bool) -> std::io::Result<()> {
        let zip64 = force64 || c.uncompressed >= U32_MAX || c.compressed >= U32_MAX;
        let offset = self.pos;
        let mut h = Vec::with_capacity(64 + name.len());
        h.extend(0x0403_4b50u32.to_le_bytes());
        h.extend((if zip64 { 45u16 } else { 20 }).to_le_bytes());
        h.extend(0u16.to_le_bytes()); // flags
        h.extend(8u16.to_le_bytes()); // deflate
        h.extend(self.time.to_le_bytes());
        h.extend(self.date.to_le_bytes());
        h.extend(c.crc.to_le_bytes());
        let (cs, us) = if zip64 { (U32_MAX, U32_MAX) } else { (c.compressed, c.uncompressed) };
        h.extend((cs as u32).to_le_bytes());
        h.extend((us as u32).to_le_bytes());
        h.extend((name.len() as u16).to_le_bytes());
        h.extend((if zip64 { 20u16 } else { 0 }).to_le_bytes());
        h.extend(name.as_bytes());
        if zip64 {
            h.extend(1u16.to_le_bytes());
            h.extend(16u16.to_le_bytes());
            h.extend(c.uncompressed.to_le_bytes());
            h.extend(c.compressed.to_le_bytes());
        }
        self.put(&h)?;
        for chunk in &c.chunks {
            self.put(chunk)?;
        }
        self.entries.push(CentralEntry {
            name: name.as_bytes().to_vec(), crc: c.crc, comp: c.compressed, uncomp: c.uncompressed, offset,
        });
        Ok(())
    }

    fn finish(mut self) -> std::io::Result<u64> {
        let cd_start = self.pos;
        let entries = std::mem::take(&mut self.entries);
        for e in &entries {
            let mut extra = Vec::new();
            if e.uncomp >= U32_MAX { extra.extend(e.uncomp.to_le_bytes()); }
            if e.comp >= U32_MAX { extra.extend(e.comp.to_le_bytes()); }
            if e.offset >= U32_MAX { extra.extend(e.offset.to_le_bytes()); }
            let zip64 = !extra.is_empty();
            let mut h = Vec::with_capacity(64 + e.name.len());
            h.extend(0x0201_4b50u32.to_le_bytes());
            h.extend(45u16.to_le_bytes()); // made by: 4.5, MS-DOS
            h.extend((if zip64 { 45u16 } else { 20 }).to_le_bytes());
            h.extend(0u16.to_le_bytes());
            h.extend(8u16.to_le_bytes());
            h.extend(self.time.to_le_bytes());
            h.extend(self.date.to_le_bytes());
            h.extend(e.crc.to_le_bytes());
            h.extend((e.comp.min(U32_MAX) as u32).to_le_bytes());
            h.extend((e.uncomp.min(U32_MAX) as u32).to_le_bytes());
            h.extend((e.name.len() as u16).to_le_bytes());
            h.extend((if zip64 { extra.len() as u16 + 4 } else { 0 }).to_le_bytes());
            h.extend(0u16.to_le_bytes()); // comment
            h.extend(0u16.to_le_bytes()); // disk
            h.extend(0u16.to_le_bytes()); // internal attributes
            h.extend(0u32.to_le_bytes()); // external attributes
            h.extend((e.offset.min(U32_MAX) as u32).to_le_bytes());
            h.extend(&e.name);
            if zip64 {
                h.extend(1u16.to_le_bytes());
                h.extend((extra.len() as u16).to_le_bytes());
                h.extend(&extra);
            }
            self.put(&h)?;
        }
        let cd_size = self.pos - cd_start;
        let n = entries.len() as u64;
        if cd_start >= U32_MAX || cd_size >= U32_MAX || n >= 0xFFFF {
            let eocd64 = self.pos;
            let mut r = Vec::with_capacity(76);
            r.extend(0x0606_4b50u32.to_le_bytes());
            r.extend(44u64.to_le_bytes());
            r.extend(45u16.to_le_bytes());
            r.extend(45u16.to_le_bytes());
            r.extend(0u32.to_le_bytes());
            r.extend(0u32.to_le_bytes());
            r.extend(n.to_le_bytes());
            r.extend(n.to_le_bytes());
            r.extend(cd_size.to_le_bytes());
            r.extend(cd_start.to_le_bytes());
            r.extend(0x0706_4b50u32.to_le_bytes()); // locator
            r.extend(0u32.to_le_bytes());
            r.extend(eocd64.to_le_bytes());
            r.extend(1u32.to_le_bytes());
            self.put(&r)?;
        }
        let mut r = Vec::with_capacity(22);
        r.extend(0x0605_4b50u32.to_le_bytes());
        r.extend(0u16.to_le_bytes());
        r.extend(0u16.to_le_bytes());
        r.extend((n.min(0xFFFF) as u16).to_le_bytes());
        r.extend((n.min(0xFFFF) as u16).to_le_bytes());
        r.extend((cd_size.min(U32_MAX) as u32).to_le_bytes());
        r.extend((cd_start.min(U32_MAX) as u32).to_le_bytes());
        r.extend(0u16.to_le_bytes());
        self.put(&r)?;
        self.w.flush()?;
        Ok(self.pos)
    }
}

fn floats(seg: &Bound<'_, PyAny>) -> PyResult<(Floats, usize)> {
    if let Ok(a) = seg.extract::<PyReadonlyArray2<f32>>() {
        let rows = a.as_array().nrows();
        Ok((Floats::F32(Arc::new(a.as_slice()?.to_vec())), rows))
    } else {
        let a = seg.extract::<PyReadonlyArray2<f64>>()?;
        let rows = a.as_array().nrows();
        Ok((Floats::F64(Arc::new(a.as_slice()?.to_vec())), rows))
    }
}

fn ints(seg: &Bound<'_, PyAny>) -> PyResult<(Ints, usize)> {
    if let Ok(a) = seg.extract::<PyReadonlyArray2<i32>>() {
        let rows = a.as_array().nrows();
        Ok((Ints::I32(Arc::new(a.as_slice()?.to_vec())), rows))
    } else {
        let a = seg.extract::<PyReadonlyArray2<i64>>()?;
        let rows = a.as_array().nrows();
        Ok((Ints::I64(Arc::new(a.as_slice()?.to_vec())), rows))
    }
}

/// Consecutive raw bytes -> work items of at most RAW_CHUNK bytes (slices of one buffer).
fn flush_raw(pending: &mut Vec<u8>, work: &mut Vec<Work>) {
    if pending.is_empty() {
        return;
    }
    let buf = Arc::new(std::mem::take(pending));
    let n = buf.len();
    for a in (0..n).step_by(RAW_CHUNK) {
        work.push(Work::Raw(buf.clone(), a, (a + RAW_CHUNK).min(n)));
    }
}

/// Segments of one entry -> work items (consecutive raw bytes merged, everything split into chunks).
fn plan(segments: &[Bound<'_, PyAny>], decimals: usize) -> PyResult<Vec<Work>> {
    let mut work: Vec<Work> = Vec::new();
    let mut pending: Vec<u8> = Vec::new();
    for seg in segments {
        if let Ok(b) = seg.cast::<PyBytes>() {
            pending.extend_from_slice(b.as_bytes());
            continue;
        }
        flush_raw(&mut pending, &mut work);
        let t = seg.cast::<PyTuple>().map_err(|_| {
            PyValueError::new_err("a segment is bytes or a (\"vertices\" | \"triangles\", array) tuple")
        })?;
        let kind: String = t.get_item(0)?.extract()?;
        let arr = t.get_item(1)?;
        let ncols: usize = arr.getattr("shape")?.extract::<(usize, usize)>()?.1;
        if ncols != 3 {
            return Err(PyValueError::new_err(format!("{kind} array must have 3 columns, not {ncols}")));
        }
        match kind.as_str() {
            "vertices" => {
                let (data, rows) = floats(&arr)?;
                for a in (0..rows).step_by(ROWS_PER_CHUNK) {
                    work.push(Work::Vertices(data.clone(), a, (a + ROWS_PER_CHUNK).min(rows), decimals));
                }
            }
            "triangles" => {
                let (data, rows) = ints(&arr)?;
                for a in (0..rows).step_by(ROWS_PER_CHUNK) {
                    work.push(Work::Triangles(data.clone(), a, (a + ROWS_PER_CHUNK).min(rows)));
                }
            }
            _ => return Err(PyValueError::new_err(format!("unknown segment kind {kind:?}"))),
        }
    }
    flush_raw(&mut pending, &mut work);
    Ok(work)
}

/// Write a ZIP archive. `entries`: [(name, [segment, ...]), ...] where a segment is `bytes`,
/// `("vertices", float32|float64 (n, 3) array)` or `("triangles", int32|int64 (m, 3) array)`
/// (C-contiguous). `date_time`: (year, month, day, hour, minute, second) for every entry;
/// `decimals`: vertex coordinate decimals. Returns the archive's size in bytes. The GIL is
/// released while formatting and compressing.
#[pyfunction]
#[pyo3(signature = (path, entries, level = 3, date_time = (1980, 1, 1, 0, 0, 0), decimals = 3))]
fn write_zip(
    py: Python<'_>,
    path: std::path::PathBuf,
    entries: Vec<(String, Vec<Bound<'_, PyAny>>)>,
    level: u32,
    date_time: (u16, u16, u16, u16, u16, u16),
    decimals: usize,
) -> PyResult<u64> {
    if level > 9 || decimals > 9 {
        return Err(PyValueError::new_err("level and decimals must be 0..9"));
    }
    let plans: Vec<(String, Vec<Work>)> = entries
        .iter()
        .map(|(name, segs)| Ok((name.clone(), plan(segs, decimals)?)))
        .collect::<PyResult<_>>()?;
    let (y, mo, d, h, mi, s) = date_time;
    let time = (h << 11) | (mi << 5) | (s / 2);
    let date = (y.saturating_sub(1980) << 9) | (mo << 5) | d;
    py.detach(move || -> std::io::Result<u64> {
        let mut z = ZipOut { w: BufWriter::with_capacity(1 << 20, File::create(&path)?), pos: 0, time, date, entries: Vec::new() };
        for (name, work) in plans {
            let force64 = work.iter().any(|w| !matches!(w, Work::Raw(..))); // the mesh entry
            let c = compress_entry(work, level);
            z.entry(&name, &c, force64)?;
        }
        z.finish()
    })
    .map_err(|e| PyIOError::new_err(e.to_string()))
}

/// Simplify a triangle mesh (meshoptimizer) for display: at most `target_triangles` triangles,
/// fewer once `max_error` (a fraction of the mesh's extent) would be exceeded. Returns the new
/// (m, 3) uint32 triangle array; it indexes the same `vertices` (unused ones are left in place).
#[pyfunction]
#[pyo3(signature = (vertices, triangles, target_triangles, max_error = 0.01))]
fn simplify<'py>(
    py: Python<'py>,
    vertices: PyReadonlyArray2<'py, f32>,
    triangles: PyReadonlyArray2<'py, i32>,
    target_triangles: usize,
    max_error: f32,
) -> PyResult<Bound<'py, numpy::PyArray2<u32>>> {
    if vertices.as_array().ncols() != 3 || triangles.as_array().ncols() != 3 {
        return Err(PyValueError::new_err("vertices and triangles must be (n, 3) arrays"));
    }
    let v: Vec<f32> = vertices.as_slice()?.to_vec();
    let nv = v.len() / 3;
    let t: Vec<u32> = triangles
        .as_slice()?
        .iter()
        .map(|&i| if i >= 0 && (i as usize) < nv { Ok(i as u32) } else { Err(()) })
        .collect::<Result<_, _>>()
        .map_err(|_| PyValueError::new_err("triangle index out of range"))?;
    let out = py.detach(move || {
        let bytes: &[u8] = unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), v.len() * 4) };
        let adapter = meshopt::VertexDataAdapter::new(bytes, 12, 0).expect("vertex adapter");
        meshopt::simplify(&t, &adapter, target_triangles * 3, max_error, meshopt::SimplifyOptions::empty(), None)
    });
    let rows = out.len() / 3;
    numpy::PyArray1::from_vec(py, out).reshape([rows, 3])
}

/// `maptomesh.pointcloud._plane_fit_ground` on all cores: per vertex bin, the weighted
/// least-squares plane through the ground points of the bin and its 8 neighbours (weights
/// 1 / 0.3 / 0.1), evaluated at the vertex; the bin mean where the fit is ill-conditioned; NaN
/// where the bin has no ground points. Same sums in the same order as the NumPy version; the
/// 3x3 solve is Cramer's rule instead of LU (results agree to ~1e-9 m).
/// `gcnt` (n,) int32, `gsum` (n,) float64, `gmom` (7, n) float64: dx, dy, dx2, dxdy, dy2, dxz, dyz.
#[pyfunction]
fn plane_fit_ground<'py>(
    py: Python<'py>,
    gcnt: PyReadonlyArray1<'py, i32>,
    gsum: PyReadonlyArray1<'py, f64>,
    gmom: PyReadonlyArray2<'py, f64>,
    height: usize,
    width: usize,
    step: f64,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    let n = height * width;
    let (cnt, sum, mom) = (gcnt.as_slice()?, gsum.as_slice()?, gmom.as_slice()?);
    if cnt.len() != n || sum.len() != n || mom.len() != 7 * n {
        return Err(PyValueError::new_err("accumulator sizes don't match height x width"));
    }
    let mut out = vec![0f64; n];
    py.detach(|| {
        out.par_chunks_mut(width).enumerate().for_each(|(i, row)| {
            for (j, o) in row.iter_mut().enumerate() {
                let (mut sn, mut sx, mut sy, mut sz) = (0f64, 0f64, 0f64, 0f64);
                let (mut sxx, mut sxy, mut syy, mut sxz, mut syz) = (0f64, 0f64, 0f64, 0f64, 0f64);
                for di in -1i64..=1 {
                    for dj in -1i64..=1 {
                        let (ii, jj) = (i as i64 + di, j as i64 + dj);
                        // outside the grid: zero sums (the NumPy version pads with zeros)
                        let k = if ii < 0 || jj < 0 || ii >= height as i64 || jj >= width as i64 {
                            None
                        } else {
                            Some(ii as usize * width + jj as usize)
                        };
                        let g = |a: usize| k.map_or(0.0, |k| mom[a * n + k]);
                        let nn = k.map_or(0.0, |k| cnt[k] as f64);
                        let s_z = k.map_or(0.0, |k| sum[k]);
                        let (mx, my, mxx, mxy, myy, mxz, myz) = (g(0), g(1), g(2), g(3), g(4), g(5), g(6));
                        let w = [1.0, 0.3, 0.1][(di.abs() + dj.abs()) as usize];
                        let (ox, oy) = (dj as f64 * step, di as f64 * step);
                        sn += w * nn;
                        sx += w * (mx + nn * ox);
                        sy += w * (my + nn * oy);
                        sz += w * s_z;
                        sxx += w * (mxx + 2.0 * ox * mx + nn * ox * ox);
                        syy += w * (myy + 2.0 * oy * my + nn * oy * oy);
                        sxy += w * (mxy + ox * my + oy * mx + nn * ox * oy);
                        sxz += w * (mxz + ox * s_z);
                        syz += w * (myz + oy * s_z);
                    }
                }
                let k = i * width + j;
                let nk = cnt[k] as f64;
                let mean = if nk > 0.0 { sum[k] / nk.max(1.0) } else { f64::NAN };
                *o = mean;
                if nk > 0.0 && sn >= 3.0 {
                    let zc = sz / sn;
                    let ridge = 1e-3 * sn * step * step;
                    let a = [[sn, sx, sy], [sx, sxx + ridge, sxy], [sy, sxy, syy + ridge]];
                    let r = [0.0, sxz - zc * sx, syz - zc * sy];
                    let det3 = |m: [[f64; 3]; 3]| {
                        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
                    };
                    let d = det3(a);
                    if d.abs() > 1e-9 * sn * sn * sn * step.powi(4) {
                        let a0 = [[r[0], a[0][1], a[0][2]], [r[1], a[1][1], a[1][2]], [r[2], a[2][1], a[2][2]]];
                        *o = zc + det3(a0) / d;
                    }
                }
            }
        });
    });
    Ok(numpy::PyArray1::from_vec(py, out))
}

#[pymodule]
fn maptomesh_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(plane_fit_ground, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(write_zip, m)?)?;
    m.add_function(wrap_pyfunction!(simplify, m)?)?;
    Ok(())
}
