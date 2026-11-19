//! Optional native speed-ups for MapToMesh.
//!
//! `write_zip` writes a ZIP archive (a 3MF is one) whose entries are lists of segments: raw
//! bytes, or numpy vertex / triangle arrays rendered as 3MF `<vertex>` / `<triangle>` XML lines.
//! The XML is formatted and deflate-compressed in parallel, chunk by chunk: every chunk is
//! compressed independently and ends on a byte boundary (a sync flush), so the chunks simply
//! concatenate into one valid deflate stream (the pigz approach); their CRC-32s are combined.
//!
//! The text is byte-for-byte what the Python writer in `maptomesh/export.py` produces (Rust's
//! `{:.4}` and Python's `{:.4f}` both round the exact binary value, ties to even), so the two
//! are interchangeable and tests can compare their entries directly.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::Arc;

use crc32fast::Hasher;
use flate2::{Compress, Compression, FlushCompress};
use numpy::PyReadonlyArray2;
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyTuple};
use rayon::prelude::*;

/// Rows of vertices / triangles per parallel work item (about 1.5-2 MB of XML).
const ROWS_PER_CHUNK: usize = 32_768;
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

/// A unit of parallel work: rows [a, b) of an array, or raw bytes.
enum Work {
    Raw(Vec<u8>),
    Vertices(Floats, usize, usize),
    Triangles(Ints, usize, usize),
}

fn fmt_vertices<T: Copy + Into<f64>>(v: &[T], a: usize, b: usize, out: &mut Vec<u8>) {
    out.reserve((b - a) * 64);
    for r in a..b {
        let (x, y, z): (f64, f64, f64) = (v[3 * r].into(), v[3 * r + 1].into(), v[3 * r + 2].into());
        write!(out, "     <vertex x=\"{x:.4}\" y=\"{y:.4}\" z=\"{z:.4}\"/>\n").expect("write to Vec");
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
                Work::Raw(b) => b,
                Work::Vertices(Floats::F32(v), a, b) => { fmt_vertices(v, *a, *b, &mut text); &text }
                Work::Vertices(Floats::F64(v), a, b) => { fmt_vertices(v, *a, *b, &mut text); &text }
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

/// Segments of one entry -> work items (consecutive raw bytes merged, arrays split into chunks).
fn plan(segments: &[Bound<'_, PyAny>]) -> PyResult<Vec<Work>> {
    let mut work: Vec<Work> = Vec::new();
    for seg in segments {
        if let Ok(b) = seg.cast::<PyBytes>() {
            if let Some(Work::Raw(prev)) = work.last_mut() {
                prev.extend_from_slice(b.as_bytes());
            } else {
                work.push(Work::Raw(b.as_bytes().to_vec()));
            }
            continue;
        }
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
                    work.push(Work::Vertices(data.clone(), a, (a + ROWS_PER_CHUNK).min(rows)));
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
    Ok(work)
}

/// Write a ZIP archive. `entries`: [(name, [segment, ...]), ...] where a segment is `bytes`,
/// `("vertices", float32|float64 (n, 3) array)` or `("triangles", int32|int64 (m, 3) array)`
/// (C-contiguous). `date_time`: (year, month, day, hour, minute, second) for every entry.
/// Returns the archive's size in bytes. The GIL is released while formatting and compressing.
#[pyfunction]
#[pyo3(signature = (path, entries, level = 3, date_time = (1980, 1, 1, 0, 0, 0)))]
fn write_zip(
    py: Python<'_>,
    path: std::path::PathBuf,
    entries: Vec<(String, Vec<Bound<'_, PyAny>>)>,
    level: u32,
    date_time: (u16, u16, u16, u16, u16, u16),
) -> PyResult<u64> {
    if level > 9 {
        return Err(PyValueError::new_err("level must be 0..9"));
    }
    let plans: Vec<(String, Vec<Work>)> = entries
        .iter()
        .map(|(name, segs)| Ok((name.clone(), plan(segs)?)))
        .collect::<PyResult<_>>()?;
    let (y, mo, d, h, mi, s) = date_time;
    let time = (h << 11) | (mi << 5) | (s / 2);
    let date = (y.saturating_sub(1980) << 9) | (mo << 5) | d;
    py.detach(move || -> std::io::Result<u64> {
        let mut z = ZipOut { w: BufWriter::with_capacity(1 << 20, File::create(&path)?), pos: 0, time, date, entries: Vec::new() };
        for (name, work) in plans {
            let force64 = work.iter().any(|w| !matches!(w, Work::Raw(_)));   // the mesh entry
            let c = compress_entry(work, level);
            z.entry(&name, &c, force64)?;
        }
        z.finish()
    })
    .map_err(|e| PyIOError::new_err(e.to_string()))
}

#[pymodule]
fn maptomesh_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(write_zip, m)?)?;
    Ok(())
}
