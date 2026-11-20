//! Mapbox Vector Tile (MVT 2.x) decoding for `maptomesh/osm.py` (`decode_tile`: OpenMapTiles
//! water / roads / buildings) and `maptomesh/mvt.py` (`decode`: Overture building and Mapterhorn
//! coverage polygons). https://github.com/mapbox/vector-tile-spec/tree/master/2.1
//!
//! A hand-written protobuf reader (no dependency). Both functions return exactly what the
//! pure-Python decoders in those modules return (the fallback when the extension is missing): the
//! same layer / feature order, property dicts with the same Python types and the same geometry.
//! Malformed or truncated input raises `ValueError`; nothing indexes out of bounds or panics.
//! Gzip is left to the Python callers (`gzip.decompress`, C speed).
//!
//! The tile is parsed with the GIL released, then the Python objects are built.

use std::collections::HashSet;

use numpy::{PyArray1, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString, PyTuple};

type R<T> = Result<T, String>;

// ---------------------------------------------------------------- protobuf

fn varint(b: &[u8], p: &mut usize) -> R<u64> {
    let mut out = 0u64;
    let mut shift = 0u32;
    loop {
        let c = *b.get(*p).ok_or("truncated varint")?;
        *p += 1;
        if shift == 63 && c > 1 {
            return Err("varint overflows 64 bits".into());
        }
        out |= u64::from(c & 0x7F) << shift;
        if c & 0x80 == 0 {
            return Ok(out);
        }
        shift += 7;
    }
}

#[derive(Clone, Copy)]
enum Wire<'a> {
    Varint(u64),
    Fixed64(&'a [u8]),
    Bytes(&'a [u8]),
    Fixed32(&'a [u8]),
}

impl<'a> Wire<'a> {
    fn bytes(self, what: &str) -> R<&'a [u8]> {
        match self {
            Wire::Bytes(b) => Ok(b),
            _ => Err(format!("{what}: expected a length-delimited field")),
        }
    }
    fn uint(self, what: &str) -> R<u64> {
        match self {
            Wire::Varint(v) => Ok(v),
            _ => Err(format!("{what}: expected a varint field")),
        }
    }
}

/// (field number, value) of a protobuf message, in order.
struct Fields<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Fields<'a> {
    fn new(b: &'a [u8]) -> Self {
        Fields { b, p: 0 }
    }

    fn take(&mut self, n: u64) -> R<&'a [u8]> {
        let end = usize::try_from(n)
            .ok()
            .and_then(|n| self.p.checked_add(n))
            .filter(|&e| e <= self.b.len())
            .ok_or("truncated protobuf field")?;
        let s = &self.b[self.p..end];
        self.p = end;
        Ok(s)
    }

    fn next(&mut self) -> R<Option<(u64, Wire<'a>)>> {
        if self.p >= self.b.len() {
            return Ok(None);
        }
        let key = varint(self.b, &mut self.p)?;
        let v = match key & 7 {
            0 => Wire::Varint(varint(self.b, &mut self.p)?),
            1 => Wire::Fixed64(self.take(8)?),
            2 => {
                let n = varint(self.b, &mut self.p)?;
                Wire::Bytes(self.take(n)?)
            }
            5 => Wire::Fixed32(self.take(4)?),
            wt => return Err(format!("unsupported protobuf wire type {wt}")),
        };
        Ok(Some((key >> 3, v)))
    }
}

fn packed(b: &[u8], out: &mut Vec<u64>) -> R<()> {
    out.clear();
    let mut p = 0;
    while p < b.len() {
        out.push(varint(b, &mut p)?);
    }
    Ok(())
}

#[inline]
fn zigzag(v: u64) -> i64 {
    ((v >> 1) as i64) ^ -((v & 1) as i64)
}

// ---------------------------------------------------------------- layers

/// A tag value (`Layer.values`); `None` when the message holds no known field.
enum Value {
    Str(String),
    Float(f64),
    UInt(u64),
    SInt(i64),
    Bool(bool),
    None,
}

fn value(b: &[u8]) -> R<Value> {
    let mut it = Fields::new(b);
    while let Some((f, v)) = it.next()? {
        return Ok(match (f, v) {
            (1, Wire::Bytes(s)) => Value::Str(String::from_utf8_lossy(s).into_owned()),
            (2, Wire::Fixed32(s)) => Value::Float(f64::from(f32::from_le_bytes([s[0], s[1], s[2], s[3]]))),
            (3, Wire::Fixed64(s)) => Value::Float(f64::from_le_bytes(s.try_into().map_err(|_| "bad double")?)),
            (4 | 5, Wire::Varint(x)) => Value::UInt(x),
            (6, Wire::Varint(x)) => Value::SInt(zigzag(x)),
            (7, Wire::Varint(x)) => Value::Bool(x != 0),
            (1..=7, _) => return Err(format!("tag value field {f}: unexpected wire type")),
            _ => continue,
        });
    }
    Ok(Value::None)
}

fn text(b: &[u8], strict: bool, what: &str) -> R<String> {
    if strict {
        std::str::from_utf8(b).map(str::to_owned).map_err(|e| format!("{what}: invalid UTF-8 ({e})"))
    } else {
        Ok(String::from_utf8_lossy(b).into_owned())
    }
}

/// One layer's metadata and raw features.
struct Layer<'a> {
    name: Option<String>,
    extent: u64,
    keys: Vec<String>,
    values: Vec<Value>,
    feats: Vec<&'a [u8]>,
}

/// Parse a layer's fields in order. `strict`: name / keys must be valid UTF-8 (osm.py decodes
/// them strictly, mvt.py with "replace"). `wanted`: stop at a name not in it, as osm.py does.
fn layer<'a>(b: &'a [u8], strict: bool, wanted: Option<&HashSet<String>>) -> R<Layer<'a>> {
    let mut l = Layer { name: None, extent: 4096, keys: vec![], values: vec![], feats: vec![] };
    let mut it = Fields::new(b);
    while let Some((f, v)) = it.next()? {
        match f {
            1 => {
                let name = text(v.bytes("layer name")?, strict, "layer name")?;
                let stop = wanted.is_some_and(|w| !w.contains(&name));
                l.name = Some(name);
                if stop {
                    break;
                }
            }
            2 => l.feats.push(v.bytes("feature")?),
            3 => l.keys.push(text(v.bytes("key")?, strict, "key")?),
            4 => l.values.push(value(v.bytes("value")?)?),
            5 => l.extent = v.uint("extent")?,
            _ => {}
        }
    }
    Ok(l)
}

/// A feature's tags, type and geometry commands (later fields of a kind win, as in Python).
fn feature(b: &[u8], tags: &mut Vec<u64>, geom: &mut Vec<u64>) -> R<u64> {
    tags.clear();
    geom.clear();
    let mut gtype = 0;
    let mut it = Fields::new(b);
    while let Some((f, v)) = it.next()? {
        match (f, v) {
            (2, Wire::Bytes(s)) => packed(s, tags)?,
            (2, Wire::Varint(x)) => {
                tags.clear();
                tags.push(x);
            }
            (3, v) => gtype = v.uint("geometry type")?,
            (4, v) => packed(v.bytes("geometry")?, geom)?,
            (2, _) => return Err("tags: unexpected wire type".into()),
            _ => {}
        }
    }
    Ok(gtype)
}

/// (key index, value index) pairs, checked against the layer's tables.
fn props(tags: &[u64], nk: usize, nv: usize) -> R<Vec<(usize, usize)>> {
    tags.chunks_exact(2)
        .map(|kv| {
            let (k, v) = (kv[0] as usize, kv[1] as usize);
            if kv[0] >= nk as u64 || k >= nk {
                Err(format!("tag key index {} out of range", kv[0]))
            } else if kv[1] >= nv as u64 || v >= nv {
                Err(format!("tag value index {} out of range", kv[1]))
            } else {
                Ok((k, v))
            }
        })
        .collect()
}

fn params(cmds: &[u64], i: &mut usize, x: &mut i64, y: &mut i64) -> R<()> {
    let (Some(&dx), Some(&dy)) = (cmds.get(*i), cmds.get(*i + 1)) else {
        return Err("truncated geometry".into());
    };
    *i += 2;
    *x = x.checked_add(zigzag(dx)).ok_or("geometry coordinate overflow")?;
    *y = y.checked_add(zigzag(dy)).ok_or("geometry coordinate overflow")?;
    Ok(())
}

fn to_py_value<'py>(py: Python<'py>, v: &Value) -> PyResult<Bound<'py, PyAny>> {
    Ok(match v {
        Value::Str(s) => PyString::new(py, s).into_any(),
        Value::Float(f) => (*f).into_pyobject(py)?.into_any(),
        Value::UInt(u) => (*u).into_pyobject(py)?.into_any(),
        Value::SInt(i) => (*i).into_pyobject(py)?.into_any(),
        Value::Bool(b) => (*b).into_pyobject(py)?.to_owned().into_any(),
        Value::None => py.None().into_bound(py),
    })
}

/// The layer's keys and values as Python objects (shared by every feature's props, as in Python).
type Tables<'py> = (Vec<Bound<'py, PyString>>, Vec<Bound<'py, PyAny>>);

fn tables<'py>(py: Python<'py>, keys: &[String], values: &[Value]) -> PyResult<Tables<'py>> {
    let k = keys.iter().map(|s| PyString::new(py, s)).collect();
    let v = values.iter().map(|v| to_py_value(py, v)).collect::<PyResult<_>>()?;
    Ok((k, v))
}

fn props_dict<'py>(py: Python<'py>, pairs: &[(usize, usize)], k: &[Bound<'py, PyString>], v: &[Bound<'py, PyAny>]) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    for &(a, b) in pairs {
        d.set_item(&k[a], &v[b])?;
    }
    Ok(d)
}

// ---------------------------------------------------------------- osm.decode_tile

struct OsmFeat {
    poly: bool,
    props: Vec<(usize, usize)>,
    nparts: usize,
}

struct OsmLayer {
    name: String,
    extent: u64,
    keys: Vec<String>,
    values: Vec<Value>,
    feats: Vec<OsmFeat>,
    xy: Vec<f64>,
    ends: Vec<usize>,
}

/// `osm._geometry`: MoveTo starts a part, LineTo extends it, ClosePath repeats its first point.
/// Appends the parts' points to `xy` and their end offsets (in points) to `ends`.
fn osm_geometry(cmds: &[u64], xy: &mut Vec<f64>, ends: &mut Vec<usize>) -> R<usize> {
    let n0 = ends.len();
    let mut start = xy.len();
    let (mut x, mut y, mut i) = (0i64, 0i64, 0usize);
    while i < cmds.len() {
        let c = cmds[i];
        i += 1;
        let (cid, cnt) = (c & 7, c >> 3);
        match cid {
            1 => {
                for _ in 0..cnt {
                    if xy.len() > start {
                        ends.push(xy.len() / 2);
                    }
                    params(cmds, &mut i, &mut x, &mut y)?;
                    start = xy.len();
                    xy.extend([x as f64, y as f64]);
                }
            }
            2 => {
                for _ in 0..cnt {
                    params(cmds, &mut i, &mut x, &mut y)?;
                    xy.extend([x as f64, y as f64]);
                }
            }
            7 if xy.len() > start => xy.extend([xy[start], xy[start + 1]]),
            _ => {}
        }
    }
    if xy.len() > start {
        ends.push(xy.len() / 2);
    }
    Ok(ends.len() - n0)
}

fn parse_osm(data: &[u8], wanted: &HashSet<String>) -> R<Vec<OsmLayer>> {
    let mut out = vec![];
    let (mut tags, mut geom) = (vec![], vec![]);
    let mut it = Fields::new(data);
    while let Some((f, v)) = it.next()? {
        let (3, Wire::Bytes(lb)) = (f, v) else { continue };
        let l = layer(lb, true, Some(wanted))?;
        let Some(name) = l.name.filter(|n| wanted.contains(n)) else { continue };
        if l.extent == 0 {
            return Err(format!("layer {name}: zero extent"));
        }
        let mut ol = OsmLayer { name, extent: l.extent, keys: l.keys, values: l.values, feats: vec![], xy: vec![], ends: vec![] };
        for fb in l.feats {
            let gtype = feature(fb, &mut tags, &mut geom)?;
            if gtype != 2 && gtype != 3 {
                continue;
            }
            let props = props(&tags, ol.keys.len(), ol.values.len())?;
            let nparts = osm_geometry(&geom, &mut ol.xy, &mut ol.ends)?;
            ol.feats.push(OsmFeat { poly: gtype == 3, props, nparts });
        }
        out.push(ol);
    }
    Ok(out)
}

type OsmOut<'py> = Vec<(Bound<'py, PyString>, u64, Bound<'py, numpy::PyArray2<f64>>, Vec<usize>, Bound<'py, PyList>)>;

/// The wanted layers of an (uncompressed) OpenMapTiles tile, for `osm.decode_tile`:
/// [(name, extent, xy, ends, features)]: `xy` (n, 2) float64 tile coordinates of every part's
/// points, `ends` the parts' end offsets into it, `features` the line / polygon features as
/// `{"layer", "type", "props", "parts"}` dicts whose "parts" holds their number of parts (the
/// caller projects `xy` once per layer and swaps the counts for the part arrays).
#[pyfunction]
pub fn mvt_decode_osm<'py>(py: Python<'py>, data: &[u8], layers: HashSet<String>) -> PyResult<OsmOut<'py>> {
    let parsed = py.detach(|| parse_osm(data, &layers)).map_err(PyValueError::new_err)?;
    let (k_layer, k_type, k_props, k_parts) = (
        pyo3::intern!(py, "layer"),
        pyo3::intern!(py, "type"),
        pyo3::intern!(py, "props"),
        pyo3::intern!(py, "parts"),
    );
    let (polygon, line) = (pyo3::intern!(py, "polygon"), pyo3::intern!(py, "line"));
    let mut out = Vec::with_capacity(parsed.len());
    for l in parsed {
        let name = PyString::new(py, &l.name);
        let (k, v) = tables(py, &l.keys, &l.values)?;
        let feats = PyList::empty(py);
        for f in &l.feats {
            let d = PyDict::new(py);
            d.set_item(k_layer, &name)?;
            d.set_item(k_type, if f.poly { polygon } else { line })?;
            d.set_item(k_props, props_dict(py, &f.props, &k, &v)?)?;
            d.set_item(k_parts, f.nparts)?;
            feats.append(d)?;
        }
        let n = l.xy.len() / 2;
        let xy = PyArray1::from_vec(py, l.xy).reshape([n, 2])?;
        out.push((name, l.extent, xy, l.ends, feats));
    }
    Ok(out)
}

// ---------------------------------------------------------------- mvt.decode

struct MvtFeat {
    props: Vec<(usize, usize)>,
    gtype: u64,
    rings: Vec<Vec<(i64, i64)>>,
}

struct MvtLayer {
    name: String,
    extent: u64,
    keys: Vec<String>,
    values: Vec<Value>,
    feats: Vec<MvtFeat>,
}

/// `mvt._rings`: rings of at least 3 points, ended by ClosePath or the next MoveTo; an unclosed
/// last ring is dropped. Every command but ClosePath consumes its parameters.
fn mvt_rings(cmds: &[u64]) -> R<Vec<Vec<(i64, i64)>>> {
    let (mut rings, mut cur) = (vec![], vec![]);
    let (mut x, mut y, mut i) = (0i64, 0i64, 0usize);
    while i < cmds.len() {
        let c = cmds[i];
        i += 1;
        let (cid, cnt) = (c & 7, c >> 3);
        if cid == 7 {
            if cur.len() >= 3 {
                rings.push(std::mem::take(&mut cur));
            } else {
                cur.clear();
            }
            continue;
        }
        for _ in 0..cnt {
            params(cmds, &mut i, &mut x, &mut y)?;
            if cid == 1 {
                if cur.len() >= 3 {
                    rings.push(std::mem::take(&mut cur));
                } else {
                    cur.clear();
                }
            }
            cur.push((x, y));
        }
    }
    Ok(rings)
}

fn parse_mvt(data: &[u8]) -> R<Vec<MvtLayer>> {
    let mut out = vec![];
    let (mut tags, mut geom) = (vec![], vec![]);
    let mut it = Fields::new(data);
    while let Some((f, v)) = it.next()? {
        if f != 3 {
            continue;
        }
        let l = layer(v.bytes("layer")?, false, None)?;
        let mut ml = MvtLayer { name: l.name.unwrap_or_default(), extent: l.extent, keys: l.keys, values: l.values, feats: vec![] };
        for fb in l.feats {
            let gtype = feature(fb, &mut tags, &mut geom)?;
            let props = props(&tags, ml.keys.len(), ml.values.len())?;
            let rings = if gtype == 3 { mvt_rings(&geom)? } else { vec![] };
            ml.feats.push(MvtFeat { props, gtype, rings });
        }
        out.push(ml);
    }
    Ok(out)
}

/// `mvt.decode` of an (uncompressed) tile: {layer name: {"extent": int, "features":
/// [{"props": {...}, "type": int, "rings": [[(x, y), ...], ...]}]}} (rings only for polygons).
#[pyfunction]
pub fn mvt_decode<'py>(py: Python<'py>, data: &[u8]) -> PyResult<Bound<'py, PyDict>> {
    let parsed = py.detach(|| parse_mvt(data)).map_err(PyValueError::new_err)?;
    let (k_extent, k_features, k_props, k_type, k_rings) = (
        pyo3::intern!(py, "extent"),
        pyo3::intern!(py, "features"),
        pyo3::intern!(py, "props"),
        pyo3::intern!(py, "type"),
        pyo3::intern!(py, "rings"),
    );
    let out = PyDict::new(py);
    for l in parsed {
        let (k, v) = tables(py, &l.keys, &l.values)?;
        let feats = PyList::empty(py);
        for f in &l.feats {
            let rings = PyList::empty(py);
            for r in &f.rings {
                let ring = PyList::empty(py);
                for &(x, y) in r {
                    ring.append(PyTuple::new(py, [x, y])?)?;
                }
                rings.append(ring)?;
            }
            let d = PyDict::new(py);
            d.set_item(k_props, props_dict(py, &f.props, &k, &v)?)?;
            d.set_item(k_type, f.gtype)?;
            d.set_item(k_rings, rings)?;
            feats.append(d)?;
        }
        let ld = PyDict::new(py);
        ld.set_item(k_extent, l.extent)?;
        ld.set_item(k_features, feats)?;
        out.set_item(l.name, ld)?;
    }
    Ok(out)
}
