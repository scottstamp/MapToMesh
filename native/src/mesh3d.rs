//! 3D building meshes (`maptomesh/buildings/plateau.py`): Draco-compressed glTF geometry
//! (KHR_draco_mesh_compression, which every PLATEAU 3D Tiles tileset requires) and rasterising
//! triangles onto the model's vertex grid. Pure Rust (draco-core), so it builds for WebAssembly
//! (Pyodide) as well as natively.

use draco_core::{DecoderBuffer, Mesh, MeshDecoder};
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

type Decoded<'py> = (Bound<'py, numpy::PyArray2<u32>>, Vec<Bound<'py, numpy::PyArray2<f32>>>);

/// Decode a Draco mesh: (faces (m, 3) uint32 point indices, one (n_points, components) float32
/// array per requested attribute unique id - the ids a glTF primitive's
/// `KHR_draco_mesh_compression.attributes` maps its attribute names to). An id the mesh doesn't
/// have gives an (n_points, 0) array.
#[pyfunction]
pub fn decode_draco<'py>(py: Python<'py>, data: &[u8], attributes: Vec<u32>) -> PyResult<Decoded<'py>> {
    let input = data.to_vec();
    let (faces, attrs, np) = py
        .detach(move || -> Result<(Vec<u32>, Vec<(Vec<f32>, usize)>, usize), String> {
            let mut buffer = DecoderBuffer::new(&input);
            let mut mesh = Mesh::new();
            MeshDecoder::new().decode(&mut buffer, &mut mesh).map_err(|e| format!("Draco: {e:?}"))?;
            let np = mesh.num_points();
            let faces: Vec<u32> = mesh.faces().iter().flat_map(|f| [f[0].0, f[1].0, f[2].0]).collect();
            let attrs = attributes
                .iter()
                .map(|&uid| match mesh.attribute_by_unique_id(uid) {
                    Some(a) => {
                        let n = a.num_components() as usize;
                        (a.read_f32s(np, n), n)
                    }
                    None => (Vec::new(), 0),
                })
                .collect();
            Ok((faces, attrs, np))
        })
        .map_err(PyValueError::new_err)?;
    let nf = faces.len() / 3;
    let faces = PyArray1::from_vec(py, faces).reshape([nf, 3])?;
    let mut out = Vec::with_capacity(attrs.len());
    for (v, n) in attrs {
        let rows = if n == 0 { np } else { v.len() / n };
        out.push(PyArray1::from_vec(py, v).reshape([rows, n])?);
    }
    Ok((faces, out))
}

/// Rasterise triangles onto a vertex grid, keeping the highest value per grid vertex.
///
/// `xy` (n, 2): (column, row) grid coordinates of the mesh points, `z` (n,) their values,
/// `tris` (m, 3) point indices, `tri_id` (m,) an id per triangle (e.g. its building). Every grid
/// vertex inside (or on the edge of) a triangle takes the triangle's value there when it is above
/// `out_z`'s (NaN counts as lowest); `out_id` records the id of the triangle that set it.
/// Triangles seen edge-on (walls) cover no area and are skipped.
#[pyfunction]
pub fn raster_max<'py>(
    py: Python<'py>,
    xy: PyReadonlyArray2<'py, f64>,
    z: PyReadonlyArray1<'py, f32>,
    tris: PyReadonlyArray2<'py, u32>,
    tri_id: PyReadonlyArray1<'py, i32>,
    mut out_z: PyReadwriteArray2<'py, f32>,
    mut out_id: PyReadwriteArray2<'py, i32>,
) -> PyResult<()> {
    let (h, w) = {
        let s = out_z.as_array().shape().to_vec();
        (s[0], s[1])
    };
    if out_id.as_array().shape() != [h, w] {
        return Err(PyValueError::new_err("out_z and out_id differ in shape"));
    }
    let xy = xy.as_slice()?;
    let z = z.as_slice()?;
    let tris = tris.as_slice()?;
    let tri_id = tri_id.as_slice()?;
    if xy.len() != 2 * z.len() || tri_id.len() * 3 != tris.len() {
        return Err(PyValueError::new_err("xy, z, tris and tri_id don't match"));
    }
    let oz = out_z.as_slice_mut()?;
    let oi = out_id.as_slice_mut()?;
    let np = z.len();
    py.detach(|| {
        for (t, &id) in tris.chunks_exact(3).zip(tri_id) {
            let (a, b, c) = (t[0] as usize, t[1] as usize, t[2] as usize);
            if a >= np || b >= np || c >= np {
                continue;
            }
            let (x0, y0, x1, y1, x2, y2) = (xy[2 * a], xy[2 * a + 1], xy[2 * b], xy[2 * b + 1], xy[2 * c], xy[2 * c + 1]);
            let den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2);
            if !den.is_finite() || den.abs() < 1e-9 {
                continue;
            }
            let cmin = x0.min(x1).min(x2).ceil().max(0.0);
            let cmax = x0.max(x1).max(x2).floor().min(w as f64 - 1.0);
            let rmin = y0.min(y1).min(y2).ceil().max(0.0);
            let rmax = y0.max(y1).max(y2).floor().min(h as f64 - 1.0);
            if cmin > cmax || rmin > rmax {
                continue;
            }
            let (za, zb, zc) = (z[a] as f64, z[b] as f64, z[c] as f64);
            let eps = 1e-7;
            for r in rmin as usize..=rmax as usize {
                let py_ = r as f64;
                for col in cmin as usize..=cmax as usize {
                    let px = col as f64;
                    let l0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py_ - y2)) / den;
                    let l1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py_ - y2)) / den;
                    let l2 = 1.0 - l0 - l1;
                    if l0 < -eps || l1 < -eps || l2 < -eps {
                        continue;
                    }
                    let v = (l0 * za + l1 * zb + l2 * zc) as f32;
                    let k = r * w + col;
                    if !(oz[k] >= v) {
                        oz[k] = v;
                        oi[k] = id;
                    }
                }
            }
        }
    });
    Ok(())
}
