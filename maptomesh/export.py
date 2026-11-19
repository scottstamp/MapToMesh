"""3MF / STL / preview writers."""

from __future__ import annotations

import io
import json
import struct
import time
import zipfile
from datetime import date
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

import numpy as np

from . import native
from .model import Part

NS_CORE = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"

CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
 <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
 <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
 <Default Extension="config" ContentType="text/xml"/>
</Types>
"""

RELS = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
</Relationships>
"""

CHUNK = 200_000
NATIVE_LEVEL = 6   # deflate level of the native writer (the Python one uses zlib's default, 6)
# vertex coordinates in the 3MF: 1 um, far below any printer's resolution (a 6.9 M-triangle model
# showed no degenerate, flipped or merged triangles from the rounding); ~2% smaller than 0.1 um
DECIMALS = 3

# Optional Rust extension (native/): formats and compresses the model XML on all cores. Without it
# (not built, or MAPTOMESH_NATIVE=0) the pure-Python writer below produces the same content.
_native = native.ext


def native_enabled() -> bool:
    return native.enabled()


# A zip entry is a list of segments: bytes, ("vertices", (n, 3) float array) or
# ("triangles", (m, 3) int array); the arrays become 3MF <vertex> / <triangle> lines.
Segment = bytes | tuple[str, np.ndarray]


def _segment_bytes(seg: Segment):
    if isinstance(seg, bytes):
        yield seg
        return
    kind, a = seg
    for s in range(0, len(a), CHUNK):
        rows = a[s:s + CHUNK].tolist()
        if kind == "vertices":
            yield "".join(f'     <vertex x="{x:.{DECIMALS}f}" y="{y:.{DECIMALS}f}" z="{z:.{DECIMALS}f}"/>\n'
                          for x, y, z in rows).encode()
        else:
            yield "".join(f'     <triangle v1="{p}" v2="{q}" v3="{r}"/>\n' for p, q, r in rows).encode()


def _write_zip_py(path: Path, entries: list[tuple[str, list[Segment]]]) -> None:
    stamp = time.localtime()[:6]
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=3) as zf:
        for name, segs in entries:
            if all(isinstance(s, bytes) for s in segs):
                zf.writestr(zipfile.ZipInfo(name, date_time=stamp), b"".join(segs), zipfile.ZIP_DEFLATED, 3)
                continue
            zi = zipfile.ZipInfo(name, date_time=stamp)
            zi.compress_type = zipfile.ZIP_DEFLATED
            with zf.open(zi, "w", force_zip64=True) as f:
                for seg in segs:
                    for b in _segment_bytes(seg):
                        f.write(b)


def write_zip(path: Path, entries: list[tuple[str, list[Segment]]]) -> Path:
    """Write the entries (see ``Segment``) with the native writer if available, else in Python."""
    if native_enabled():
        _native.write_zip(str(path), [(n, [s if isinstance(s, bytes) else (s[0], np.ascontiguousarray(s[1]))
                                          for s in segs]) for n, segs in entries],
                          NATIVE_LEVEL, time.localtime()[:6], DECIMALS)
    else:
        _write_zip_py(path, entries)
    return path


def _3mf_entries(parts: list[Part], filaments: list[dict], title: str, layout: str, description: str,
                 thumbnail: bytes | None) -> list[tuple[str, list[Segment]]]:
    slot = {f["color"]: f["slot"] for f in filaments}
    colors = [f["color"] for f in filaments]
    tiles: dict[str, list[int]] = {}
    for i, p in enumerate(parts):
        tiles.setdefault(p.tile, []).append(i)
    entries: list[tuple[str, list[Segment]]] = []
    if thumbnail:
        entries.append(("[Content_Types].xml", [CONTENT_TYPES.replace(
            "</Types>", ' <Default Extension="png" ContentType="image/png"/>\n</Types>').encode()]))
        entries.append(("_rels/.rels", [RELS.replace(
            "</Relationships>",
            ' <Relationship Target="/Metadata/thumbnail.png" Id="rel1" '
            'Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/thumbnail"/>\n'
            "</Relationships>").encode()]))
        entries.append(("Metadata/thumbnail.png", [thumbnail]))
    else:
        entries.append(("[Content_Types].xml", [CONTENT_TYPES.encode()]))
        entries.append(("_rels/.rels", [RELS.encode()]))

    part_obj_id: dict[int, int] = {}
    assembly_ids: dict[str, int] = {}
    m: list[Segment] = []
    w = lambda s: m.append(s.encode())  # noqa: E731
    w(f'<?xml version="1.0" encoding="UTF-8"?>\n<model unit="millimeter" xml:lang="en-US" xmlns="{NS_CORE}">\n')
    meta = {"Title": title, "Application": "MapToMesh", "CreationDate": date.today().isoformat(),
            "Description": description or "Topographic model from NRCan CanElevation LiDAR",
            "Copyright": "Contains information licensed under the Open Government Licence - Canada; "
                         "map data (c) OpenStreetMap contributors (ODbL)"}
    for k, v in meta.items():
        w(f' <metadata name="{k}">{escape(v)}</metadata>\n')
    w(' <resources>\n  <basematerials id="1">\n')
    for c in colors:
        w(f'   <base name={quoteattr("Filament " + str(slot[c]) + " " + c)} displaycolor="{c}FF"/>\n')
    w("  </basematerials>\n")
    next_id = 2
    for i, p in enumerate(parts):
        oid = next_id; next_id += 1
        part_obj_id[i] = oid
        w(f'  <object id="{oid}" type="model" name={quoteattr(p.name)} pid="1" pindex="{slot[p.color] - 1}">\n')
        w("   <mesh>\n    <vertices>\n")
        m.append(("vertices", p.mesh.vertices))
        w("    </vertices>\n    <triangles>\n")
        m.append(("triangles", p.mesh.triangles))
        w("    </triangles>\n   </mesh>\n  </object>\n")
    if layout == "parts":
        for tname, idxs in tiles.items():
            aid = next_id; next_id += 1
            assembly_ids[tname] = aid
            w(f'  <object id="{aid}" type="model" name={quoteattr(tname)}>\n   <components>\n')
            for i in idxs:
                w(f'    <component objectid="{part_obj_id[i]}"/>\n')
            w("   </components>\n  </object>\n")
    w(" </resources>\n <build>\n")
    for oid in (assembly_ids.values() if layout == "parts" else part_obj_id.values()):
        w(f'  <item objectid="{oid}"/>\n')
    w(" </build>\n</model>\n")
    entries.append(("3D/3dmodel.model", m))

    # Bambu Studio / OrcaSlicer per-part filament assignment
    cfg = ['<?xml version="1.0" encoding="UTF-8"?>', "<config>"]
    if layout == "parts":
        for tname, idxs in tiles.items():
            cfg.append(f'  <object id="{assembly_ids[tname]}">')
            cfg.append(f'    <metadata key="name" value={quoteattr(tname)}/>')
            cfg.append(f'    <metadata key="extruder" value="{slot[parts[idxs[0]].color]}"/>')
            for i in idxs:
                p = parts[i]
                cfg.append(f'    <part id="{part_obj_id[i]}" subtype="normal_part">')
                cfg.append(f'      <metadata key="name" value={quoteattr(p.name)}/>')
                cfg.append(f'      <metadata key="extruder" value="{slot[p.color]}"/>')
                cfg.append("    </part>")
            cfg.append("  </object>")
    else:
        for i, p in enumerate(parts):
            cfg.append(f'  <object id="{part_obj_id[i]}">')
            cfg.append(f'    <metadata key="name" value={quoteattr(p.name)}/>')
            cfg.append(f'    <metadata key="extruder" value="{slot[p.color]}"/>')
            cfg.append("  </object>")
    cfg.append("</config>")
    entries.append(("Metadata/model_settings.config", [("\n".join(cfg) + "\n").encode()]))
    return entries


def write_3mf(path: Path, parts: list[Part], filaments: list[dict], title: str,
              layout: str = "parts", description: str = "", thumbnail: bytes | None = None) -> Path:
    return write_zip(path, _3mf_entries(parts, filaments, title, layout, description, thumbnail))


def png_bytes(rgb: np.ndarray) -> bytes:
    """Minimal PNG encoder (RGB8) - avoids a Pillow dependency."""
    import zlib
    h, w, _ = rgb.shape
    raw = np.concatenate([np.zeros((h, 1), np.uint8), rgb.reshape(h, w * 3)], axis=1).tobytes()

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def render_topdown(cell_rgb: np.ndarray, z_mm: np.ndarray, cell_mm: float, max_px: int = 1024) -> bytes:
    """Hill-shaded top view (north up) as PNG bytes."""
    zc = (z_mm[:-1, :-1] + z_mm[:-1, 1:] + z_mm[1:, :-1] + z_mm[1:, 1:]) / 4
    gy, gx = np.gradient(zc, cell_mm)
    # light from the north-west, 45 deg elevation (row 0 = south, so +y is north)
    nx_, ny_, nz_ = -gx, -gy, np.ones_like(zc)
    n = np.sqrt(nx_ ** 2 + ny_ ** 2 + nz_ ** 2)
    lx, ly, lz = -0.5, 0.5, 0.7071
    shade = np.clip((nx_ * lx + ny_ * ly + nz_ * lz) / n, 0, 1)
    shade = 0.35 + 0.65 * shade
    img = (cell_rgb.astype(np.float32) * shade[..., None]).clip(0, 255).astype(np.uint8)
    white = (cell_rgb == 255).all(axis=2)
    img[white] = 255
    img = img[::-1]  # north up
    k = int(np.ceil(max(img.shape[:2]) / max_px))
    if k > 1:
        img = img[::k, ::k]
    return png_bytes(np.ascontiguousarray(img))


def _stl_bytes(name: str, v: np.ndarray, t: np.ndarray) -> bytes:
    tri = v[t].astype(np.float32)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.divide(n, ln, out=np.zeros_like(n), where=ln > 0)
    rec = np.zeros(len(t), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"] = n
    rec["v"] = tri
    header = name.encode("ascii", "replace")[:80].ljust(80, b" ")
    return header + struct.pack("<I", len(t)) + rec.tobytes()


def write_stl_zip(path: Path, parts: list[Part]) -> Path:
    entries: list[tuple[str, list[Segment]]] = []
    used = set()
    for p in parts:
        base = "".join(ch if ch.isalnum() or ch in "-_ " else "_" for ch in p.name).strip().replace(" ", "_")
        fname = f"{base}_{p.color[1:]}.stl"
        k = 2
        while fname in used:
            fname = f"{base}_{p.color[1:]}_{k}.stl"; k += 1
        used.add(fname)
        entries.append((fname, [_stl_bytes(p.name, p.mesh.vertices, p.mesh.triangles)]))
    return write_zip(path, entries)


# 3D-preview levels of detail (triangle budgets): browsers turn every triangle into three unshared
# vertices for crisp-edged shading, so a 7 M-triangle model needs ~0.5 GB there; phones get "low"
PREVIEW_LODS = {"high": 2_000_000, "low": 600_000}


def _preview_triangles(parts: list[Part], budget: int) -> tuple[list[np.ndarray], bool]:
    """Per-part triangle arrays with about ``budget`` triangles in total: meshoptimizer
    simplification (native extension), else every k-th triangle (holes, but a bounded size)."""
    total = sum(len(p.mesh.triangles) for p in parts)
    if total <= budget:
        return [p.mesh.triangles for p in parts], False
    share = budget / total
    if native_enabled():
        from concurrent.futures import ThreadPoolExecutor

        def one(p: Part) -> np.ndarray:
            t = p.mesh.triangles
            if len(t) <= 256:
                return t
            return _native.simplify(np.ascontiguousarray(p.mesh.vertices, dtype=np.float32),
                                    np.ascontiguousarray(t, dtype=np.int32), max(128, int(len(t) * share)), 0.01)
        with ThreadPoolExecutor(max_workers=min(8, len(parts))) as ex:   # the GIL is released inside
            return list(ex.map(one, parts)), True
    return [p.mesh.triangles[:: max(1, int(round(1 / share)))] for p in parts], True


def preview_bytes(parts: list[Part], budget: int = PREVIEW_LODS["high"]) -> bytes:
    """Compact binary for the three.js viewer (version 2), gzip-compressed:

        uint32 header length | JSON header | per part: uint16 positions, int32 index deltas

    Positions are quantised over the whole model's bounding box (value = lo + q * scale, about
    3 um steps on a 200 mm model; shared corners of adjacent parts stay identical); indices are
    stored as differences from the previous index (prefix-sum to decode), which compresses far
    better than absolute indices. Models over ``budget`` triangles are simplified for display."""
    import gzip

    tris, simplified = _preview_triangles(parts, budget)
    allv = [p.mesh.vertices for p in parts if len(p.mesh.vertices)]
    lo = np.min([v.min(axis=0) for v in allv], axis=0).astype(np.float64) if allv else np.zeros(3)
    hi = np.max([v.max(axis=0) for v in allv], axis=0).astype(np.float64) if allv else np.ones(3)
    scale = np.maximum(hi - lo, 1e-9) / 65535.0
    header = {"version": 2, "lo": lo.tolist(), "scale": scale.tolist(), "parts": [],
              "triangles": int(sum(len(t) for t in tris)),
              "triangles_full": int(sum(len(p.mesh.triangles) for p in parts)), "simplified": simplified}
    blobs: list[bytes] = []
    offset = 0
    for p, t in zip(parts, tris):
        t = np.asarray(t, dtype=np.int64)
        used = np.zeros(len(p.mesh.vertices), bool)   # drop the vertices simplification left unused
        used[t.ravel()] = True
        remap = np.cumsum(used) - 1
        v = p.mesh.vertices[used].astype(np.float64)
        q = np.clip(np.round((v - lo) / scale), 0, 65535).astype("<u2")
        flat = remap[t].ravel()
        d = np.diff(flat, prepend=0).astype("<i4")
        vb = q.tobytes()
        vb += b"\0" * (-len(vb) % 4)                  # the int32 view needs 4-byte alignment
        header["parts"].append({"name": p.name, "color": p.color, "category": p.category,
                                "vOffset": offset, "vCount": len(q), "tOffset": offset + len(vb), "tCount": len(t)})
        blobs += [vb, d.tobytes()]
        offset += len(vb) + 4 * len(d)
    hj = json.dumps(header).encode()
    hj += b" " * ((4 - (len(hj) + 4) % 4) % 4)
    buf = io.BytesIO()
    buf.write(struct.pack("<I", len(hj)))
    buf.write(hj)
    for blob in blobs:
        buf.write(blob)
    return gzip.compress(buf.getvalue(), compresslevel=6, mtime=0)
