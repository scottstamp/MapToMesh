"""The native 3MF writer (native/, Rust) must produce exactly the Python writer's content.

    python tests/test_native_writer.py      # skips when the extension isn't built

Compares every zip entry of both writers for awkward numbers (rounding ties on a 1/32 mm grid,
-0.0, tiny negatives, float64, int64, empty and non-contiguous arrays) and for a full 3MF of a
heightfield mesh; both archives must also pass zipfile's CRC check.
"""

from __future__ import annotations

import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from maptomesh import export  # noqa: E402
from maptomesh.mesh import heightfield_solid  # noqa: E402
from maptomesh.model import Part  # noqa: E402


def entries(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None, f"{path.name}: CRC error"
        return {i.filename: z.read(i) for i in z.infolist()}


def same(a: Path, b: Path) -> bool:
    ea, eb = entries(a), entries(b)
    return list(ea) == list(eb) and all(ea[k] == eb[k] for k in ea)


def native_zip(path: Path, items) -> None:
    export._native.write_zip(str(path), [(n, [s if isinstance(s, bytes) else (s[0], np.ascontiguousarray(s[1]))
                                              for s in segs]) for n, segs in items], export.NATIVE_LEVEL,
                                 decimals=export.DECIMALS)


def main() -> int:
    if export._native is None:
        print("SKIP: maptomesh_native is not built (maturin develop --release -m native/Cargo.toml)")
        return 0
    rng = np.random.default_rng(7)
    ties = np.array([[0.03125, -0.03125, 12.00005], [-0.0, 0.0, -0.00001], [1e6, -123456.78125, 0.99995],
                     [0.0625, -0.0625, 0.0005], [2.0625, -0.0004, 0.9995],
                     [2.5e-5, 7.5e-5, 0.00015]], np.float32)
    rand = (rng.random((40_000, 3)) * 400 - 200).astype(np.float32)
    cases = {
        "rounding ties / signs (float32)": [("vertices", ties)],
        "rounding ties / signs (float64)": [("vertices", ties.astype(np.float64))],
        "1/32 mm grid (ties)": [("vertices", (np.round(rng.random((40_000, 3)) * 8000) / 32).astype(np.float32))],
        "1/16 mm grid (ties at 3 decimals)": [("vertices", (np.round(rng.random((40_000, 3)) * 4000) / 16).astype(np.float32))],
        "random float32": [("vertices", rand)],
        "random float64": [("vertices", rng.random((10_000, 3)) * 1e4 - 5e3)],
        "int32 triangles": [("triangles", rng.integers(0, 2**31 - 1, (70_000, 3)).astype(np.int32))],
        "int64 triangles": [("triangles", rng.integers(0, 2**40, (1_000, 3)).astype(np.int64))],
        "empty arrays": [b"<a>", ("vertices", np.zeros((0, 3), np.float32)), ("triangles", np.zeros((0, 3), np.int32)), b"</a>"],
        "non-contiguous": [("vertices", rand[::3])],
        "large raw bytes (split across cores)": [rng.integers(0, 4, 9_500_000, dtype=np.uint8).tobytes(), b"tail",
                                                 ("triangles", np.arange(30, dtype=np.int32).reshape(10, 3)), b"x" * 5_000_000],
    }
    failures = 0
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        for name, segs in cases.items():
            items = [("data.xml", segs), ("small.txt", [b"hello"])]
            export._write_zip_py(d / "py.zip", items)
            native_zip(d / "rs.zip", items)
            ok = same(d / "py.zip", d / "rs.zip")
            failures += not ok
            print(f"{'PASS' if ok else 'FAIL'} {name}")

        # a real 3MF: heightfield solids, two parts, thumbnail, both layouts
        R, C = 180, 240
        yy, xx = np.mgrid[0:R + 1, 0:C + 1]
        top = (3 + 2 * np.sin(xx / 17.0) * np.cos(yy / 23.0)).astype(np.float32)
        mask = np.ones((R, C), bool)
        m1 = heightfield_solid(mask, top, 0.0, 0.25, 0.25)
        m2 = heightfield_solid(mask[:60, :80], top[:61, :81] + 1, 0.5, 0.25, 0.25, ox=5.0, oy=3.0)
        parts = [Part("Ground", "#E9E4D8", m1, "Tile 1", "ground"), Part("Water", "#2F8FD8", m2, "Tile 1", "water")]
        fil = [{"slot": 1, "color": "#E9E4D8", "used_by": ["Ground"]}, {"slot": 2, "color": "#2F8FD8", "used_by": ["Water"]}]
        for layout in ("parts", "objects"):
            items = export._3mf_entries(parts, fil, "Test <&> \"model\"", layout, "desc", b"\x89PNG fake")
            export._write_zip_py(d / "py.3mf", items)
            native_zip(d / "rs.3mf", items)
            ok = same(d / "py.3mf", d / "rs.3mf")
            failures += not ok
            print(f"{'PASS' if ok else 'FAIL'} 3MF, layout={layout} ({len(m1.triangles) + len(m2.triangles):,} triangles)")
    print("all passed" if not failures else f"{failures} FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
