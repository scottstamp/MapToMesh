"""Build a trimmed SciPy wheel for in-browser builds (web/pyodide/).

Pyodide's SciPy wheel is ~14 MB, about half of what a first visit downloads; maptomesh uses only
``scipy.ndimage`` and ``scipy.spatial.cKDTree``. This keeps the subpackages those import (found by
running the integration cases and offline tests with every imported module recorded; all are
imported when ``scipy.ndimage`` / ``scipy.spatial`` load, nothing later) plus SciPy's top-level
files, its bundled OpenBLAS and metadata, and drops everything else including the test suites:
~6.5 MB.

If a build in the browser needs a dropped subpackage, the worker reloads with Pyodide's full SciPy
and retries, and the failure is reported (``python -m maptomesh.device_failures``): add the
subpackage to KEEP and run this again.

    python native/trim_scipy.py        # downloads the wheel of PYODIDE_VERSION, checks its sha256
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

PYODIDE_VERSION = "314.0.8"     # as in web/browser-worker.js
BASE = f"https://cdn.jsdelivr.net/pyodide/v{PYODIDE_VERSION}/full/"
KEEP = {"_external", "_lib", "constants", "linalg", "ndimage", "sparse", "spatial", "special"}
OUT_DIR = Path(__file__).resolve().parent.parent / "web" / "pyodide"
UA = {"User-Agent": "MapToMesh/0.1 (topographic 3MF generator)"}


def _get(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300) as r:
        return r.read()


def keep(name: str) -> bool:
    parts = name.split("/")
    if parts[0] != "scipy":
        return True                  # scipy.libs (OpenBLAS), scipy-*.dist-info
    if "tests" in parts:
        return False
    return len(parts) == 2 or parts[1] in KEEP


def main() -> None:
    lock = json.loads(_get(BASE + "pyodide-lock.json"))
    pkg = lock["packages"]["scipy"]
    name, sha = pkg["file_name"], pkg["sha256"]
    with tempfile.TemporaryDirectory() as tmp:     # the download stays outside the repo until checked
        src = Path(tmp) / name
        src.write_bytes(_get(BASE + name))
        full_mb = src.stat().st_size / 1e6
        digest = hashlib.sha256(src.read_bytes()).hexdigest()
        if digest != sha:
            sys.exit(f"sha256 mismatch for {name}: {digest} != {sha} (pyodide-lock.json)")
        buf = io.BytesIO()
        lines = []
        with zipfile.ZipFile(src) as zin, zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zout:
            record_name = next(n for n in zin.namelist() if n.endswith(".dist-info/RECORD"))
            for info in zin.infolist():
                if info.filename == record_name or not keep(info.filename):
                    continue
                data = zin.read(info)
                zout.writestr(info, data)
                h = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
                lines.append(f"{info.filename},sha256={h},{len(data)}")
            lines.append(f"{record_name},,")
            zout.writestr(record_name, "\n".join(lines) + "\n")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for old in OUT_DIR.glob("scipy-*.whl"):
        old.unlink()
    out = OUT_DIR / name
    out.write_bytes(buf.getvalue())
    print(f"{out.name}: {len(buf.getvalue()) / 1e6:.2f} MB of {full_mb:.2f} MB, {len(lines) - 1} files "
          f"(kept: {', '.join(sorted(KEEP))})")


if __name__ == "__main__":
    main()
