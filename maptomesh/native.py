"""Optional Rust extension (``native/``, PyO3): faster 3MF / zip writing, preview simplification
and ground plane fitting. Everything has a pure-Python fallback, used when the extension isn't
built or ``MAPTOMESH_NATIVE=0``."""

from __future__ import annotations

import os

try:
    import maptomesh_native as ext
except ImportError:  # not built: maturin develop --release -m native/Cargo.toml
    ext = None


def enabled() -> bool:
    return ext is not None and os.environ.get("MAPTOMESH_NATIVE", "1") != "0"
