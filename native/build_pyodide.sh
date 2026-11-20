#!/usr/bin/env bash
# Build the maptomesh_native extension as a Pyodide (WebAssembly) wheel and copy it to web/pyodide/.
#
# pyodide-build only runs on Linux, so on Windows run this inside WSL:
#
#   wsl -d Ubuntu -- bash /mnt/c/Users/Scott/source/repos/MapToMesh/native/build_pyodide.sh
#
# (from Git Bash prefix MSYS_NO_PATHCONV=1, or Git Bash rewrites the /mnt/c path).
#
# Idempotent: first installs whatever is missing, all under $HOME and without sudo (uv, a Python
# 3.14 venv with pyodide-build, the Pyodide cross-build environment and its Emscripten SDK, rustup
# with the Rust toolchain that Pyodide release asks for plus the wasm32-unknown-emscripten target;
# shared with build_lazrs_pyodide.sh in pyodide_env.sh), then builds. The crate is copied out of the
# repo for building (building on /mnt/c is slow); the cargo target dir there is kept, so rebuilds are
# incremental. The wheel is byte-for-byte reproducible (no build paths or timestamps in it). First
# run ~2 min + ~2.7 GB in $HOME.
#
# Overrides (environment): PYODIDE_VERSION (default 314.0.8: must match the Pyodide the web app
# loads; wheels are tied to its CPython / Emscripten ABI), PYODIDE_BUILD_VERSION (pyodide-build on
# PyPI, default 0.39.4), MTM_PYODIDE_ROOT (work dir, default ~/.cache/maptomesh-pyodide).
set -euo pipefail

NATIVE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$NATIVE_DIR")"
OUT_DIR="$REPO/web/pyodide"
# shellcheck source=pyodide_env.sh
source "$NATIVE_DIR/pyodide_env.sh"

# --- build ----------------------------------------------------------------------------------------
SRC="$ROOT/native"
log "copying $NATIVE_DIR -> $SRC"
mkdir -p "$SRC"
rm -rf "$SRC/src" "$SRC/dist"
(cd "$NATIVE_DIR" && tar --exclude=./target --exclude=./dist -cf - .) | tar -xf - -C "$SRC"
# maturin's CycloneDX SBOM records the absolute build path (home directory, so the user name) and
# a timestamp; the browser has no use for it. Disable it in the build copy only.
grep -q '^\[tool\.maturin\.sbom\]' "$SRC/pyproject.toml" || printf '\n[tool.maturin.sbom]\nrust = false\n' >> "$SRC/pyproject.toml"

log "building (Pyodide $PYODIDE_VERSION, Emscripten $(pyodide config get emscripten_version), Rust $RUST_TOOLCHAIN)"
cd "$SRC"
export CARGO_TARGET_DIR="$SRC/target"
# The meshopt crate's build.rs puts its own stripped-down libc headers (meant for bare
# wasm32-unknown-unknown) first on the include path of every wasm32 target; its math.h lacks the
# floorf / log2f its vendored meshoptimizer now uses. Emscripten has a real libc: pre-include its
# math.h (the stub only #defines a few names to builtins on top, which is harmless).
export CFLAGS_wasm32_unknown_emscripten="-include $EMSDK_DIR/upstream/emscripten/cache/sysroot/include/math.h"
# Keep build paths (home directory, user name) out of the published wheel; it also makes the
# output independent of where it was built. Environment overrides pyodide-build's own config.
export CARGO_TARGET_WASM32_UNKNOWN_EMSCRIPTEN_RUSTFLAGS="$(pyodide config get rustflags) --remap-path-prefix=$HOME=/build --remap-path-prefix=$SRC=/build/maptomesh-native"
export SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-315532800}"   # fixed wheel timestamps (1980-01-01)
pyodide build -o dist

WHEEL="$(ls dist/*.whl)"
mkdir -p "$OUT_DIR"
rm -f "$OUT_DIR"/maptomesh_native-*.whl
cp "$WHEEL" "$OUT_DIR/"
log "wheel: $OUT_DIR/$(basename "$WHEEL") ($(du -k "$WHEEL" | cut -f1) KB)"
