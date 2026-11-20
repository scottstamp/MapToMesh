#!/usr/bin/env bash
# Build lazrs (laspy's LAZ / COPC decompression backend: Rust + PyO3, github.com/laz-rs/lazrs-python)
# from its PyPI sdist as a Pyodide (WebAssembly) wheel and copy it to web/pyodide/, so laspy can read
# LAZ / COPC point clouds in the browser (Pyodide ships laspy's dependencies but not lazrs).
#
# Same toolchain as build_pyodide.sh (shared setup in pyodide_env.sh); on Windows run it inside WSL:
#
#   wsl -d Ubuntu -- bash /mnt/c/Users/Scott/source/repos/MapToMesh/native/build_lazrs_pyodide.sh
#
# (from Git Bash prefix MSYS_NO_PATHCONV=1, or Git Bash rewrites the /mnt/c path).
#
# The sdist must match both the sha256 PyPI publishes for it and the digest pinned here; the build uses
# its Cargo.lock as is (--locked). Like build_pyodide.sh's, the wheel is byte-for-byte reproducible.
# It is tagged cp314 (lazrs doesn't use PyO3's abi3): rebuild when the web app moves to a Pyodide on
# another CPython. ~15 s once the toolchain is installed.
# Threads: lazrs decompresses COPC (and laspy's default LazrsParallel backend) with rayon. Pyodide has
# no threads, so rayon's global pool can't spawn any; rayon-core (>= 1.11) then falls back to a pool of
# just the calling thread, i.e. sequential decompression. No patch needed.
#
# Overrides (environment): LAZRS_VERSION (default 0.8.2, the newest that laspy 2.7's `lazrs` extra
# accepts (<0.9) and what requirements.txt installs natively), LAZRS_SHA256 (sdist digest; defaults to
# the pinned one for the default version, else empty = trust PyPI's), plus those of build_pyodide.sh.
set -euo pipefail

LAZRS_DEFAULT_VERSION=0.8.2
LAZRS_DEFAULT_SHA256=80a30ad1798a9fd58e84f238ca23f5002555ba16553e2afedafa6bde494229e2
LAZRS_VERSION="${LAZRS_VERSION:-$LAZRS_DEFAULT_VERSION}"
if [ "$LAZRS_VERSION" = "$LAZRS_DEFAULT_VERSION" ]; then
    LAZRS_SHA256="${LAZRS_SHA256-$LAZRS_DEFAULT_SHA256}"
else
    LAZRS_SHA256="${LAZRS_SHA256-}"
fi

NATIVE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$NATIVE_DIR")"
OUT_DIR="$REPO/web/pyodide"
# shellcheck source=pyodide_env.sh
source "$NATIVE_DIR/pyodide_env.sh"

# --- sdist ----------------------------------------------------------------------------------------
WORK="$ROOT/lazrs"
SDIST="$WORK/lazrs-$LAZRS_VERSION.tar.gz"
mkdir -p "$WORK"
log "fetching lazrs $LAZRS_VERSION sdist (PyPI)"
read -r SDIST_URL PYPI_SHA256 < <(
    curl -fsSL "https://pypi.org/pypi/lazrs/$LAZRS_VERSION/json" | python -I -c '
import json, sys
(f,) = [f for f in json.load(sys.stdin)["urls"] if f["packagetype"] == "sdist"]
print(f["url"], f["digests"]["sha256"])'
) || true
if [ -z "${PYPI_SHA256:-}" ]; then
    echo "build_lazrs_pyodide.sh: no sdist for lazrs $LAZRS_VERSION on PyPI (or PyPI unreachable)" >&2
    exit 1
fi
curl -fsSL -o "$SDIST.part" "$SDIST_URL"
mv "$SDIST.part" "$SDIST"
SHA256="$(sha256sum "$SDIST" | cut -d' ' -f1)"
if [ "$SHA256" != "$PYPI_SHA256" ] || { [ -n "$LAZRS_SHA256" ] && [ "$SHA256" != "$LAZRS_SHA256" ]; }; then
    echo "build_lazrs_pyodide.sh: sha256 mismatch for $SDIST_URL: got $SHA256, PyPI says $PYPI_SHA256, pinned ${LAZRS_SHA256:-(none)}" >&2
    exit 1
fi
echo "sha256 $SHA256 OK"
SRC="$WORK/lazrs-$LAZRS_VERSION"
rm -rf "$SRC"
tar -xzf "$SDIST" -C "$WORK" --no-same-owner
# maturin's CycloneDX SBOM records the absolute build path (home directory, so the user name) and
# a timestamp; the browser has no use for it. Disable it in the build copy only.
grep -q '^\[tool\.maturin\.sbom\]' "$SRC/pyproject.toml" || printf '\n[tool.maturin.sbom]\nrust = false\n' >> "$SRC/pyproject.toml"

# --- build ----------------------------------------------------------------------------------------
log "building lazrs $LAZRS_VERSION (Pyodide $PYODIDE_VERSION, Emscripten $(pyodide config get emscripten_version), Rust $RUST_TOOLCHAIN)"
cd "$SRC"
export CARGO_TARGET_DIR="$WORK/target"      # kept across runs: rebuilds are incremental
export MATURIN_PEP517_ARGS="--locked"       # build exactly the dependency versions of the sdist's Cargo.lock
# Keep build paths (home directory, user name) out of the published wheel; it also makes the
# output independent of where it was built. Environment overrides pyodide-build's own config.
export CARGO_TARGET_WASM32_UNKNOWN_EMSCRIPTEN_RUSTFLAGS="$(pyodide config get rustflags) --remap-path-prefix=$HOME=/build --remap-path-prefix=$SRC=/build/lazrs"
export SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-315532800}"   # fixed wheel timestamps (1980-01-01)
rm -rf dist
pyodide build -o dist

WHEEL="$(ls dist/lazrs-*.whl)"
mkdir -p "$OUT_DIR"
rm -f "$OUT_DIR"/lazrs-*.whl
cp "$WHEEL" "$OUT_DIR/"
log "wheel: $OUT_DIR/$(basename "$WHEEL") ($(du -k "$WHEEL" | cut -f1) KB)"
