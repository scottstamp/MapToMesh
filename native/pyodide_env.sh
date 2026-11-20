# Pyodide cross-build toolchain shared by build_pyodide.sh and build_lazrs_pyodide.sh: source it, don't
# run it. Idempotent: installs whatever is missing, all under $HOME and without sudo (uv, a Python 3.14
# venv with pyodide-build, the Pyodide cross-build environment and its Emscripten SDK, rustup with the
# Rust toolchain that Pyodide release asks for plus the wasm32-unknown-emscripten target), then
# activates it: the venv (so `pyodide` is on PATH), RUSTUP_TOOLCHAIN, and a PATH without the Windows
# entries WSL appends. Leaves PYODIDE_VERSION, ROOT, EMSDK_DIR and RUST_TOOLCHAIN set and a log()
# helper defined. Overrides (environment): see the header of build_pyodide.sh.

PYODIDE_VERSION="${PYODIDE_VERSION:-314.0.8}"
PYODIDE_BUILD_VERSION="${PYODIDE_BUILD_VERSION:-0.39.4}"
ROOT="${MTM_PYODIDE_ROOT:-$HOME/.cache/maptomesh-pyodide}"

if [ "$(uname -s)" != "Linux" ]; then
    echo "$(basename "$0"): run this on Linux (on Windows: inside WSL, see the header)" >&2
    exit 1
fi

# WSL appends the Windows PATH: drop it so Windows cargo / python / node can't be picked up.
PATH="$(printf '%s' "$PATH" | tr ':' '\n' | grep -v '^/mnt/' | paste -sd: -)"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
log() { printf '\n== %s\n' "$*"; }

mkdir -p "$ROOT"

# --- uv + Python 3.14 venv with pyodide-build -------------------------------------------------
if ! command -v uv >/dev/null; then
    log "installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
fi
VENV="$ROOT/venv"
if [ ! -x "$VENV/bin/python" ]; then
    log "creating Python 3.14 venv"
    uv venv --python 3.14 "$VENV"
fi
if ! "$VENV/bin/python" -c "import importlib.metadata as m, sys; sys.exit(m.version('pyodide-build') != '$PYODIDE_BUILD_VERSION')" 2>/dev/null; then
    log "installing pyodide-build $PYODIDE_BUILD_VERSION"
    uv pip install --python "$VENV/bin/python" "pyodide-build==$PYODIDE_BUILD_VERSION"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# --- Pyodide cross-build environment + Emscripten ---------------------------------------------
if ! pyodide xbuildenv versions 2>/dev/null | grep -qw "$PYODIDE_VERSION"; then
    log "installing Pyodide xbuildenv $PYODIDE_VERSION"
    pyodide xbuildenv install "$PYODIDE_VERSION"
fi
pyodide xbuildenv use "$PYODIDE_VERSION" >/dev/null
EMSDK_DIR="$(pyodide config get emsdk_dir)"
if [ ! -x "$EMSDK_DIR/upstream/emscripten/emcc" ]; then
    log "installing Emscripten $(pyodide config get emscripten_version)"
    pyodide xbuildenv install-emscripten
fi

# --- Rust ---------------------------------------------------------------------------------------
RUST_TOOLCHAIN="$(pyodide config get rust_toolchain)"
if ! command -v rustup >/dev/null; then
    log "installing rustup"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs |
        sh -s -- -y --profile minimal --default-toolchain none --no-modify-path
fi
if ! rustup target list --installed --toolchain "$RUST_TOOLCHAIN" 2>/dev/null | grep -qx wasm32-unknown-emscripten; then
    log "installing Rust $RUST_TOOLCHAIN + wasm32-unknown-emscripten"
    rustup toolchain install "$RUST_TOOLCHAIN" --profile minimal --target wasm32-unknown-emscripten
fi
export RUSTUP_TOOLCHAIN="$RUST_TOOLCHAIN"
