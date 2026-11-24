# Running the server on Alpine Linux

Alpine uses musl, and rasterio publishes no musl (`musllinux`) wheels, so pip builds it from
source. That build needs GDAL's headers and `gdal-config`; without them it stops with

```
ERROR: A GDAL API version must be specified. Provide a path to gdal-config using a GDAL_CONFIG
environment variable or use a GDAL_VERSION environment variable.
```

Installing `gdal-dev` fixes it. Everything else in `requirements.txt` has musl wheels. lazrs is
the one exception if PyPI has no wheel for your Python, and it then builds with the Rust
toolchain installed below.

Run everything below as root.

## 1. System packages

```sh
apk add python3 python3-dev build-base git gdal-dev cargo
```

- `gdal-dev`: GDAL headers and `gdal-config` for the rasterio build. It pulls in the `gdal` and
  `proj` runtime libraries.
- `build-base`, `python3-dev`: C/C++ compiler and Python headers (rasterio is Cython).
- `cargo`: Rust, for the optional native extension (`native/`, 5-15x faster meshing and 3MF
  writing). It needs Rust 1.85 or newer (edition 2024); check `rustc --version`.

## 2. Service user and code

The code is owned by root and read-only to the server. Its cache, outputs and data go to
`/var/lib/maptomesh`.

```sh
addgroup -S maptomesh
adduser -S -D -H -h /var/lib/maptomesh -s /sbin/nologin -G maptomesh maptomesh

git clone https://github.com/scottstamp/MapToMesh.git /opt/maptomesh
cd /opt/maptomesh
python3 -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install -r requirements.txt       # rasterio builds from source: a few minutes
.venv/bin/pip install ./native                  # optional native extension: a few minutes, ~1-2 GB RAM
.venv/bin/python -m compileall -q maptomesh run.py
```

`compileall` matters. Each job runs in a fresh worker process, and the service user can't write
`__pycache__` under `/opt`. Without it, every job would recompile every module.

Check that rasterio found GDAL and PROJ, and that the native extension loads:

```sh
.venv/bin/python -c "import rasterio; from rasterio.crs import CRS; print('GDAL', rasterio.__gdal_version__, CRS.from_epsg(3979))"
.venv/bin/python -c "import maptomesh_native; print('native extension OK')"
```

## 3. OpenRC service

```sh
install -m 0755 deploy/alpine/maptomesh.initd /etc/init.d/maptomesh
install -m 0644 deploy/alpine/maptomesh.confd /etc/conf.d/maptomesh
rc-update add maptomesh default
rc-service maptomesh start
tail -f /var/log/maptomesh/maptomesh.log
```

The service runs `run.py --no-browser` as `maptomesh` under `supervise-daemon`, which restarts it
if it crashes. By default it listens on `127.0.0.1:8000`. On first start it downloads NRCan's
LiDAR tile index (~230 MB, once). Quick check: `wget -qO- http://127.0.0.1:8000/ | head -5`.

Settings go in `/etc/conf.d/maptomesh`. That covers the listen address, paths, and any
`MAPTOMESH_*` variable from the README's Configuration table; write those with `export`. In LXC
or other containers, set `MAPTOMESH_MEMORY_MB` to the container's memory limit. The default is
80% of the RAM psutil reports, which can be the host's.

Log rotation:

```sh
apk add logrotate
install -m 0644 deploy/alpine/maptomesh.logrotate /etc/logrotate.d/maptomesh
rc-update add crond default && rc-service crond start
```

## 4. Exposing it

- **cloudflared tunnel** (simplest): keep `127.0.0.1` and point the tunnel at
  `http://127.0.0.1:8000`.
- **Proxied origin**: set `maptomesh_host="0.0.0.0"`, `MAPTOMESH_ORIGIN_SECRET` and
  `MAPTOMESH_ALLOWED_HOSTS`, and firewall the port to Cloudflare's ranges.

Both are covered in the README's "Deployment behind Cloudflare". Ports below 1024 need root, so
put a tunnel or reverse proxy in front instead of running the server on 80/443. Never set
`MAPTOMESH_LOCAL_UNLIMITED` on a server.

## 5. Updating

```sh
cd /opt/maptomesh
git pull
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install --force-reinstall --no-deps ./native    # when native/ changed
.venv/bin/python -m compileall -q maptomesh run.py
rc-service maptomesh restart
```

If `deploy/alpine/maptomesh.initd` changed, install it again (step 3).

## Troubleshooting

| Symptom | Fix |
|---|---|
| `A GDAL API version must be specified` | `apk add gdal-dev` (provides `gdal-config`), then rerun pip |
| lazrs or `./native` build: `cargo` not found | `apk add cargo` |
| `./native`: `feature edition2024 is required` | Rust older than 1.85: newer Alpine, or Rust from rustup |
| rasterio check: error mentioning `proj.db` | GDAL can't find PROJ's database: `apk add proj` and check `/usr/share/proj/proj.db` exists |
| Jobs killed for memory / OOM in a container | set `MAPTOMESH_MEMORY_MB` to the container's limit |
| `rc-service maptomesh start` fails at once | read `/var/log/maptomesh/maptomesh.log`, and check `maptomesh_dir` in `/etc/conf.d/maptomesh` points at the clone with `.venv` |
