"""Streaming pass-through for data hosts a browser can't read directly (no CORS headers), for
generation in the browser (docs/plans/browser-wasm.md). ``/api/proxy?url=...`` forwards the
request (with ``Range``) and streams the response back without holding it.

Only URLs matching ``RULES`` (https, exact host, path prefix) are fetched - otherwise anyone could
use the server to reach arbitrary or internal addresses. Redirects are followed by hand and must
match too. Per client: a few concurrent requests and an hourly byte budget; overall: a cap on
concurrent requests. Responses can't run script under our origin (sandbox CSP, no HTML).
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from urllib.parse import urljoin, urlsplit

from . import config

# (host, path prefix): the elevation / map data hosts the providers read
RULES: tuple[tuple[str, str], ...] = (
    # range-read files (COG, COPC, PMTiles, EPT)
    ("canelevation-dem.s3.ca-central-1.amazonaws.com", "/"),
    ("canelevation-lidar-point-clouds.s3.ca-central-1.amazonaws.com", "/"),
    ("s3-us-west-2.amazonaws.com", "/usgs-lidar-public/"),
    ("prd-tnm.s3.amazonaws.com", "/StagedProducts/Elevation/"),
    ("usgslidareuwest.blob.core.windows.net", "/usgs-3dep-copc/"),
    ("ai4edataeuwest.blob.core.windows.net", "/alos-dem/"),
    ("copernicus-dem-30m.s3.amazonaws.com", "/"),
    ("copernicus-dem-90m.s3.amazonaws.com", "/"),
    ("huggingface.co", "/buckets/links-ads/fabdem/"),
    ("pgc-opendata-dems.s3.us-west-2.amazonaws.com", "/"),
    ("cdn.proj.org", "/"),
    ("download.mapterhorn.com", "/"),
    ("single-archive-tiles.mapterhorn.com", "/"),
    ("nz-elevation.s3.ap-southeast-2.amazonaws.com", "/"),
    ("open-lidar-data.s3.eu-central-1.amazonaws.com", "/"),
    ("phil-lidar-taal-s3.s3.us-east-2.amazonaws.com", "/"),
    ("data.geo.admin.ch", "/"),
    ("data.geopf.fr", "/"),
    ("service.pdok.nl", "/rws/actueel-hoogtebestand-nederland/"),
    ("download.data.public.lu", "/resources/"),
    ("data.bev.gv.at", "/download/"),
    ("srsp-open-data.s3.eu-west-2.amazonaws.com", "/"),
    ("srsp-open-data.s3-eu-west-2.amazonaws.com", "/"),
    ("dmwproductionblob.blob.core.windows.net", "/cogs/"),
    ("gsi.geodata.gov.ie", "/"),
    ("bdom20.s3.eu-de.cloud-object-storage.appdomain.cloud", "/"),
    ("dgm1.s3.eu-de.cloud-object-storage.appdomain.cloud", "/"),
    ("www.opengeodata.nrw.de", "/produkte/geobasis/"),
    ("download1.bayernwolke.de", "/a/"),
    ("geobasis-rlp.de", "/data/"),
    ("data.geobasis-bb.de", "/geobasis/daten/"),
    ("geocloud.landesvermessung.sachsen.de", "/public.php/"),
    ("www.nic.funet.fi", "/index/geodata/"),
    ("geoportaal.maaruum.ee", "/index.php"),
    ("ftp.lmi.is", "/gisdata/"),
    ("cyberjapandata.gsi.go.jp", "/xyz/"),
    ("bulkdata.csdi.gov.hk", "/"),
    # building sources (maptomesh.buildings; both send CORS headers, this is the fallback)
    ("tiles.overturemaps.org", "/"),
    ("assets.cms.plateau.reearth.io", "/assets/"),
    # catalogues and services
    ("stac.overturemaps.org", "/"),
    ("api.plateauview.mlit.go.jp", "/datacatalog/"),
    ("datacube.services.geo.ca", "/stac/"),
    ("planetarycomputer.microsoft.com", "/api/"),
    ("usgs.entwine.io", "/boundaries/"),
    ("raw.githubusercontent.com", "/hobuinc/usgs-lidar/"),
    ("stac.pgc.umn.edu", "/api/"),
    ("dgm.stac.lgln.niedersachsen.de", "/"),
    ("bdom.stac.lgln.niedersachsen.de", "/"),
    ("www.geodaten-mv.de", "/dienste/"),
    ("geo.api.vlaanderen.be", "/DHMV/"),
    ("environment.data.gov.uk", "/spatialdata/"),
    ("api.remotesensing.data.gov.scot", "/"),
    ("servicios.idee.es", "/wcs-inspire/"),
    ("geoservices9.civis.bz.it", "/geoserver/"),
    ("www502.regione.toscana.it", "/wmsraster/"),
    ("utility.arcgis.com", "/usrsvcs/servers/fef66dec83c14b0295180ecafa662aa0/"),
    ("services3.arcgis.com", "/6j1KwZfY2fZrfNMR/"),
    ("phillidar-dad.github.io", "/data/"),
    ("spatial-img.information.qld.gov.au", "/arcgis/"),
    ("services.ga.gov.au", "/gis/"),
    ("gis.mdba.gov.au", "/arcgis/"),
    ("tiles.openfreemap.org", "/"),
)
# POST only for searches (STAC / catalogue APIs)
POST_RULES: tuple[tuple[str, str], ...] = (
    ("planetarycomputer.microsoft.com", "/api/stac/"),
    ("api.remotesensing.data.gov.scot", "/search/"),
    ("stac.pgc.umn.edu", "/api/"),
    ("datacube.services.geo.ca", "/stac/"),
)
# hosts without CORS: the browser goes through the proxy straight away (measured 2026-11-19,
# _scratch/wasm/cors.py); everything else is tried directly first
NO_CORS: tuple[str, ...] = (
    "copernicus-dem-30m.s3.amazonaws.com", "copernicus-dem-90m.s3.amazonaws.com",
    "dmwproductionblob.blob.core.windows.net", "phil-lidar-taal-s3.s3.us-east-2.amazonaws.com",
    "ftp.lmi.is", "geoportaal.maaruum.ee", "gsi.geodata.gov.ie", "bulkdata.csdi.gov.hk",
    "download1.bayernwolke.de", "geocloud.landesvermessung.sachsen.de",
)

MAX_RESPONSE_BYTES = int(float(os.environ.get("MAPTOMESH_PROXY_MAX_MB", "1024")) * 1e6)
CLIENT_BYTES_PER_HOUR = int(float(os.environ.get("MAPTOMESH_PROXY_GB_PER_HOUR", "4")) * 1e9)
CLIENT_CONCURRENT = 16
GLOBAL_CONCURRENT = int(os.environ.get("MAPTOMESH_PROXY_CONCURRENT", "64"))
FORWARD_REQUEST = ("range", "if-none-match", "if-modified-since", "if-match", "accept", "content-type")
FORWARD_RESPONSE = ("content-type", "content-length", "content-range", "accept-ranges", "etag",
                    "last-modified", "content-encoding", "cache-control")
SAFE_TYPES = ("application/", "image/", "text/plain", "text/xml", "text/csv")


def _extra_rules() -> tuple[tuple[str, str], ...]:
    """Configurable data URLs (a self-hosted Mapterhorn mirror, another vector-tile server)."""
    out = []
    for var in ("MAPTOMESH_MAPTERHORN_URL", "MAPTOMESH_VECTOR_TILES"):
        u = os.environ.get(var)
        if u:
            p = urlsplit(u)
            if p.scheme == "https" and p.hostname:
                out.append((p.hostname, "/"))
    return tuple(out)


def allowed(url: str, method: str = "GET") -> bool:
    try:
        p = urlsplit(url)
    except ValueError:
        return False
    if p.scheme != "https" or not p.hostname or p.username or p.password or p.port not in (None, 443):
        return False
    rules = POST_RULES if method == "POST" else RULES + _extra_rules()
    path = p.path or "/"
    return any(p.hostname == h and path.startswith(prefix) for h, prefix in rules)


class Refused(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


class _Clients:
    def __init__(self):
        self.lock = threading.Lock()
        self.active: dict[str, int] = {}
        self.used: dict[str, deque] = {}
        self.total = 0

    def enter(self, client: str) -> None:
        with self.lock:
            if self.total >= GLOBAL_CONCURRENT:
                raise Refused(503, "The data proxy is busy; try again shortly")
            if self.active.get(client, 0) >= CLIENT_CONCURRENT:
                raise Refused(429, "Too many simultaneous downloads")
            q = self.used.setdefault(client, deque())
            now = time.time()
            while q and q[0][0] < now - 3600:
                q.popleft()
            if sum(n for _, n in q) >= CLIENT_BYTES_PER_HOUR:
                raise Refused(429, "Hourly download allowance through the server used up; try later")
            self.active[client] = self.active.get(client, 0) + 1
            self.total += 1

    def leave(self, client: str, nbytes: int) -> None:
        with self.lock:
            self.active[client] = max(0, self.active.get(client, 1) - 1)
            if not self.active[client]:
                del self.active[client]
            self.total = max(0, self.total - 1)
            self.used.setdefault(client, deque()).append((time.time(), nbytes))
            if len(self.used) > 10_000:          # forget idle clients
                for k in [k for k, q in self.used.items() if k not in self.active][:5000]:
                    del self.used[k]


CLIENTS = _Clients()
_local = threading.local()


def _session():
    import requests
    s = getattr(_local, "s", None)
    if s is None:
        s = requests.Session()
        s.headers["User-Agent"] = config.USER_AGENT
        _local.s = s
    return s


def open_upstream(url: str, method: str, headers: dict, body: bytes | None, client: str):
    """-> (status, response headers, byte iterator). The iterator releases the client's slot."""
    if not allowed(url, method):
        raise Refused(403, "Not a data source this server forwards")
    CLIENTS.enter(client)
    sent = 0
    try:
        h = {k: v for k, v in headers.items() if k.lower() in FORWARD_REQUEST}
        # byte ranges must be of the file itself, not of a compressed transfer; otherwise let the
        # upstream compress (the response's Content-Encoding is passed on; the browser decodes it)
        h["Accept-Encoding"] = "identity" if any(k.lower() == "range" for k in h) else             headers.get("accept-encoding", "gzip, deflate")
        r = None
        for _ in range(4):
            r = _session().request(method, url, headers=h, data=body, stream=True, allow_redirects=False,
                                   timeout=(15, 120))
            if r.status_code in (301, 302, 303, 307, 308) and r.headers.get("Location"):
                nxt = urljoin(url, r.headers["Location"])
                r.close()
                if not allowed(nxt, "GET" if r.status_code == 303 else method):
                    raise Refused(502, "The data source redirected somewhere this server doesn't forward")
                url, method = nxt, ("GET" if r.status_code == 303 else method)
                continue
            break
        else:
            raise Refused(502, "Too many redirects")
        size = r.headers.get("Content-Length")
        if size and size.isdigit() and int(size) > MAX_RESPONSE_BYTES:
            r.close()
            raise Refused(413, "File too large to forward; pick a smaller area")
        out = {k: r.headers[k] for k in r.headers if k.lower() in FORWARD_RESPONSE}
        ctype = r.headers.get("Content-Type", "application/octet-stream").lower()
        if not ctype.startswith(SAFE_TYPES) or "html" in ctype or "svg" in ctype or "javascript" in ctype:
            out["Content-Type"] = "text/plain; charset=utf-8" if ("html" in ctype or "javascript" in ctype) \
                else "application/octet-stream"
        out["Content-Security-Policy"] = "default-src 'none'; sandbox"
        out["X-Content-Type-Options"] = "nosniff"
        out["Cross-Origin-Resource-Policy"] = "same-origin"
    except BaseException:
        CLIENTS.leave(client, sent)
        raise

    def body_iter():
        nonlocal sent
        try:
            for chunk in r.raw.stream(64 * 1024, decode_content=False):
                sent += len(chunk)
                if sent > MAX_RESPONSE_BYTES:
                    break
                yield chunk
        finally:
            r.close()
            CLIENTS.leave(client, sent)

    return r.status_code, out, body_iter()
