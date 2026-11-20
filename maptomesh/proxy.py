"""Streaming pass-through for data hosts a browser can't read directly (no CORS headers), for
generation in the browser (docs/plans/browser-wasm.md). ``/api/proxy?url=...`` forwards the
request (with ``Range``) and streams the response back without holding it.

Only URLs matching ``RULES`` (https, exact host, path prefix) are fetched - otherwise anyone could
use the server to reach arbitrary or internal addresses. A path prefix only means something if the
path can't climb out of it, so paths with dot segments (``..``, ``%2e%2e``, ``..;``), empty segments,
backslashes or encoded slashes are refused, and the URL checked is the exact URL sent (``check``
canonicalises it the way ``requests`` would and refuses anything the library would still change).
Redirects are followed by hand and every hop is checked the same way.

Per client (IPv4 address / IPv6 /64): a few concurrent requests and an hourly byte budget counted
while the bytes stream; overall: a cap on concurrent requests. A stream whose reader stalls or
trickles is cut off so it can't hold its slot (``reap``). Responses can't run script under our
origin (sandbox CSP, no HTML), and the upstream session keeps no cookies and ignores the
environment (no ``.netrc`` credentials, no proxy variables).
"""

from __future__ import annotations

import http.cookiejar
import ipaddress
import os
import re
import threading
import time
import weakref
from collections import deque
from urllib.parse import urljoin, urlsplit

from .limits import LOCAL

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
CLIENT_CONCURRENT = int(os.environ.get("MAPTOMESH_PROXY_CLIENT_CONCURRENT", "16"))
GLOBAL_CONCURRENT = int(os.environ.get("MAPTOMESH_PROXY_CONCURRENT", "64"))
MAX_BODY = 64_000                   # POST bodies (STAC searches)
MAX_URL = 4000
# a stream whose reader takes nothing for IDLE_SECONDS, or less than MIN_BYTES_PER_SECOND over a
# RATE_WINDOW, is cut off and frees its slot (time spent waiting on the upstream doesn't count)
IDLE_SECONDS = float(os.environ.get("MAPTOMESH_PROXY_IDLE_S", "30"))
RATE_WINDOW = 60.0
MIN_BYTES_PER_SECOND = float(os.environ.get("MAPTOMESH_PROXY_MIN_KB_PER_S", "4")) * 1e3
REAP_EVERY = 5.0
CHUNK = 64 * 1024
METHODS = ("GET", "HEAD", "POST")
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


class Refused(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


class StreamAborted(Exception):
    """A response cut off part-way (budget used up, too large, reader too slow): the client sees a
    broken transfer rather than a short file that looks complete."""


_NETLOC = re.compile(r"[a-z0-9.-]+(?::443)?")
# encoded '.', '/', '\', '%' (double encoding) and NUL: no data URL needs them in its path, and
# each can become a segment that climbs out of a prefix at some server or library
_BAD_ESCAPES = ("%2e", "%2f", "%5c", "%25", "%00")


def _path_ok(path: str) -> bool:
    if not path.startswith("/") or "\\" in path:
        return False
    low = path.lower()
    if any(e in low for e in _BAD_ESCAPES):
        return False
    segs = path.split("/")[1:]
    for i, seg in enumerate(segs):
        if not seg:
            if i != len(segs) - 1:          # '//' (a trailing '/' is fine)
                return False
        elif seg.split(";", 1)[0] in (".", ".."):   # also '..;x' (path parameters)
            return False
    return True


def _rule_ok(url: str, method: str) -> bool:
    p = urlsplit(url)
    path = p.path or "/"            # what requests sends for an empty path
    if p.scheme != "https" or "#" in url or not _NETLOC.fullmatch(p.netloc.lower()) or not _path_ok(path):
        return False
    rules = POST_RULES if method == "POST" else RULES + _extra_rules()
    return any(p.hostname == h and path.startswith(prefix) for h, prefix in rules)


def _prepared_url(method: str, url: str) -> str:
    import requests
    return requests.Request(method, url).prepare().url


def check(url: str, method: str = "GET") -> str:
    """The exact URL to send for ``url``, or ``Refused(403)``. ``url`` must be on the allowlist both
    as given and as ``requests`` rewrites it (it decodes %2e, removes dot segments, encodes spaces),
    and the rewritten form must be final: sent unchanged, with no further normalising between this
    check and the wire."""
    refused = Refused(403, "Not a data source this server forwards")
    method = (method or "").upper()
    if method not in METHODS or not isinstance(url, str) or len(url) > MAX_URL:
        raise refused
    if any(ord(c) < 0x20 or ord(c) == 0x7F for c in url) or "\\" in url:
        raise refused
    try:
        if not _rule_ok(url, method):
            raise refused
        sent = _prepared_url(method, url)
        if not _rule_ok(sent, method) or _prepared_url(method, sent) != sent:
            raise refused
        from urllib3.util.url import _encode_target
        p = urlsplit(sent)
        target = p.path + (f"?{p.query}" if p.query else "")
        if _encode_target(target) != target:
            raise refused
    except Refused:
        raise
    except Exception:  # noqa: BLE001 - unparseable (bad port, IDNA, ...)
        raise refused from None
    return sent


def allowed(url: str, method: str = "GET") -> bool:
    """Whether ``url`` may be forwarded with ``method``. Network-free and cheap: the route calls it
    before reading a request body."""
    try:
        check(url, method)
        return True
    except Refused:
        return False


def client_key(host: str) -> str:
    """Whom limits apply to: an IPv4 address, or an IPv6 /64 (one host usually has a whole /64)."""
    try:
        ip = ipaddress.ip_address((host or "").split("%", 1)[0])
    except ValueError:
        return host or "unknown"
    if ip.version == 6:
        if ip.ipv4_mapped:
            return str(ip.ipv4_mapped)
        return str(ipaddress.ip_network(f"{ip}/64", strict=False))
    return str(ip)


class _Clients:
    """Slots and hourly byte budgets per client key (``client_key``)."""
    BUCKET = 60.0                   # bytes are counted per minute

    def __init__(self):
        self.lock = threading.Lock()
        self.active: dict[str, int] = {}
        self.used: dict[str, deque] = {}       # key -> [[minute start, bytes], ...] over the last hour
        self.used_total: dict[str, int] = {}
        self.total = 0

    def _used(self, key: str, now: float) -> int:
        q = self.used.get(key)
        if not q:
            return 0
        while q and q[0][0] <= now - 3600:
            self.used_total[key] -= q.popleft()[1]
        if not q:
            del self.used[key], self.used_total[key]
            return 0
        return self.used_total[key]

    def enter(self, key: str, now: float | None = None) -> None:
        if key == LOCAL:                # this machine (tests): no limits, not counted
            return
        now = time.monotonic() if now is None else now
        with self.lock:
            if self.total >= GLOBAL_CONCURRENT:
                raise Refused(503, "The data proxy is busy; try again shortly")
            if self.active.get(key, 0) >= CLIENT_CONCURRENT:
                raise Refused(429, "Too many simultaneous downloads")
            if self._used(key, now) >= CLIENT_BYTES_PER_HOUR:
                raise Refused(429, "Hourly download allowance through the server used up; try later")
            self.active[key] = self.active.get(key, 0) + 1
            self.total += 1

    def add(self, key: str, nbytes: int, now: float | None = None) -> bool:
        """Count bytes as they stream; False once the key's hourly budget is exceeded."""
        if key == LOCAL:
            return True
        now = time.monotonic() if now is None else now
        with self.lock:
            used = self._used(key, now) + nbytes
            q = self.used.setdefault(key, deque())
            if q and q[-1][0] > now - self.BUCKET:
                q[-1][1] += nbytes
            else:
                q.append([now, nbytes])
            self.used_total[key] = used
            if len(self.used) > 10_000:          # forget clients with nothing in the last hour
                for k in list(self.used):
                    self._used(k, now)
            return used <= CLIENT_BYTES_PER_HOUR

    def leave(self, key: str) -> None:
        if key == LOCAL:
            return
        with self.lock:
            self.active[key] = max(0, self.active.get(key, 1) - 1)
            if not self.active[key]:
                del self.active[key]
            self.total = max(0, self.total - 1)


CLIENTS = _Clients()
_STREAMS: weakref.WeakSet = weakref.WeakSet()   # open streams, for reap
_STREAMS_LOCK = threading.Lock()
_reaper_started = False
_local = threading.local()


class _Stream:
    """The upstream body chunk by chunk, counted against the client's budget as it goes. Holds the
    client's slot until ``close``: at the end, on an error, when dropped, or when ``reap`` cuts off
    a stalled reader."""

    def __init__(self, r, key: str, now: float | None = None):
        self.r, self.key = r, key
        self._chunks = r.raw.stream(CHUNK, decode_content=False)
        self.sent = 0
        self.last = self.mark_t = time.monotonic() if now is None else now
        self.mark_sent = 0
        self.reading = False        # waiting on the upstream, not on the reader
        self.closed: str | None = None
        self._lock = threading.Lock()
        with _STREAMS_LOCK:
            _STREAMS.add(self)

    def __iter__(self):
        return self

    def __next__(self) -> bytes:
        if self.closed:
            if self.closed == "done":
                raise StopIteration
            raise StreamAborted(self.closed)
        self.reading = True
        try:
            chunk = next(self._chunks)
        except StopIteration:
            if self.closed:                 # closed under us (reap): not a complete file
                raise StreamAborted(self.closed) from None
            self.close("done")
            raise
        except Exception:
            if self.closed:
                raise StreamAborted(self.closed) from None
            self.close("upstream error")
            raise
        finally:
            self.last = time.monotonic()    # before 'reading' drops, so reap never sees a stale time
            self.reading = False
        if self.closed:
            raise StreamAborted(self.closed)
        self.sent += len(chunk)
        if self.sent > MAX_RESPONSE_BYTES:
            self.close("response too large")
            raise StreamAborted(self.closed)
        if not CLIENTS.add(self.key, len(chunk)):
            self.close("hourly download allowance used up")
            raise StreamAborted(self.closed)
        return chunk

    def __del__(self):              # dropped unfinished (client gone, never started)
        if hasattr(self, "_lock"):
            self.close("dropped")

    def close(self, reason: str = "closed") -> None:
        with self._lock:
            if self.closed:
                return
            self.closed = reason
        with _STREAMS_LOCK:
            _STREAMS.discard(self)
        try:
            self.r.close()
        finally:
            CLIENTS.leave(self.key)

    def stale(self, now: float) -> str | None:
        """Why this stream should be cut off now, if it should."""
        if self.reading:            # the upstream is slow, not the reader (it has its own timeout)
            self.mark_t, self.mark_sent = now, self.sent
            return None
        if now - self.last > IDLE_SECONDS:
            return "reader idle"
        if now - self.mark_t >= RATE_WINDOW:
            if self.sent - self.mark_sent < MIN_BYTES_PER_SECOND * (now - self.mark_t):
                return "reader too slow"
            self.mark_t, self.mark_sent = now, self.sent
        return None


def reap(now: float | None = None) -> int:
    """Close streams whose reader stalled or trickles; -> how many."""
    now = time.monotonic() if now is None else now
    with _STREAMS_LOCK:
        streams = list(_STREAMS)
    n = 0
    for s in streams:
        why = s.stale(now)
        if why:
            s.close(why)
            n += 1
    return n


def _reaper() -> None:
    while True:
        time.sleep(REAP_EVERY)
        try:
            reap()
        except Exception:  # noqa: BLE001 - keep reaping
            pass


def _start_reaper() -> None:
    global _reaper_started
    with _STREAMS_LOCK:
        if _reaper_started:
            return
        _reaper_started = True
    threading.Thread(target=_reaper, name="proxy-reaper", daemon=True).start()


def _session():
    import requests
    s = getattr(_local, "s", None)
    if s is None:
        s = requests.Session()
        s.trust_env = False         # no ~/.netrc credentials, no proxy / CA environment variables
        s.cookies.set_policy(http.cookiejar.DefaultCookiePolicy(allowed_domains=[]))  # keep none
        s.headers["User-Agent"] = config.USER_AGENT
        _local.s = s
    return s


def _response_headers(upstream) -> dict:
    out = {k.lower(): v for k, v in upstream.items() if k.lower() in FORWARD_RESPONSE}
    ctype = (upstream.get("Content-Type") or "").strip().lower()
    if not ctype:
        out["content-type"] = "application/octet-stream"
    elif "html" in ctype or "javascript" in ctype or "ecmascript" in ctype:
        out["content-type"] = "text/plain; charset=utf-8"
    elif not ctype.startswith(SAFE_TYPES) or "svg" in ctype:
        out["content-type"] = "application/octet-stream"
    out["content-security-policy"] = "default-src 'none'; sandbox"
    out["x-content-type-options"] = "nosniff"
    out["cross-origin-resource-policy"] = "same-origin"
    return out


def open_upstream(url: str, method: str, headers: dict, body: bytes | None, client: str):
    """-> (status, response headers, byte iterator). The iterator releases the client's slot when
    it ends or is dropped; ``client`` is the peer address (limits apply per ``client_key``)."""
    import requests

    method = (method or "").upper()
    url = check(url, method)
    if body is not None and len(body) > MAX_BODY:
        raise Refused(413, "Request body too large")
    key = client_key(client)
    _start_reaper()
    CLIENTS.enter(key)
    r = None
    try:
        h = {k: v for k, v in headers.items() if k.lower() in FORWARD_REQUEST}
        # byte ranges must be of the file itself, not of a compressed transfer; otherwise let the
        # upstream compress (the response's Content-Encoding is passed on; the browser decodes it)
        h["Accept-Encoding"] = "identity" if any(k.lower() == "range" for k in h) else \
            headers.get("accept-encoding", "gzip, deflate")
        s = _session()
        for _ in range(4):
            prep = s.prepare_request(requests.Request(method, url, headers=h, data=body))
            if prep.url != url:             # the library changed what was checked: don't send it
                raise Refused(403, "Not a data source this server forwards")
            try:
                r = s.send(prep, stream=True, allow_redirects=False, timeout=(15, 120))
            finally:
                s.cookies.clear()
            if r.status_code in (301, 302, 303, 307, 308) and r.headers.get("Location"):
                status, loc = r.status_code, r.headers["Location"]
                r.close()
                r = None
                nxt = "GET" if status == 303 and method != "HEAD" else method
                try:
                    url = check(urljoin(url, loc), nxt)
                except Refused:
                    raise Refused(502, "The data source redirected somewhere this server doesn't forward") \
                        from None
                if nxt != method:           # 303: the body isn't sent again
                    body = None
                    h = {k: v for k, v in h.items() if k.lower() != "content-type"}
                method = nxt
                continue
            break
        else:
            raise Refused(502, "Too many redirects")
        size = r.headers.get("Content-Length")
        if size and size.isdigit() and int(size) > MAX_RESPONSE_BYTES:
            raise Refused(413, "File too large to forward; pick a smaller area")
        out = _response_headers(r.headers)
        stream = _Stream(r, key)            # from here on the stream releases the slot
    except BaseException:
        if r is not None:
            r.close()
        CLIENTS.leave(key)
        raise
    return r.status_code, out, stream
