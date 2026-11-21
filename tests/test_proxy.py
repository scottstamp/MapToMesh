"""The data proxy (maptomesh/proxy.py), offline: allowlist bypass attempts, the URL actually sent,
redirect re-checks, response headers, session hygiene, slots, byte budgets and client keys.

    python tests/test_proxy.py
"""

from __future__ import annotations

import gc
import http.client
import io
import os
import sys
import tempfile
import time
import types
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.response import HTTPResponse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from maptomesh import proxy  # noqa: E402

GH = "https://raw.githubusercontent.com/hobuinc/usgs-lidar/master/boundaries/"
GH_REPO = "https://raw.githubusercontent.com/hobuinc/usgs-lidar/"


# --- the allowlist ------------------------------------------------------------------------------

BYPASSES = [
    # dot segments, raw and encoded, any case, mixed
    GH + "../../evil/repo/main/evil.3mf",
    GH + "master/../../../evil/repo/main/evil.3mf",
    GH + "./../evil.3mf",
    GH + ".",
    GH + "..",
    GH + "%2e%2e/%2e%2e/evil/repo/main/evil.3mf",
    GH + "%2E%2E/%2E%2E/evil/repo/main/evil.3mf",
    GH + "%2E%2e/%2e%2E/evil/repo/main/evil.3mf",
    GH + ".%2e/.%2E/evil/repo/main/evil.3mf",
    GH + "%2e./%2e./evil/repo/main/evil.3mf",
    GH + "%2e/x",
    GH + "%252e%252e/%252e%252e/evil/repo/main/evil.3mf",   # double encoded
    GH + "..;/..;/evil/repo/main/evil.3mf",                  # path parameters (Tomcat style)
    GH + "..;x=1/evil.3mf",
    GH + "%2e%2e;/evil.3mf",
    GH + "..%00/evil.3mf",
    # encoded slashes and backslashes, raw backslashes
    GH + "..%2f..%2fevil/repo/main/evil.3mf",
    GH + "..%2F..%2Fevil/repo/main/evil.3mf",
    GH + "..%5c..%5cevil/repo/main/evil.3mf",
    GH + "..%5C..%5Cevil/repo/main/evil.3mf",
    GH + "..\\..\\evil/repo/main/evil.3mf",
    GH + "x\\evil.3mf",
    GH + "%5c",
    # empty segments
    GH + "/../evil.3mf",
    GH + "a//b",
    "https://raw.githubusercontent.com//hobuinc/usgs-lidar/x",
    # control characters (urlsplit drops tabs / newlines, so '.\t.' would become '..')
    GH + ".\t./.\t./evil/repo/main/evil.3mf",
    GH + ".\n./evil.3mf",
    GH + ".\r./evil.3mf",
    GH + "x\x00",
    GH + "x\x7f",
    # fragments are never sent; refuse them rather than reason about them
    GH + "x#/../../evil",
    # the other prefix-restricted hosts
    "https://s3-us-west-2.amazonaws.com/usgs-lidar-public/../another-bucket/x.3mf",
    "https://s3-us-west-2.amazonaws.com/usgs-lidar-public/%2e%2e/another-bucket/x.3mf",
    "https://huggingface.co/buckets/links-ads/fabdem/../../../someone/model/resolve/main/x.3mf",
    "https://services3.arcgis.com/6j1KwZfY2fZrfNMR/../OtherOrg/arcgis/rest/x",
    "https://utility.arcgis.com/usrsvcs/servers/fef66dec83c14b0295180ecafa662aa0/../other/x",
    "https://planetarycomputer.microsoft.com/api/../evil",
    "https://www.opengeodata.nrw.de/produkte/geobasis/..%2F..%2Fother",
    "https://geoportaal.maaruum.ee/index.php/../../other.3mf",
    # the prefix itself without its trailing '/', other prefixes' near misses
    "https://raw.githubusercontent.com/hobuinc/usgs-lidar",
    "https://raw.githubusercontent.com/hobuinc/usgs-lidar-evil/x",
    "https://raw.githubusercontent.com/HOBUINC/usgs-lidar/x",
    # hosts, schemes, ports, credentials
    "https://raw.githubusercontent.com./hobuinc/usgs-lidar/x",
    "https://raw.githubusercontent.com:8443/hobuinc/usgs-lidar/x",
    "https://raw.githubusercontent.com:99999/hobuinc/usgs-lidar/x",
    "https://u:p@raw.githubusercontent.com/hobuinc/usgs-lidar/x",
    "https://@raw.githubusercontent.com/hobuinc/usgs-lidar/x",
    "https://raw.githubusercontent.com\\@evil.example/hobuinc/usgs-lidar/x",
    "https://evil.example\\@raw.githubusercontent.com/hobuinc/usgs-lidar/x",
    "http://raw.githubusercontent.com/hobuinc/usgs-lidar/x",
    "HTTPS://evil.example/hobuinc/usgs-lidar/x",
    "https:raw.githubusercontent.com/hobuinc/usgs-lidar/x",
    "https:/raw.githubusercontent.com/hobuinc/usgs-lidar/x",
    "//raw.githubusercontent.com/hobuinc/usgs-lidar/x",
    "https://169.254.169.254/latest/meta-data/",
    "https://[::1]/hobuinc/usgs-lidar/x",
    "https://127.0.0.1/",
    "",
    "https://",
    GH + "x" * 5000,
    # the rest of the repository: commits of forks / pull requests are readable through it by hash,
    # so anyone could place a file there (a fake job.json, a crafted 3MF)
    GH_REPO + "0123456789abcdef0123456789abcdef01234567/job.json",
    GH_REPO + "0123456789abcdef0123456789abcdef01234567/City.3mf",
    GH_REPO + "refs/pull/1/head/x.3mf",
    GH_REPO + "master/x.json",
    # ';' as an escape (a '..;' segment once decoded) and overlong UTF-8 for '.' / '/'
    GH + "..%3b/..%3b/evil.3mf",
    GH + "..%3B/evil.3mf",
    GH + "%c0%ae%c0%ae/%c0%ae%c0%ae/evil.3mf",
    GH + "%e0%80%ae/evil.3mf",
    GH + "%c0%af/evil.3mf",
    GH + "%ff/evil.3mf",
]

POST_BYPASSES = [
    "https://planetarycomputer.microsoft.com/api/stac/../../evil",
    "https://planetarycomputer.microsoft.com/api/stac/%2e%2e/%2e%2e/evil",
    "https://planetarycomputer.microsoft.com/api/stac/..%2f..%2fevil",
    "https://planetarycomputer.microsoft.com/api/data/v1/x",          # GET-only prefix
    "https://stac.pgc.umn.edu/api/../x",
    "https://download.mapterhorn.com/x",
]

ALLOWED = [
    # (given, sent)
    (GH + "resources.geojson", None),
    ("https://RAW.githubusercontent.com/hobuinc/usgs-lidar/master/boundaries/x.json", GH + "x.json"),   # host case
    ("https://raw.githubusercontent.com:443/hobuinc/usgs-lidar/master/boundaries/x.json", None),
    ("https://s3-us-west-2.amazonaws.com/usgs-lidar-public/CA_X/ept.json", None),
    ("https://s3-us-west-2.amazonaws.com/usgs-lidar-public/CA_X/ept-data/0-0-0-0.laz", None),
    ("https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N47_00_W053_00_DEM/"
     "Copernicus_DSM_COG_10_N47_00_W053_00_DEM.tif", None),
    ("https://copernicus-dem-30m.s3.amazonaws.com/", None),
    ("https://copernicus-dem-30m.s3.amazonaws.com", "https://copernicus-dem-30m.s3.amazonaws.com/"),
    ("https://huggingface.co/buckets/links-ads/fabdem/resolve/main/N40W080.tif", None),
    ("https://planetarycomputer.microsoft.com/api/sas/v1/token/usgs-3dep-copc", None),
    ("https://data.geopf.fr/wms-r?SERVICE=WMS&REQUEST=GetMap&LAYERS=x&BBOX=1,2,3,4", None),
    ("https://data.geopf.fr/x?q=../../y", None),                        # dots in the query only
    ("https://dmwproductionblob.blob.core.windows.net/cogs/a b.tif",       # spaces get encoded
     "https://dmwproductionblob.blob.core.windows.net/cogs/a%20b.tif"),
    ("https://dmwproductionblob.blob.core.windows.net/cogs/%c3%a9.tif",    # escapes get upper case
     "https://dmwproductionblob.blob.core.windows.net/cogs/%C3%A9.tif"),
    ("https://geoportaal.maaruum.ee/index.php?lang_id=1&plugin_act=otsing&kaardiruut=1&andmetyyp=x", None),
    ("https://cdn.proj.org/ca_nrc_HT2_2010v70.tif", None),
    ("https://cdn.proj.org/a..b/x...tif", None),                         # dots inside names are fine
    ("https://usgs.entwine.io/boundaries/resources.geojson", None),
    ("https://services3.arcgis.com/6j1KwZfY2fZrfNMR/arcgis/rest/services/x/ImageServer/exportImage"
     "?bbox=1,2,3,4&f=image", None),
]

POST_ALLOWED = [
    "https://planetarycomputer.microsoft.com/api/stac/v1/search",
    "https://stac.pgc.umn.edu/api/v1/search",
    "https://datacube.services.geo.ca/stac/api/search",
]


def test_bypasses_refused():
    for u in BYPASSES:
        for m in ("GET", "HEAD"):
            assert not proxy.allowed(u, m), (m, u)
    for u in POST_BYPASSES:
        assert not proxy.allowed(u, "POST"), u
    assert not proxy.allowed(GH + "x", "PUT") and not proxy.allowed(GH + "x", "DELETE")
    try:
        proxy.check(GH + "../../evil/x")
        raise AssertionError("not refused")
    except proxy.Refused as e:
        assert e.status == 403


def test_allowed_and_sent_unchanged():
    for given, sent in ALLOWED:
        out = proxy.check(given)
        assert out == (sent or given), (given, out)
        assert proxy.allowed(given) and proxy.allowed(given, "HEAD")
        # what's sent is final: requests doesn't rewrite it again
        p = requests.Request("GET", out).prepare()
        assert p.url == out, (out, p.url)
    for u in POST_ALLOWED:
        assert proxy.check(u, "post") == u


def test_extra_rules():
    old = os.environ.get("MAPTOMESH_MAPTERHORN_URL")
    os.environ["MAPTOMESH_MAPTERHORN_URL"] = "https://tiles.example.org/mapterhorn/"
    try:
        assert proxy.allowed("https://tiles.example.org/x.pmtiles")
        assert not proxy.allowed("https://tiles.example.org/x/../../y/%2e%2e/z")
    finally:
        if old is None:
            del os.environ["MAPTOMESH_MAPTERHORN_URL"]
        else:
            os.environ["MAPTOMESH_MAPTERHORN_URL"] = old


# --- fake upstream ------------------------------------------------------------------------------

class FakeUpstream(HTTPAdapter):
    """Answers from ``routes(prepared request) -> (status, headers, body)`` and records what would
    have gone on the wire."""

    def __init__(self, routes):
        super().__init__()
        self.routes = routes
        self.seen: list[dict] = []
        self.closed = 0

    def send(self, request, stream=False, timeout=None, verify=True, cert=None, proxies=None):
        self.seen.append({"method": request.method, "url": request.url, "body": request.body,
                          "target": self.request_url(request, proxies or {}),
                          "headers": dict(request.headers), "proxies": dict(proxies or {})})
        status, headers, body = self.routes(request)
        msg = http.client.HTTPMessage()
        for k, v in headers.items():
            msg[k] = v
        raw = HTTPResponse(body=io.BytesIO(body), headers=headers, status=status, preload_content=False,
                           decode_content=False, request_method=request.method)
        # so Set-Cookie reaches the session's jar (requests reads it from the http.client response)
        raw._original_response = types.SimpleNamespace(msg=msg, isclosed=lambda: False, close=lambda: None)
        return self.build_response(request, raw)


class _Fake:
    def __init__(self, routes):
        self.adapter = FakeUpstream(routes)

    def __enter__(self):
        proxy._local.__dict__.pop("s", None)
        proxy._session().mount("https://", self.adapter)
        return self.adapter

    def __exit__(self, *exc):
        proxy._local.__dict__.pop("s", None)
        assert not proxy.CLIENTS.active and proxy.CLIENTS.total == 0, (proxy.CLIENTS.active, "slot leaked")


def _ok(body=b"x" * 1000, headers=None):
    return lambda req: (200, headers or {"Content-Type": "image/tiff", "Content-Length": str(len(body))}, body)


def _fetch(url, method="GET", headers=None, body=None, client="203.0.113.5"):
    status, h, it = proxy.open_upstream(url, method, headers or {}, body, client)
    return status, h, b"".join(it)


def _refused(fn, status):
    try:
        fn()
    except proxy.Refused as e:
        assert e.status == status, (e.status, status, e)
        return
    raise AssertionError("not refused")


def _reset():
    proxy.CLIENTS = proxy._Clients()
    with proxy._STREAMS_LOCK:
        proxy._STREAMS.clear()


def test_sent_target_is_checked_url():
    _reset()
    with _Fake(_ok()) as up:
        for given, sent in ALLOWED:
            st, _, data = _fetch(given)
            assert st == 200 and len(data) == 1000
            want = sent or given
            assert up.seen[-1]["url"] == want
            p = requests.Request("GET", want).prepare()
            assert up.seen[-1]["target"] == p.path_url          # the request line urllib3 gets
            from urllib3.util.url import _encode_target
            assert _encode_target(p.path_url) == p.path_url     # ... and sends unchanged
        n = len(up.seen)
        for u in BYPASSES:
            _refused(lambda: _fetch(u), 403)
        assert len(up.seen) == n                                # nothing went upstream


def test_redirects_rechecked():
    _reset()
    start = GH + "master/a.json"
    good = GH + "master/b.json"
    bad_locations = [
        "/hobuinc/usgs-lidar/../../evil/repo/main/evil.3mf",
        "../../../evil/repo/main/evil.3mf",                    # relative, resolves out of the prefix
        GH + "%2e%2e/%2e%2e/evil/repo/main/evil.3mf",
        GH + "..%2f..%2fevil.3mf",
        GH + "..\\..\\evil.3mf",
        "https://169.254.169.254/latest/meta-data/",
        "//evil.example/x",
        "http://raw.githubusercontent.com/hobuinc/usgs-lidar/x",
        "https://raw.githubusercontent.com/someone/else/main/x",
    ]
    for loc in bad_locations:
        for code in (301, 302, 303, 307, 308):
            def routes(req, loc=loc, code=code):
                return (code, {"Location": loc}, b"") if req.url == start else (200, {}, b"evil")
            with _Fake(routes) as up:
                _refused(lambda: _fetch(start), 502)
                assert [s["url"] for s in up.seen] == [start], (loc, up.seen)

    # allowed hops are followed (relative too), each hop re-checked; 303 turns a POST into a GET
    def chain(req):
        if req.url == start:
            return 302, {"Location": "b.json"}, b""
        if req.url == good:
            return 200, {"Content-Type": "application/json"}, b"{}"
        raise AssertionError(req.url)
    with _Fake(chain) as up:
        assert _fetch(start)[::2] == (200, b"{}")
        assert [s["url"] for s in up.seen] == [start, good]

    search = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
    page = "https://planetarycomputer.microsoft.com/api/stac/v1/search?token=next"
    def post(req):
        if req.method == "POST":
            return 303, {"Location": page}, b""
        return 200, {"Content-Type": "application/geo+json"}, b"{}"
    with _Fake(post) as up:
        assert _fetch(search, "POST", {"content-type": "application/json"}, b'{"q":1}')[0] == 200
        assert up.seen[1]["method"] == "GET" and up.seen[1]["body"] is None
        assert "Content-Type" not in up.seen[1]["headers"]
    # a POST may only be redirected (307/308) to another POST-allowed URL
    def post307(req):
        return (307, {"Location": "https://planetarycomputer.microsoft.com/api/data/v1/x"}, b"") \
            if req.url == search else (200, {}, b"")
    with _Fake(post307):
        _refused(lambda: _fetch(search, "POST", {}, b"{}"), 502)

    # loops end
    with _Fake(lambda req: (302, {"Location": good if req.url == start else start}, b"")) as up:
        _refused(lambda: _fetch(start), 502)
        assert len(up.seen) == 4


def test_response_headers():
    _reset()
    cases = [
        ({}, "application/octet-stream"),                                   # none: explicit
        ({"Content-Type": ""}, "application/octet-stream"),
        ({"Content-Type": "image/tiff"}, "image/tiff"),
        ({"Content-Type": "application/json"}, "application/json"),
        ({"Content-Type": "text/html; charset=utf-8"}, "text/plain; charset=utf-8"),
        ({"content-type": "TEXT/HTML"}, "text/plain; charset=utf-8"),        # lower-case key
        ({"Content-Type": "application/xhtml+xml"}, "text/plain; charset=utf-8"),
        ({"Content-Type": "application/javascript"}, "text/plain; charset=utf-8"),
        ({"Content-Type": "image/svg+xml"}, "application/octet-stream"),
        ({"Content-Type": "model/3mf"}, "application/octet-stream"),
        ({"Content-Type": "multipart/x-mixed-replace"}, "application/octet-stream"),
    ]
    for up_headers, want in cases:
        hdrs = dict(up_headers, **{"Set-Cookie": "a=b", "X-Other": "1", "Content-Range": "bytes 0-9/99",
                                   "ETag": '"e"'})
        with _Fake(lambda req, h=hdrs: (206, h, b"0123456789")):
            st, out, data = _fetch(GH + "x", headers={"Range": "bytes=0-9", "Cookie": "c=d",
                                                      "Authorization": "x", "accept-encoding": "br"})
        assert st == 206 and data == b"0123456789"
        assert all(k == k.lower() for k in out), out                       # no duplicate spellings
        assert out["content-type"] == want, (up_headers, out)
        assert out["content-security-policy"] == "default-src 'none'; sandbox"
        assert out["x-content-type-options"] == "nosniff"
        assert out["cross-origin-resource-policy"] == "same-origin"
        assert out["content-range"] == "bytes 0-9/99" and out["etag"] == '"e"'
        assert "set-cookie" not in out and "x-other" not in out


def test_request_headers_and_session():
    _reset()
    with tempfile.TemporaryDirectory() as d:
        netrc = Path(d) / "netrc"
        netrc.write_text("machine raw.githubusercontent.com login operator password secret\n")
        saved = {k: os.environ.get(k) for k in ("NETRC", "HTTPS_PROXY", "https_proxy", "ALL_PROXY")}
        os.environ.update({"NETRC": str(netrc), "HTTPS_PROXY": "http://127.0.0.1:9",
                           "https_proxy": "http://127.0.0.1:9", "ALL_PROXY": "http://127.0.0.1:9"})
        try:
            # sanity: an ordinary session would have used both
            assert requests.utils.get_netrc_auth(GH + "x") == ("operator", "secret")

            def routes(req):
                return 200, {"Content-Type": "application/json", "Set-Cookie": "sid=abc; Path=/"}, b"{}"
            plain = requests.Session()
            plain.mount("https://", FakeUpstream(routes))
            plain.get(GH + "x")
            assert plain.cookies.get("sid") == "abc"                   # the fake does set cookies

            with _Fake(routes) as up:
                s = proxy._session()
                assert s.trust_env is False
                _fetch(GH + "x", headers={"Range": "bytes=0-1023", "Cookie": "c=d", "Authorization": "x",
                                          "Host": "evil.example", "X-Forwarded-For": "1.2.3.4"})
                _fetch(GH + "y", client="198.51.100.7")
                assert len(s.cookies) == 0
                for seen in up.seen:
                    h = {k.lower(): v for k, v in seen["headers"].items()}
                    assert "authorization" not in h and "cookie" not in h, h
                    assert "host" not in h and "x-forwarded-for" not in h
                    assert h["user-agent"].startswith("MapToMesh/")
                    assert not seen["proxies"], seen["proxies"]
                h0 = {k.lower(): v for k, v in up.seen[0]["headers"].items()}
                assert h0["range"] == "bytes=0-1023" and h0["accept-encoding"] == "identity"
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


# --- slots, budgets, slow readers ---------------------------------------------------------------

def test_client_keys():
    k = proxy.client_key
    assert k("203.0.113.5") == "203.0.113.5" and k("203.0.113.6") != k("203.0.113.5")
    a = k("2001:db8:1:2:3:4:5:6")
    assert a == "2001:db8:1:2::/64"
    assert k("2001:db8:1:2:ffff:ffff:ffff:ffff") == a and k("2001:DB8:1:2::1") == a
    assert k("2001:db8:1:3::1") != a
    assert k("::ffff:203.0.113.5") == "203.0.113.5"
    assert k("fe80::1%eth0") == "fe80::/64"
    assert k("unknown") == "unknown" and k("") == "unknown"


def test_local_unlimited():
    """Requests from this machine (limits.LOCAL: tests) aren't limited or counted."""
    _reset()
    c, g = proxy.CLIENT_CONCURRENT, proxy.GLOBAL_CONCURRENT
    proxy.CLIENT_CONCURRENT, proxy.GLOBAL_CONCURRENT = 1, 1
    try:
        with _Fake(_ok(b"z" * 1000)):
            streams = [proxy.open_upstream(GH + f"l{i}", "GET", {}, None, proxy.LOCAL)[2] for i in range(5)]
            assert proxy.CLIENTS.total == 0 and not proxy.CLIENTS.active
            a = proxy.open_upstream(GH + "x", "GET", {}, None, "203.0.113.9")[2]    # a visitor still gets one
            assert len(b"".join(a)) == 1000
            for st in streams:
                assert len(b"".join(st)) == 1000
            assert proxy.CLIENTS.total == 0 and "local" not in proxy.CLIENTS.used
    finally:
        proxy.CLIENT_CONCURRENT, proxy.GLOBAL_CONCURRENT = c, g


def test_slots_per_client_and_global():
    _reset()
    c, g = proxy.CLIENT_CONCURRENT, proxy.GLOBAL_CONCURRENT
    proxy.CLIENT_CONCURRENT, proxy.GLOBAL_CONCURRENT = 2, 3
    try:
        with _Fake(_ok(b"y" * 300_000)):
            # one IPv6 /64 is one client, whichever addresses it uses
            a = proxy.open_upstream(GH + "a", "GET", {}, None, "2001:db8::1")[2]
            b = proxy.open_upstream(GH + "b", "GET", {}, None, "2001:db8::2")[2]
            _refused(lambda: proxy.open_upstream(GH + "c", "GET", {}, None, "2001:db8::ffff"), 429)
            c1 = proxy.open_upstream(GH + "c", "GET", {}, None, "2001:db8:0:1::1")[2]   # next /64
            _refused(lambda: proxy.open_upstream(GH + "d", "GET", {}, None, "203.0.113.9"), 503)
            assert proxy.CLIENTS.active == {"2001:db8::/64": 2, "2001:db8:0:1::/64": 1}
            next(a)
            a.close()                          # abandoned part-way: slot back at once
            del b                              # dropped without reading (client went away)
            gc.collect()
            assert proxy.CLIENTS.active == {"2001:db8:0:1::/64": 1}
            assert len(b"".join(c1)) == 300_000
            # refused requests don't keep slots either
            _refused(lambda: proxy.open_upstream(GH + "../x", "GET", {}, None, "2001:db8::1"), 403)
            _refused(lambda: proxy.open_upstream(search_url(), "POST", {}, b"x" * 70_000, "2001:db8::1"), 413)
    finally:
        proxy.CLIENT_CONCURRENT, proxy.GLOBAL_CONCURRENT = c, g


def search_url():
    return "https://planetarycomputer.microsoft.com/api/stac/v1/search"


def test_budget_counted_while_streaming():
    _reset()
    budget = proxy.CLIENT_BYTES_PER_HOUR
    proxy.CLIENT_BYTES_PER_HOUR = 200_000
    try:
        def routes(req):
            body = b"z" * (1_000_000 if req.url.endswith("big.tif") else 1000)
            return 200, {"Content-Type": "image/tiff"}, body
        with _Fake(routes):
            st, _, it = proxy.open_upstream(GH + "big.tif", "GET", {}, None, "203.0.113.5")
            got = 0
            try:
                for chunk in it:
                    got += len(chunk)
                raise AssertionError("not cut off")
            except proxy.StreamAborted:
                pass
            assert got <= 200_000, got                       # stopped as soon as it ran over
            assert not proxy.CLIENTS.active
            _refused(lambda: proxy.open_upstream(GH + "more.tif", "GET", {}, None, "203.0.113.5"), 429)
            assert _fetch(GH + "x", client="203.0.113.6")[0] == 200   # others unaffected
    finally:
        proxy.CLIENT_BYTES_PER_HOUR = budget


def test_budget_window():
    budget = proxy.CLIENT_BYTES_PER_HOUR
    proxy.CLIENT_BYTES_PER_HOUR = 1000
    try:
        cl = proxy._Clients()
        t = 10_000.0
        assert cl.add("k", 600, now=t)
        assert cl.add("k", 300, now=t + 30)                     # same minute bucket
        assert len(cl.used["k"]) == 1
        assert not cl.add("k", 200, now=t + 120)                # 1100 within the hour
        _refused(lambda: cl.enter("k", now=t + 200), 429)
        cl.enter("other", now=t + 200)
        cl.leave("other")
        cl.enter("k", now=t + 3601)                             # first bucket expired: 200 left
        cl.leave("k")
        assert cl.used_total["k"] == 200
        assert cl.add("k", 0, now=t + 3781) and cl.used_total["k"] == 0   # all expired
    finally:
        proxy.CLIENT_BYTES_PER_HOUR = budget


class _FakeR:
    def __init__(self, chunks):
        self.raw = types.SimpleNamespace(stream=lambda n, decode_content=False: iter(chunks))
        self.closed = False

    def close(self):
        self.closed = True


def test_slow_readers_reaped():
    _reset()
    t = time.monotonic()
    # idle: the reader took nothing for IDLE_SECONDS
    proxy.CLIENTS.enter("a")
    s = proxy._Stream(_FakeR([b"x" * 1000] * 10), "a", now=t)
    it = s
    next(it)
    assert proxy.reap(time.monotonic() + proxy.IDLE_SECONDS / 2) == 0
    assert proxy.reap(time.monotonic() + proxy.IDLE_SECONDS + 1) == 1
    assert s.r.closed and not proxy.CLIENTS.active and s not in proxy._STREAMS
    try:
        next(it)
        raise AssertionError("still streaming")
    except proxy.StreamAborted:
        pass

    # trickle: keeps taking a little, below MIN_BYTES_PER_SECOND over a window
    proxy.CLIENTS.enter("b")
    s = proxy._Stream(_FakeR([b"x" * 1000] * 1000), "b", now=t)
    now = t
    reaped = False
    for _ in range(200):
        now += proxy.IDLE_SECONDS / 2                          # never idle long enough on its own
        next(s)
        s.last = now
        if s.stale(now):
            reaped = True
            break
    assert reaped and now - t <= proxy.RATE_WINDOW + proxy.IDLE_SECONDS
    s.close("test")

    # a fast reader isn't touched, nor one waiting on a slow upstream
    proxy.CLIENTS.enter("c")
    s = proxy._Stream(_FakeR([b"x" * 65536] * 1000), "c", now=t)
    now = t
    for _ in range(50):
        now += 5
        for _ in range(4):
            next(s)
        s.last = now
        assert s.stale(now) is None
    s.reading = True
    assert s.stale(now + 10 * proxy.RATE_WINDOW) is None
    s.reading = False
    s.close("test")
    assert not proxy.CLIENTS.active and not proxy._STREAMS


def test_reaped_mid_read_is_not_a_complete_file():
    _reset()
    proxy.CLIENTS.enter("a")
    holder = {}

    def chunks():
        yield b"x"
        holder["s"].close("reader idle")    # closed by the reaper while the upstream read ran
        return

    s = proxy._Stream(_FakeR(chunks()), "a")
    holder["s"] = s
    next(s)
    try:
        next(s)
        raise AssertionError("looked complete")
    except proxy.StreamAborted:
        pass
    assert not proxy.CLIENTS.active


def test_response_size_cap():
    _reset()
    cap = proxy.MAX_RESPONSE_BYTES
    proxy.MAX_RESPONSE_BYTES = 100_000
    try:
        with _Fake(_ok(b"q" * 500_000)):                       # declared too large: refused up front
            _refused(lambda: _fetch(GH + "x"), 413)
        with _Fake(lambda req: (200, {"Content-Type": "image/tiff"}, b"q" * 500_000)):  # undeclared
            try:
                _fetch(GH + "x")
                raise AssertionError("not cut off")
            except proxy.StreamAborted:
                pass
    finally:
        proxy.MAX_RESPONSE_BYTES = cap


def test_stream_lifetime_capped():
    """However it is read - even a reader keeping just above the minimum rate, or a stream waiting
    on its upstream - no stream holds its slot past MAX_STREAM_SECONDS."""
    _reset()
    t = time.monotonic()
    proxy.CLIENTS.enter("a")
    s = proxy._Stream(_FakeR([b"x" * 1000] * 10), "a", now=t)
    assert s.stale(t + proxy.MAX_STREAM_SECONDS - 1) in (None, "reader idle")
    s.reading = True                                           # waiting on the upstream
    assert s.stale(t + proxy.MAX_STREAM_SECONDS - 1) is None
    assert s.stale(t + proxy.MAX_STREAM_SECONDS + 1) == "open too long"
    assert proxy.reap(t + proxy.MAX_STREAM_SECONDS + 1) == 1
    assert s.r.closed and not proxy.CLIENTS.active


def test_group_budgets_and_reserve():
    """An IPv6 /48 (256 /56s, 65,536 /64s) gets GROUP_FACTOR clients' worth, not one budget per
    /64; and a quarter of the global slots stays free for clients with no download yet."""
    _reset()
    saved = (proxy.CLIENT_CONCURRENT, proxy.GLOBAL_CONCURRENT, proxy.GROUP_FACTOR, proxy.CLIENT_BYTES_PER_HOUR)
    try:
        proxy.CLIENT_CONCURRENT, proxy.GLOBAL_CONCURRENT, proxy.GROUP_FACTOR = 1, 100, 2
        cl = proxy._Clients()
        cl.enter("2001:db8:0:1::/64")
        cl.enter("2001:db8:0:2::/64")
        _refused(lambda: cl.enter("2001:db8:0:3::/64"), 429)            # same /48: its share is used
        cl.enter("2001:db8:1:1::/64")                                     # another /48
        cl.leave("2001:db8:0:1::/64")
        cl.enter("2001:db8:0:3::/64")
        assert cl.active_groups == {"2001:db8::/48": 2, "2001:db8:1::/48": 1}
        # bytes: the /48's budget is shared by its /64s
        proxy.CLIENT_BYTES_PER_HOUR = 1000
        t = 50_000.0
        assert cl.add("2001:db8:5:1::/64", 900, now=t)
        assert cl.add("2001:db8:5:2::/64", 900, now=t)
        assert not cl.add("2001:db8:5:3::/64", 300, now=t)              # 2100 > 2 x 1000 for the /48
        _refused(lambda: cl.enter("2001:db8:5:4::/64", now=t), 429)
        cl.enter("198.51.100.1", now=t)                                   # IPv4: no group
        # reserve: with 4 slots, a client already streaming takes a third only while 2+ are free
        proxy.CLIENT_CONCURRENT, proxy.GLOBAL_CONCURRENT = 16, 4
        cl = proxy._Clients()
        cl.enter("a"); cl.enter("a"); cl.enter("a")                       # noqa: E702 - 3 of 4 in use
        _refused(lambda: cl.enter("a"), 429)                              # the last one is for newcomers
        cl.enter("b")
        _refused(lambda: cl.enter("c"), 503)
        try:
            cl.enter("c")
        except proxy.Refused as e:
            assert e.retry_after
    finally:
        proxy.CLIENT_CONCURRENT, proxy.GLOBAL_CONCURRENT, proxy.GROUP_FACTOR, proxy.CLIENT_BYTES_PER_HOUR = saved


def test_request_rate():
    """Requests per client per hour: any method, and HEAD / POST on a tighter allowance."""
    saved = proxy.REQUESTS, proxy.HEAD_POST
    try:
        proxy.REQUESTS, proxy.HEAD_POST = proxy.RateLimit(5), proxy.RateLimit(2)
        assert [proxy.rate_wait("203.0.113.1", "HEAD") > 0 for _ in range(3)] == [False, False, True]
        assert [proxy.rate_wait("203.0.113.1", "GET") > 0 for _ in range(3)] == [False, False, True]
        assert proxy.rate_wait("203.0.113.2", "POST") == 0
        assert all(proxy.rate_wait(proxy.LOCAL, "HEAD") == 0 for _ in range(10))
        # the bytes-per-minute sweep doesn't run on every chunk any more: 20,000 keys stay cheap
        cl = proxy._Clients()
        t = 1000.0
        for i in range(20_000):
            cl.add(f"10.0.{i >> 8}.{i & 255}", 10, now=t)
        t0 = time.perf_counter()
        for _ in range(2000):
            cl.add("10.9.9.9", 10, now=t + 1)
        assert time.perf_counter() - t0 < 1.0
        cl.add("10.9.9.9", 10, now=t + 3700)                              # an hour on: swept
        assert len(cl.used) == 1
    finally:
        proxy.REQUESTS, proxy.HEAD_POST = saved


if __name__ == "__main__":
    for fn in (test_bypasses_refused, test_allowed_and_sent_unchanged, test_extra_rules,
               test_sent_target_is_checked_url, test_redirects_rechecked, test_response_headers,
               test_request_headers_and_session, test_client_keys, test_local_unlimited,
               test_slots_per_client_and_global,
               test_budget_counted_while_streaming, test_budget_window, test_slow_readers_reaped,
               test_reaped_mid_read_is_not_a_complete_file, test_response_size_cap,
               test_stream_lifetime_capped, test_group_budgets_and_reserve, test_request_rate):
        fn()
        print(f"{fn.__name__}: ok")
    print("all passed")
