"""Hardening of the public API (maptomesh/server.py): input bounds, body caps, job-id checks,
per-client limits, cancel tokens, error text, hidden API docs and security headers. Offline: the
app is called in-process through ASGI (no network, no LiDAR index, temporary cache and data dirs).

    .venv/Scripts/python tests/test_server_security.py
"""

import asyncio
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

_TMP = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
os.environ["MAPTOMESH_CACHE"] = os.path.join(_TMP.name, "cache")
os.environ["MAPTOMESH_DATA_DIR"] = os.path.join(_TMP.name, "data")
os.environ["MAPTOMESH_LOCAL_UNLIMITED"] = "1"     # opt-in (off by default): requests from 127.0.0.1 are exempt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np  # noqa: E402

from maptomesh import config, estimate, limits, reviews, server, worker  # noqa: E402
from maptomesh.lidar_index import MAX_HITS, TileIndex  # noqa: E402
from maptomesh.options import Options  # noqa: E402


# ---------------------------------------------------------------- a minimal ASGI client

def call(method: str, path: str, query: str = "", headers: dict | None = None, body: bytes = b"",
         chunked: bool = False, client: str = "203.0.113.7", on_receive=None):
    """-> (status, headers (lower-case names), body). ``chunked``: no Content-Length, the body
    arrives in 16 kB pieces."""
    hdrs = [(k.lower().encode(), str(v).encode()) for k, v in (headers or {}).items()]
    if body and not chunked:
        hdrs.append((b"content-length", str(len(body)).encode()))
    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": method,
             "scheme": "http", "path": path, "raw_path": path.encode("utf-8", "surrogateescape"),
             "query_string": query.encode(), "root_path": "", "headers": hdrs,
             "client": (client, 50000), "server": ("testserver", 80)}
    pieces = [body[i:i + 16384] for i in range(0, len(body), 16384)] or [b""]
    out = {"status": None, "headers": {}, "body": b""}

    async def receive():
        if on_receive:
            on_receive()
        if pieces:
            p = pieces.pop(0)
            return {"type": "http.request", "body": p, "more_body": bool(pieces)}
        await asyncio.sleep(3600)
        return {"type": "http.disconnect"}

    async def send(msg):
        if msg["type"] == "http.response.start":
            out["status"] = msg["status"]
            out["headers"] = {k.decode().lower(): v.decode() for k, v in msg.get("headers", [])}
        elif msg["type"] == "http.response.body":
            out["body"] += msg.get("body", b"")

    try:
        asyncio.run(server.app(scope, receive, send))
    except Exception:
        if out["status"] is None:     # Starlette re-raises after its 500 response (for the server's log)
            raise
    return out["status"], out["headers"], out["body"]


def post_json(path: str, data, headers: dict | None = None, **kw):
    return call("POST", path, headers={"content-type": "application/json", **(headers or {})},
                body=json.dumps(data).encode(), **kw)


AREA = {"lat": 47.56, "lon": -52.71, "width_m": 2000, "height_m": 1500}


# ---------------------------------------------------------------- docs, headers

def test_docs_hidden():
    for p in ("/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect"):
        assert call("GET", p)[0] == 404, p


def test_security_headers():
    status, h, _ = call("GET", "/")
    assert status == 200
    assert h["x-content-type-options"] == "nosniff"
    assert h["referrer-policy"] == "strict-origin-when-cross-origin"
    assert h["x-frame-options"] == "DENY"
    assert h["cache-control"] == "no-cache"
    assert "frame-ancestors 'none'" in h["content-security-policy"]   # details: tests/test_web_security.py
    status, h, _ = call("GET", "/static/app.js")
    assert status == 200 and h["x-frame-options"] == "DENY" and h["cache-control"] == "no-cache"
    status, h, _ = call("GET", "/api/status")
    assert status == 200 and h["x-content-type-options"] == "nosniff"


# ---------------------------------------------------------------- request bodies

def test_body_limit():
    big = b"{" + b" " * 200_000 + b"}"
    # by Content-Length: refused before anything is read
    reads = []
    status, h, body = call("POST", "/api/shares", headers={"content-type": "application/json"}, body=big,
                           on_receive=lambda: reads.append(1))
    assert status == 413 and json.loads(body)["detail"] == "Request body too large" and not reads
    assert h["x-content-type-options"] == "nosniff"
    # chunked (no Content-Length): counted as it is read
    status, _, _ = call("POST", "/api/shares", headers={"content-type": "application/json"}, body=big, chunked=True)
    assert status == 413
    status, _, _ = call("POST", "/api/device-failures", headers={"content-type": "application/json"},
                        body=b"{" + b" " * 140_000 + b"}", chunked=True)
    assert status == 413
    assert call("POST", "/api/shares", headers={"content-length": "abc"})[0] == 413
    # within the limits: handled by the route (422: not valid settings)
    assert post_json("/api/shares", {"lat": 1})[0] == 422
    # a review may carry ~200 kB of job snapshot
    assert server.body_limit("/api/jobs/0123456789ab/review") >= 250_000
    status, _, _ = call("POST", "/api/jobs/0123456789ab/review", headers={"content-type": "application/json"},
                        body=b'{"rating": 9' + b" " * 200_000 + b"}", chunked=True)
    assert status == 422


def test_proxy_checks_url_before_body():
    reads = []
    status, _, _ = call("POST", "/api/proxy", query="url=https://evil.example.com/x",
                        headers={"content-type": "application/json"}, body=b"{}", on_receive=lambda: reads.append(1))
    assert status == 403 and not reads
    status, _, _ = call("GET", "/api/proxy", query="url=http://169.254.169.254/latest/meta-data/")
    assert status == 403
    # an allowlisted POST host, but a body over 64 kB
    url = "url=https://planetarycomputer.microsoft.com/api/stac/v1/search"
    status, _, _ = call("POST", "/api/proxy", query=url, headers={"content-type": "application/json"},
                        body=b"{" + b" " * 70_000 + b"}", chunked=True)
    assert status == 413
    status, _, _ = call("POST", "/api/proxy", query=url, headers={"content-type": "application/json"},
                        body=b"{" + b" " * 70_000 + b"}")
    assert status == 413


# ---------------------------------------------------------------- coverage

def test_coverage_bounds():
    for bad in ({**AREA, "lat": 95}, {**AREA, "lat": -90}, {**AREA, "lon": 181},
                {**AREA, "width_m": 1_500_000, "height_m": 1_500_000}, {**AREA, "width_m": 0},
                {**AREA, "grid_cells": 10_000}, {**AREA, "source": "x"}, {**AREA, "us_source": "evil"}):
        assert post_json("/api/coverage", bad)[0] == 422, bad
    # NaN / Infinity (Python's json writes them; the server must not accept them)
    for k in ("lat", "width_m"):
        status, _, _ = call("POST", "/api/coverage", headers={"content-type": "application/json"},
                            body=json.dumps({**AREA, k: float("nan")}).encode())
        assert status == 422, k
        status, _, _ = call("POST", "/api/coverage", headers={"content-type": "application/json"},
                            body=json.dumps({**AREA, k: float("inf")}).encode())
        assert status == 422, k
    # the area fields follow Options (types, bounds and defaults)
    for k in server.AREA_FIELDS:
        a, o = server.AreaIn.model_fields[k], Options.model_fields[k]
        assert a.annotation == o.annotation and a.metadata == o.metadata and a.default == o.default, k


def test_coverage_usgs_bounds():
    import maptomesh.usgs3dep as u3
    real = u3.footprints
    calls = []
    u3.footprints = lambda bbox, *a: calls.append(bbox) or {"type": "FeatureCollection", "features": []}
    try:
        q = lambda w, s, e, n: call("GET", "/api/coverage/usgs", query=f"w={w}&s={s}&e={e}&n={n}")  # noqa: E731
        assert q(-100, 40, -101, 41)[0] == 400                        # w > e: was a negative area
        assert q(-100, 41, -99, 40)[0] == 400                         # s > n
        assert q("nan", 40, -99, 41)[0] == 400
        assert q(-100, 40, "inf", 41)[0] == 400
        assert q(-100, 95, -99, 96)[0] == 400
        st, _, b = q(-460, 40, -459.5, 40.5)                          # a wrapped map view
        assert st == 200 and json.loads(b)["in_us"] is False
        st, _, b = q(-105, 39, -103, 41)                              # too large: no search
        assert st == 200 and json.loads(b).get("too_large")
        assert not calls
        st, _, b = q(-105, 39.5, -104.5, 40)
        assert st == 200 and calls == [(-105, 39.5, -104.5, 40)]
        u3.footprints = lambda *a: (_ for _ in ()).throw(OSError(r"C:\Users\someone\cache\stac\x.json"))
        st, _, b = q(-105, 39.5, -104.5, 40)
        assert st == 502 and b"Users" not in b and b"error id" in b
    finally:
        u3.footprints = real


def test_tile_query_capped():
    n = 50_000
    x = np.linspace(-80, -70, n)
    bbox = np.stack([x, np.full(n, 45.0), x + 0.01, np.full(n, 45.01)], axis=1)
    idx = TileIndex(bbox, np.zeros(n, dtype=np.int32), [f"t{i}.copc.laz" for i in range(n)], ["Proj_2020"])
    t0 = time.time()
    hits = idx.query(-180, -90, 180, 90)
    assert len(hits) == MAX_HITS and time.time() - t0 < 5
    assert len(idx.query(-180, -90, 180, 90, limit=7)) == 7
    assert len(idx.query(-75.001, 44, -74.999, 46)) > 0


def test_coverage_rate_limit():
    real = limits.COVERAGE_RATE
    limits.COVERAGE_RATE = limits.RateLimit(2)
    try:
        q = lambda c: call("GET", "/api/lidar/tiles", query="w=-80&s=45&e=-79&n=46", client=c)  # noqa: E731
        assert q("198.51.100.1")[0] == 503 and q("198.51.100.1")[0] == 503   # index not loaded here
        st, h, _ = q("198.51.100.1")
        assert st == 429 and int(h["retry-after"]) > 0
        assert q("198.51.100.2")[0] == 503
        assert call("GET", "/api/lidar/tiles", query="w=-79&s=45&e=-80&n=46")[0] == 400
    finally:
        limits.COVERAGE_RATE = real


# ---------------------------------------------------------------- job ids and files

def test_job_ids():
    assert server.valid_job_id("0123456789ab")
    for bad in ("0123456789ab\n", "0123456789AB", "0123456789a", "../0123456789", "x\\..\\0123456789ab",
                "C:\\Windows\\x", "\\\\host\\share", None, 5):
        assert not server.valid_job_id(bad), bad
    p = server.job_path("0123456789ab", "thumbnail.png")
    assert p.parent.name == "0123456789ab"
    for name in ("..", "../x", "a/b", "a\\b", "C:x", "\\\\h\\s"):
        try:
            server.job_path("0123456789ab", name)
            raise AssertionError(name)
        except server.HTTPException as e:
            assert e.status_code == 404


def test_thumbnail_traversal():
    good = "aaaaaaaaaaaa"
    d = config.OUTPUT_DIR / good
    d.mkdir(parents=True, exist_ok=True)
    (d / "thumbnail.png").write_bytes(b"\x89PNG fake")
    other = Path(_TMP.name) / "elsewhere"
    other.mkdir(exist_ok=True)
    (other / "thumbnail.png").write_bytes(b"secret")
    assert call("GET", f"/api/jobs/{good}/thumbnail.png")[0] == 200
    for jid in (f"x\\..\\{good}", "..\\..\\elsewhere", str(other).replace("/", "\\"), "\\\\127.0.0.1\\share",
                f"{good}\n", "C:"):
        st, _, body = call("GET", f"/api/jobs/{jid}/thumbnail.png")
        assert st == 404 and body != b"secret", jid
    for route in ("", "/model.3mf", "/stl.zip", "/base.3mf", "/preview.bin"):
        assert call("GET", f"/api/jobs/x\\..\\{good}{route}")[0] == 404, route
    assert post_json(f"/api/jobs/x\\..\\{good}/cancel", {})[0] == 404


# ---------------------------------------------------------------- clients, limits

class _Req:
    def __init__(self, host):
        self.client = type("C", (), {"host": host})() if host else None


def test_client_key():
    assert limits.client_key(_Req("198.51.100.4")) == "198.51.100.4"
    a = limits.client_key(_Req("2001:db8:1:2:aaaa::1"))
    assert a == "2001:db8:1:2::/64" and a == limits.client_key(_Req("2001:db8:1:2:ffff:1:2:3"))
    assert limits.client_key(_Req("2001:db8:1:3::1")) != a
    assert limits.client_key(_Req("::ffff:198.51.100.4")) == "198.51.100.4"
    assert limits.client_key(_Req(None)) == "unknown"


def test_rate_and_slots():
    r = limits.RateLimit(2)
    assert r.hit("a") == 0 and r.hit("a") == 0 and r.hit("a") > 0 and r.hit("b") == 0
    s = limits.Slots(3, per_client=1)
    assert s.enter("a") is None and s.enter("a") == "client"
    assert s.enter("b") is None and s.enter("c") is None and s.enter("d") == "busy"
    s.leave("a")
    assert s.enter("d") is None


def test_estimate_fails_fast():
    est_slots = estimate.SLOTS
    estimate.SLOTS = limits.Slots(2, per_client=1)
    body = Options(**AREA).model_dump()
    try:
        me = "198.51.100.9"
        estimate.SLOTS.enter(me)                       # this client's estimate is still running
        t0 = time.time()
        st, h, _ = post_json("/api/estimate", body, client=me)
        assert st == 429 and "retry-after" in h and time.time() - t0 < 3
        estimate.SLOTS.enter("someone else")           # all slots busy
        st, h, _ = post_json("/api/estimate", body, client="198.51.100.10")
        assert st == 503 and "retry-after" in h and time.time() - t0 < 3
    finally:
        estimate.SLOTS = est_slots


def _queue_jobs_without_running():
    real = server._run_job
    server._run_job = lambda job: None              # jobs stay "queued"
    return real


class _HReq:
    """A request with a peer address and headers (lower-case names, as Starlette's)."""
    def __init__(self, host, headers=None):
        self.client = type("C", (), {"host": host})() if host else None
        self.headers = {k.lower(): v for k, v in (headers or {}).items()}


def test_cloudflare_client_ip():
    cf = {"CF-Connecting-IP": "198.51.100.77"}
    # through a cloudflared tunnel (peer: this machine) or Cloudflare's edge (its ranges): the header
    assert limits.client_key(_HReq("127.0.0.1", cf)) == "198.51.100.77"
    assert limits.client_key(_HReq("::1", cf)) == "198.51.100.77"
    assert limits.client_key(_HReq("172.70.1.2", cf)) == "198.51.100.77"              # 172.64.0.0/13
    assert limits.client_key(_HReq("2606:4700:10::1", cf)) == "198.51.100.77"
    assert limits.client_key(_HReq("127.0.0.1", {"CF-Connecting-IP": "2001:db8:5:6::9"})) == "2001:db8:5:6::/64"
    # straight to the server from elsewhere: a forged header doesn't move the visitor's bucket
    assert limits.client_key(_HReq("203.0.113.50", cf)) == "203.0.113.50"
    assert limits.client_key(_HReq("203.0.113.50", {"X-Forwarded-For": "198.51.100.77"})) == "203.0.113.50"
    # this machine without the header: local (tests), exempt from limits
    assert limits.client_key(_HReq("127.0.0.1")) == limits.LOCAL
    assert limits.client_key(_HReq("::ffff:127.0.0.1")) == limits.LOCAL
    assert limits.client_key(_HReq("127.0.0.1", {"X-Forwarded-For": "198.51.100.77"})) == limits.LOCAL
    # a header that isn't an address never makes a request local
    assert limits.client_key(_HReq("127.0.0.1", {"CF-Connecting-IP": "nonsense"})) == "127.0.0.1"
    # MAPTOMESH_TRUSTED_PROXIES-style extra range
    saved = limits.TRUSTED_PROXIES
    try:
        import ipaddress
        limits.TRUSTED_PROXIES = saved + (ipaddress.ip_network("192.0.2.0/24"),)
        assert limits.client_key(_HReq("192.0.2.9", cf)) == "198.51.100.77"
    finally:
        limits.TRUSTED_PROXIES = saved
    # the switch (off unless MAPTOMESH_LOCAL_UNLIMITED=1: a local reverse proxy forwarding without
    # the header must not make its requests unlimited)
    saved = limits.LOCAL_UNLIMITED
    try:
        limits.LOCAL_UNLIMITED = False
        assert limits.client_key(_HReq("127.0.0.1")) == "127.0.0.1"
    finally:
        limits.LOCAL_UNLIMITED = saved
    import subprocess
    out = subprocess.run([sys.executable, "-c", "from maptomesh import limits; print(limits.LOCAL_UNLIMITED)"],
                         capture_output=True, text=True, cwd=str(Path(__file__).resolve().parent.parent),
                         env={k: v for k, v in os.environ.items() if k != "MAPTOMESH_LOCAL_UNLIMITED"})
    assert out.stdout.strip() == "False", out
    # an origin secret: the header counts only on requests that carry it (Cloudflare's ranges are
    # shared by every customer's zones and Workers)
    saved = limits.ORIGIN_SECRET
    try:
        limits.ORIGIN_SECRET = "s3cret-value"
        assert limits.client_key(_HReq("172.70.1.2", cf)) == "172.70.1.2"
        assert limits.client_key(_HReq("172.70.1.2", {**cf, "X-Origin-Secret": "wrong"})) == "172.70.1.2"
        assert limits.client_key(_HReq("172.70.1.2", {**cf, "X-Origin-Secret": "s3cret-value"})) == "198.51.100.77"
        assert limits.client_key(_HReq("127.0.0.1", {**cf, "X-Origin-Secret": "s3cret-value"})) == "198.51.100.77"
    finally:
        limits.ORIGIN_SECRET = saved


def test_local_requests_unlimited():
    r = limits.RateLimit(1)
    assert all(r.hit(limits.LOCAL) == 0 for _ in range(100))
    s = limits.Slots(1, per_client=1)
    assert all(s.enter(limits.LOCAL) is None for _ in range(10))
    assert s.enter("a") is None and s.enter("b") == "busy"      # local slots weren't counted
    for _ in range(10):
        s.leave(limits.LOCAL)
    assert s.enter("b") == "busy"
    # through the app: estimates / coverage from this machine never hit their limits
    for _ in range(5):
        st, _, _ = call("GET", "/api/coverage/usgs", query="w=-80&s=43&e=-79.9&n=43.1", client="127.0.0.1")
        assert st != 429, st


def test_jobs_per_client_and_cancel_token():
    real = _queue_jobs_without_running()
    server.JOBS.clear()
    body = Options(**AREA, grid_cells=300).model_dump()
    try:
        a = "198.51.100.20"
        r1 = post_json("/api/jobs", body, client=a)
        r2 = post_json("/api/jobs", body, client=a)
        assert r1[0] == 200 and r2[0] == 200
        d1 = json.loads(r1[2])
        assert server.valid_job_id(d1["id"]) and len(d1["cancel_token"]) >= 20
        st, h, b = post_json("/api/jobs", body, client=a)
        assert st == 429 and json.loads(b)["error_kind"] == "busy" and "retry-after" in h
        assert post_json("/api/jobs", body, client="198.51.100.21")[0] == 200     # someone else
        # IPv6: the whole /64 is one client
        assert post_json("/api/jobs", body, client="2001:db8::1")[0] == 200
        assert post_json("/api/jobs", body, client="2001:db8::2")[0] == 200
        assert post_json("/api/jobs", body, client="2001:db8::3")[0] == 429
        # status never shows the token
        st, _, b = call("GET", f"/api/jobs/{d1['id']}")
        assert st == 200 and d1["cancel_token"] not in b.decode() and "cancel_token" not in json.loads(b)
        # cancelling needs the token
        path = f"/api/jobs/{d1['id']}/cancel"
        assert call("POST", path)[0] == 403
        assert call("POST", path, headers={"x-cancel-token": "nope"})[0] == 403
        assert call("POST", path, headers={"x-cancel-token": "\u00e9"})[0] == 403
        assert not server.JOBS[d1["id"]].cancel.is_set()
        assert call("POST", path, headers={"x-cancel-token": d1["cancel_token"]})[0] == 200
        assert server.JOBS[d1["id"]].cancel.is_set()
        # the whole queue: refused for everyone but this machine (tests run in parallel)
        limits_q = limits.MAX_QUEUE
        limits.MAX_QUEUE = len([j for j in server.JOBS.values() if j.state == "queued"])
        try:
            st, _, b = post_json("/api/jobs", body, client="198.51.100.99")
            assert st == 429 and "busy" in json.loads(b)["detail"]
            for _ in range(limits.JOBS_PER_CLIENT + 2):
                assert post_json("/api/jobs", body, client="127.0.0.1")[0] == 200
            # through Cloudflare (tunnel: peer is this machine, header set): limited again
            st, _, _ = post_json("/api/jobs", body, client="127.0.0.1", headers={"CF-Connecting-IP": "198.51.100.98"})
            assert st == 429
        finally:
            limits.MAX_QUEUE = limits_q
    finally:
        server._run_job = real
        server.JOBS.clear()


# ---------------------------------------------------------------- error text

def test_worker_error_classification():
    msg, public = worker.public_error(RuntimeError("No elevation data covers this area."))
    assert public and msg == "No elevation data covers this area."
    msg, public = worker.public_error(RuntimeError(r"can't open C:\Users\someone\cache\x.tif"))
    assert public and "someone" not in msg and "<file>" in msg
    msg, public = worker.public_error(KeyError("C:\\secret"))
    assert not public and "secret" not in msg
    assert worker.public_error(MemoryError())[1]
    assert worker.scrub("https://data.example.com/home/x.tif") == "https://data.example.com/home/x.tif"
    assert worker.scrub("x" * 1000, 50).endswith("...") and len(worker.scrub("x" * 1000, 50)) == 50


def test_job_failure_hides_internals():
    job = server.Job("bbbbbbbbbbbb", Options(**AREA, grid_cells=300))
    real = server.AOI
    server.AOI = lambda *a: (_ for _ in ()).throw(KeyError(r"C:\Users\someone\secret.py"))
    try:
        server._run_job(job)
    finally:
        server.AOI = real
    assert job.state == "error" and "error id" in job.error
    text = json.dumps(job.public())
    assert "Traceback" not in text and "someone" not in text and "KeyError" not in text
    # a stop with text written for users keeps it (cleaned of paths)
    job2 = server.Job("cccccccccccc", Options(**AREA, grid_cells=300))
    job2.set(0.1, r"Reading C:\Users\someone\cache\tile.tif")
    assert "someone" not in job2.message and "someone" not in "".join(job2.log)


def test_value_error_handler():
    def boom():
        raise ValueError(r"bad value in C:\Users\someone\x")
    server.app.add_api_route("/api/_test_boom", boom, methods=["GET"])
    st, _, b = call("GET", "/api/_test_boom")
    assert st == 400 and b"someone" not in b and b"error id" in b

    def crash():
        raise KeyError(r"C:\Users\someone\y")
    server.app.add_api_route("/api/_test_crash", crash, methods=["GET"])
    st, h, b = call("GET", "/api/_test_crash")
    assert st == 500 and b"someone" not in b and b"error id" in b and h["content-type"] == "application/json"


# ---------------------------------------------------------------- reviews

def test_reviews_budget():
    keep, mx = reviews.KEEP, reviews.MAX_MB
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        st = reviews.ReviewStore(os.path.join(d, "r.sqlite"))
        kw = dict(job_state="done", rating=5, problems=[], comment="", wants_followup=False, contact_name=None,
                  contact_email=None, options={}, snapshot_source="client", app_version="t", user_agent="")
        def refused(**k) -> bool:
            try:
                st.add(**{**dict(job_id="ffffffffffff", result=None), **k}, **kw)
                return False
            except reviews.StoreFull:
                return True

        try:
            # full by rows: new reviews are refused, stored ones are never deleted to make room
            reviews.KEEP = 5
            ids = [st.add(job_id=f"{i:012x}", result=None, **kw) for i in range(5)]
            assert refused()
            assert [r["id"] for r in st.recent(100)] == ids[::-1]
            # full by size
            reviews.KEEP, reviews.MAX_MB = 10_000, 0.5
            n = 0
            for i in range(20):
                if refused(job_id=f"{i:012x}", result={"blob": "x" * 100_000}):
                    break
                n += 1
            assert 1 <= n <= 5, n
            rows = st.recent(1000)
            assert rows[-1]["id"] == ids[0] and len(rows) == 5 + n     # the oldest still there
            assert not refused(result={"small": 1})                     # a small one still fits
            size = os.path.getsize(os.path.join(d, "r.sqlite"))
            assert size < 3e6, size
            # a new store (restart) sums what's stored before taking more
            st = reviews.ReviewStore(os.path.join(d, "r.sqlite"))
            assert refused(result={"blob": "x" * 100_000})
        finally:
            reviews.KEEP, reviews.MAX_MB = keep, mx


def test_review_endpoint():
    server.REVIEW_RATE.clear()
    snap = {"state": "done", "options": Options(**AREA).model_dump(), "result": {"stats": {}}, "log": []}
    ok = {"rating": 4, "snapshot": snap}
    assert post_json("/api/jobs/0123456789ab\n/review", ok)[0] == 404
    assert post_json("/api/jobs/not-a-job/review", ok)[0] == 404
    assert post_json("/api/jobs/0123456789ab/review", {"rating": 4})[0] == 404     # unknown, no snapshot
    st, _, b = post_json("/api/jobs/0123456789ab/review", {"rating": 4, "snapshot": {"options": {"lat": 1}}})
    assert st == 400 and b"lat" not in b
    st, _, b = post_json("/api/jobs/0123456789ab/review", ok)
    assert st == 200 and json.loads(b)["ok"]
    # a server job: the server's own record is used, whatever the browser sends
    job = server.Job("dddddddddddd", Options(**AREA), state="done", result={"stats": {}})
    server.JOBS[job.id] = job
    try:
        st, _, b = post_json(f"/api/jobs/{job.id}/review", {"rating": 2, "snapshot": {"options": {"lat": 1}}})
        assert st == 200
        row = reviews.store().recent(1)[0]
        assert row["job_id"] == job.id and row["snapshot_source"] == "server"
    finally:
        server.JOBS.pop(job.id, None)


def test_shares_and_failures_still_work():
    st, _, b = post_json("/api/shares", Options(**AREA).model_dump(), client="2001:db8:5::1")
    assert st == 200
    sid = json.loads(b)["id"]
    st, _, b = call("GET", f"/api/shares/{sid}")
    assert st == 200 and json.loads(b)["options"]["lat"] == AREA["lat"]
    assert call("GET", "/api/shares/..%5C..%5Cx")[0] == 404
    st, _, b = post_json("/api/device-failures", {"kind": "memory", "message": "m", "env": {"x": 1}})
    assert st == 200 and json.loads(b)["ok"]


# ---------------------------------------------------------------- data proxy: threads, navigations, rate

async def acall(method: str, path: str, query: str = "", headers: dict | None = None,
                client: str = "203.0.113.7"):
    """``call`` as a coroutine, for many requests at once on one event loop (as uvicorn serves)."""
    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": method,
             "scheme": "http", "path": path, "raw_path": path.encode(), "query_string": query.encode(),
             "root_path": "", "headers": [(k.lower().encode(), str(v).encode()) for k, v in (headers or {}).items()],
             "client": (client, 50000), "server": ("testserver", 80)}
    out = {"status": None, "body": b""}
    sent = []

    async def receive():
        if not sent:
            sent.append(1)
            return {"type": "http.request", "body": b"", "more_body": False}
        await asyncio.sleep(3600)
        return {"type": "http.disconnect"}

    async def send(msg):
        if msg["type"] == "http.response.start":
            out["status"] = msg["status"]
        elif msg["type"] == "http.response.body":
            out["body"] += msg.get("body", b"")

    await server.app(scope, receive, send)
    return out["status"], out["body"]


PROXY_URL = "url=https://usgs.entwine.io/boundaries/resources.geojson"


def test_proxy_threads_isolated():
    """Slow upstreams hold proxy threads only: with 60 proxied requests stuck on their upstreams,
    the rest of the site still answers, at most PROXY_THREADS upstream calls block at once, and
    one client never holds more than PROXY_CLIENT_THREADS of them."""
    from maptomesh import proxy
    real = proxy.open_upstream
    release = threading.Event()
    lock = threading.Lock()
    state = {"now": 0, "peak": 0, "by": {}, "peak_by": 0}

    def slow(url, method, headers, body, client):
        with lock:
            state["now"] += 1
            state["by"][client] = state["by"].get(client, 0) + 1
            state["peak"] = max(state["peak"], state["now"])
            state["peak_by"] = max(state["peak_by"], state["by"][client])
        release.wait(20)
        with lock:
            state["now"] -= 1
            state["by"][client] -= 1
        raise proxy.Refused(418, "stub upstream")

    async def main():
        stuck = [asyncio.create_task(acall("GET", "/api/proxy", query=PROXY_URL, client=f"198.51.100.{i % 12}"))
                 for i in range(60)]
        await asyncio.sleep(1.0)
        t0 = time.time()
        st, _ = await asyncio.wait_for(acall("GET", "/api/status"), 10)
        took = time.time() - t0
        release.set()
        done = await asyncio.gather(*stuck)
        return st, took, done

    proxy.open_upstream = slow
    server._PROXY_LIMITER.clear()
    try:
        st, took, done = asyncio.run(main())
        assert st == 200 and took < 5, (st, took)
        assert all(s == 418 for s, _ in done), {s for s, _ in done}
        assert state["peak"] <= server.PROXY_THREADS, state
        assert state["peak_by"] <= server.PROXY_CLIENT_THREADS, state
        assert not server._PROXY_KEYS                                  # per-client limiters cleaned up
    finally:
        proxy.open_upstream = real
        release.set()
        server._PROXY_LIMITER.clear()


def test_proxy_navigation_and_rate():
    from maptomesh import proxy
    real, saved = proxy.open_upstream, (proxy.REQUESTS, proxy.HEAD_POST)

    def stub(*a):
        raise proxy.Refused(418, "stub upstream")

    proxy.open_upstream = stub
    try:
        # opened as a page, in a frame or an <embed>: never (only the build's own fetches use it)
        for dest in ("document", "iframe", "embed", "object", "frame"):
            st, _, _ = call("GET", "/api/proxy", query=PROXY_URL, headers={"Sec-Fetch-Dest": dest})
            assert st == 403, dest
        assert call("GET", "/api/proxy", query=PROXY_URL, headers={"Sec-Fetch-Dest": "empty"})[0] == 418
        # hourly request allowances, before anything is sent upstream
        proxy.REQUESTS, proxy.HEAD_POST = proxy.RateLimit(3), proxy.RateLimit(1)
        me = "198.51.100.150"
        assert call("HEAD", "/api/proxy", query=PROXY_URL, client=me)[0] == 418
        st, h, _ = call("HEAD", "/api/proxy", query=PROXY_URL, client=me)
        assert st == 429 and "retry-after" in h
        assert call("GET", "/api/proxy", query=PROXY_URL, client=me)[0] == 418
        st, h, _ = call("GET", "/api/proxy", query=PROXY_URL, client=me)
        assert st == 429 and int(h["retry-after"]) > 0
        assert call("GET", "/api/proxy", query=PROXY_URL, client="198.51.100.151")[0] == 418
    finally:
        proxy.open_upstream, (proxy.REQUESTS, proxy.HEAD_POST) = real, saved


def test_proxy_route_streams_and_releases():
    """Through the route: the body streams whole (read on the proxy's threads), HEAD frees its slot
    at once, and a visitor who leaves mid-download frees theirs too."""
    import requests

    from maptomesh import proxy
    from test_proxy import FakeUpstream
    body = bytes(range(256)) * 4096                                  # 1 MB
    huge = b"\0" * (48 << 20)
    session = requests.Session()
    session.trust_env = False
    session.mount("https://", FakeUpstream(lambda req: (200, {"Content-Type": "image/tiff",
                                                              "Content-Length": str(len(huge if "big" in req.url else body))},
                                                        huge if "big" in req.url else body)))
    real = proxy._session
    proxy._session = lambda: session
    proxy.CLIENTS = proxy._Clients()
    me = "198.51.100.160"
    try:
        st, h, b = call("GET", "/api/proxy", query=PROXY_URL, client=me)
        assert st == 200 and b == body and h["content-security-policy"] == "default-src 'none'; sandbox"
        st, _, b = call("HEAD", "/api/proxy", query=PROXY_URL, client=me)
        assert st == 200 and b == b""
        assert not proxy.CLIENTS.active and proxy.CLIENTS.total == 0
        assert proxy.CLIENTS.used_total[me] == len(body)

        async def leave_early():
            scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "GET",
                     "scheme": "http", "path": "/api/proxy", "raw_path": b"/api/proxy",
                     "query_string": b"url=https://usgs.entwine.io/boundaries/big.json", "root_path": "",
                     "headers": [], "client": (me, 50000), "server": ("testserver", 80)}
            got = {"n": 0}
            gone = asyncio.Event()

            async def receive():
                if got["n"] == 0 and not gone.is_set():
                    got["n"] = -1
                    return {"type": "http.request", "body": b"", "more_body": False}
                await gone.wait()
                return {"type": "http.disconnect"}

            async def send(msg):
                if msg["type"] == "http.response.body":
                    got["n"] = max(got["n"], 0) + len(msg.get("body", b""))
                    if got["n"] > 1 << 20:
                        gone.set()                                     # the visitor closes the tab

            await asyncio.wait_for(server.app(scope, receive, send), 20)
            return got["n"]

        n = asyncio.run(leave_early())
        assert n < len(huge), n
        time.sleep(0.2)
        assert not proxy.CLIENTS.active and proxy.CLIENTS.total == 0, proxy.CLIENTS.active
    finally:
        proxy._session = real
        proxy.CLIENTS = proxy._Clients()


# ---------------------------------------------------------------- limits: groups, reserve, memory

def test_group_limits_and_reserve():
    # an IPv6 /48 gets GROUP_FACTOR clients' worth, however many /64s it rotates through
    r = limits.RateLimit(2)
    allowed = sum(r.hit(f"2001:db8:0:{n:x}::/64") == 0 for n in range(10) for _ in range(2))
    assert allowed == int(2 * limits.GROUP_FACTOR), allowed
    assert r.hit("2001:db8:1::/64") == 0                       # the next /48
    s = limits.Slots(100, per_client=1)
    got = [s.enter(f"2001:db8:0:{n:x}::/64") for n in range(6)]
    assert got.count(None) == int(limits.GROUP_FACTOR) and got[-1] == "client"
    # a quarter of the slots stays for clients with none yet
    s = limits.Slots(4, per_client=2)
    assert s.enter("a") is None and s.enter("a") is None and s.enter("b") is None
    assert s.enter("b") == "client"                             # 3 of 4 in use: the last one is for newcomers
    assert s.enter("c") is None and s.enter("d") == "busy"
    s.leave("a")
    s.leave("a")
    assert s._active == {"b": 1, "c": 1} and s._n == 2


def test_rate_limit_memory():
    """Tables of many clients stay bounded and cheap: idle keys are swept on a timer (not scanned
    on every event), and an event costs the same with 65,536 keys as with ten."""
    r = limits.RateLimit(5)
    t = 1_000.0
    for i in range(65_536):
        r.hit(f"10.{i >> 16}.{(i >> 8) & 255}.{i & 255}", now=t)
    t0 = time.perf_counter()
    for i in range(5_000):
        r.hit(f"10.200.{(i >> 8) & 255}.{i & 255}", now=t + 1)
    assert time.perf_counter() - t0 < 1.0, time.perf_counter() - t0
    r.hit("10.255.255.255", now=t + 3700)                      # an hour on: swept
    assert r.keys() == 1, r.keys()
    small = limits.RateLimit(1)
    small.MAX_KEYS = 100
    for i in range(1_000):
        small.hit(f"192.0.2.{i % 250}-{i}", now=t)
    assert small.keys() <= 101


# ---------------------------------------------------------------- jobs: hourly limit, groups, reserve, retention

def test_job_rate_groups_and_reserve():
    real = _queue_jobs_without_running()
    server.JOBS.clear()
    body = Options(**AREA, grid_cells=300).model_dump()
    saved = limits.JOB_RATE, limits.MAX_QUEUE
    try:
        # hourly: finished jobs don't free the allowance
        limits.JOB_RATE = limits.RateLimit(3)
        me = "198.51.100.60"
        for _ in range(3):
            st, _, b = post_json("/api/jobs", body, client=me)
            assert st == 200
            server.JOBS[json.loads(b)["id"]].state = "done"
        st, h, b = post_json("/api/jobs", body, client=me)
        assert st == 429 and "last hour" in json.loads(b)["detail"] and int(h["retry-after"]) > 0
        assert post_json("/api/jobs", body, client="127.0.0.1")[0] == 200          # local: exempt
        server.JOBS.clear()
        # a /48: JOBS_PER_CLIENT x GROUP_FACTOR queued, however many /64s
        limits.JOB_RATE, limits.MAX_QUEUE = limits.RateLimit(1000), 100
        ok = sum(post_json("/api/jobs", body, client=f"2001:db8:7:{n:x}::1")[0] == 200
                 for n in range(10) for _ in range(limits.JOBS_PER_CLIENT))
        assert ok == int(limits.JOBS_PER_CLIENT * limits.GROUP_FACTOR), ok
        server.JOBS.clear()
        # reserve: a second job only while a quarter of the queue is free
        limits.MAX_QUEUE = 4
        assert post_json("/api/jobs", body, client="198.51.100.61")[0] == 200
        assert post_json("/api/jobs", body, client="198.51.100.61")[0] == 200
        assert post_json("/api/jobs", body, client="198.51.100.62")[0] == 200
        assert post_json("/api/jobs", body, client="198.51.100.62")[0] == 429       # 3 of 4: kept for newcomers
        assert post_json("/api/jobs", body, client="198.51.100.63")[0] == 200
        assert post_json("/api/jobs", body, client="198.51.100.64")[0] == 429       # full
    finally:
        limits.JOB_RATE, limits.MAX_QUEUE = saved
        server._run_job = real
        server.JOBS.clear()


def test_outputs_kept_for_a_while():
    """Others' jobs can't wipe a model finished a moment ago: beyond the newest KEEP_JOBS, files go
    only once older than KEEP_JOB_HOURS, or when over OUTPUT_MAX_GB."""
    import shutil
    saved = server.KEEP_JOBS, server.KEEP_JOB_HOURS, server.OUTPUT_MAX_GB
    root = config.OUTPUT_DIR
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    now = time.time()
    try:
        server.KEEP_JOBS, server.KEEP_JOB_HOURS, server.OUTPUT_MAX_GB = 3, 1.0, 5
        for i in range(8):
            d = root / f"{i:012x}"
            d.mkdir()
            (d / "model.3mf").write_bytes(b"m" * 1000)
            age = 60 * i if i < 5 else 7200 + i                   # 5 recent, 3 older than an hour
            os.utime(d, (now - age, now - age))
        server._prune_outputs(now)
        assert sorted(p.name for p in root.iterdir()) == [f"{i:012x}" for i in range(5)]
        server.OUTPUT_MAX_GB = 3500 / 1e9                          # over the size budget: oldest first
        server._prune_outputs(now)
        assert sorted(p.name for p in root.iterdir()) == [f"{i:012x}" for i in range(3)]
    finally:
        server.KEEP_JOBS, server.KEEP_JOB_HOURS, server.OUTPUT_MAX_GB = saved
        shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True)


# ---------------------------------------------------------------- reviews, device failures, USGS list

def test_review_snapshot_checked_and_trimmed():
    server.REVIEW_RATE.clear()
    opts = Options(**AREA).model_dump()
    jid = "abcabcabcabc"
    for bad in ({"log": {"a": 1}}, {"log": 5}, {"log": [1, 2]}, {"result": [1]}, {"result": "x"}, {"error": {"x": 1}}):
        st, _, b = post_json(f"/api/jobs/{jid}/review", {"rating": 3, "snapshot": {"options": opts, **bad}})
        assert st == 400, (bad, st, b)
    big = {"stats": {"scale": "1:1000"}, "data": {"sources": [{"label": "NRCan", "share": 1.0}]}, "notes": ["n"],
           "color_changes": [{"z_mm": i} for i in range(2000)], "tips": ["t" * 1000] * 20,
           "filaments": [{"slot": 1, "color": "x" * 20_000}]}
    st, _, _ = post_json(f"/api/jobs/{jid}/review", {"rating": 3, "snapshot": {
        "state": "done", "options": opts, "result": big, "log": ["l" * 1000] * 100, "error": "e" * 5000}})
    assert st == 200
    row = reviews.store().recent(1)[0]
    res = json.loads(row["result_json"])
    assert set(res["result"]) == {"stats", "data", "notes"}, set(res["result"])     # filaments too large
    assert len(row["result_json"]) < server.CLIENT_RESULT_MAX + 16_000
    assert len(res["log"]) == 40 and all(len(x) <= 300 for x in res["log"]) and len(res["error"]) == 2000
    # everyone's reviews per hour
    saved = server.REVIEW_GLOBAL
    try:
        server.REVIEW_GLOBAL = limits.RateLimit(1, group_factor=0)
        ok = {"rating": 3, "snapshot": {"options": opts}}
        assert post_json("/api/jobs/aaaaaaaaaaa1/review", ok, client="198.51.100.70")[0] == 200
        assert post_json("/api/jobs/aaaaaaaaaaa2/review", ok, client="198.51.100.71")[0] == 429
    finally:
        server.REVIEW_GLOBAL = saved
    # a full store: refused with 503, nothing deleted
    keep = reviews.KEEP
    st_ = reviews.store()
    st_.forget()
    n = len(st_.recent(10_000))
    try:
        reviews.KEEP = n
        server.REVIEW_RATE.clear()
        st, _, _ = post_json("/api/jobs/aaaaaaaaaaa3/review", {"rating": 3, "snapshot": {"options": opts}})
        assert st == 503 and len(st_.recent(10_000)) == n
    finally:
        reviews.KEEP = keep
        st_.forget()


def test_device_failure_env_typed():
    from maptomesh import device_failures as df
    df._LIMIT.clear()
    good = {"memory_mb": 2048, "device_memory_gb": 8, "cores": 8, "phone": False, "secure": True, "seconds": 3.5,
            "python": {"ready": True, "jspi": True, "native": True, "pyodide": "0.29.0", "extra": "x"},
            "log": ["a", "b"] * 20 + ["x" * 2000], "unknown": "dropped"}
    st, _, b = post_json("/api/device-failures", {"kind": "error", "message": "m", "env": good})
    assert st == 200
    env = json.loads(df.store().recent(1)[0]["env_json"])
    assert env["memory_mb"] == 2048 and env["python"]["pyodide"] == "0.29.0"
    assert "unknown" not in env and "extra" not in env["python"]
    assert len(env["log"]) == 30 and env["log"][-1] == "x" * 500           # long logs trimmed, not dropped
    for bad in ({"memory_mb": "\u001b]0;pwned\u0007\u001b[2J"}, {"python": "x"}, {"log": "x"}, {"cores": "\u001b[2J"}):
        st, _, _ = post_json("/api/device-failures", {"kind": "error", "message": "m", "env": bad})
        assert st == 200
        assert json.loads(df.store().recent(1)[0]["env_json"]) == {"invalid": True}, bad


def test_usgs_projects_never_wait_on_a_download():
    from maptomesh import usgs_ept
    idx = usgs_ept.INDEX
    saved = idx._projects, idx._loading
    usgs_ept.OVERLAY_PATH.unlink(missing_ok=True)
    try:
        idx._projects, idx._loading = None, True              # another request is downloading the list
        t0 = time.time()
        st, h, _ = call("GET", "/api/projects/usgs")
        assert st == 503 and "retry-after" in h and time.time() - t0 < 2
    finally:
        idx._projects, idx._loading = saved


TESTS = [test_docs_hidden, test_security_headers, test_body_limit, test_proxy_checks_url_before_body,
         test_coverage_bounds, test_coverage_usgs_bounds, test_tile_query_capped, test_coverage_rate_limit,
         test_job_ids, test_thumbnail_traversal, test_client_key, test_cloudflare_client_ip,
         test_local_requests_unlimited, test_rate_and_slots, test_estimate_fails_fast,
         test_jobs_per_client_and_cancel_token, test_worker_error_classification, test_job_failure_hides_internals,
         test_value_error_handler, test_reviews_budget, test_review_endpoint, test_shares_and_failures_still_work,
         test_proxy_threads_isolated, test_proxy_navigation_and_rate, test_proxy_route_streams_and_releases,
         test_group_limits_and_reserve,
         test_rate_limit_memory, test_job_rate_groups_and_reserve, test_outputs_kept_for_a_while,
         test_review_snapshot_checked_and_trimmed, test_device_failure_env_typed,
         test_usgs_projects_never_wait_on_a_download]


if __name__ == "__main__":
    for fn in TESTS:
        fn()
        print(f"{fn.__name__}: ok")
    print("all passed")
