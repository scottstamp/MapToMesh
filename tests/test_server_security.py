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
import time
from pathlib import Path

_TMP = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
os.environ["MAPTOMESH_CACHE"] = os.path.join(_TMP.name, "cache")
os.environ["MAPTOMESH_DATA_DIR"] = os.path.join(_TMP.name, "data")
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


def post_json(path: str, data, **kw):
    return call("POST", path, headers={"content-type": "application/json"},
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
        # the whole queue: refused for everyone
        limits_q = limits.MAX_QUEUE
        limits.MAX_QUEUE = len([j for j in server.JOBS.values() if j.state == "queued"])
        try:
            st, _, b = post_json("/api/jobs", body, client="198.51.100.99")
            assert st == 429 and "busy" in json.loads(b)["detail"]
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
        try:
            reviews.KEEP = 5
            ids = [st.add(job_id=f"{i:012x}", result=None, **kw) for i in range(8)]
            assert [r["id"] for r in st.recent(100)] == ids[::-1][:5]
            reviews.KEEP, reviews.MAX_MB = 10_000, 0.5
            for i in range(20):
                st.add(job_id=f"{i:012x}", result={"blob": "x" * 100_000}, **kw)
            rows = st.recent(1000)
            assert 1 <= len(rows) < 10
            assert rows[0]["job_id"] == f"{19:012x}"                 # the newest kept
            size = os.path.getsize(os.path.join(d, "r.sqlite"))
            assert size < 3e6, size
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


TESTS = [test_docs_hidden, test_security_headers, test_body_limit, test_proxy_checks_url_before_body,
         test_coverage_bounds, test_coverage_usgs_bounds, test_tile_query_capped, test_coverage_rate_limit,
         test_job_ids, test_thumbnail_traversal, test_client_key, test_rate_and_slots, test_estimate_fails_fast,
         test_jobs_per_client_and_cancel_token, test_worker_error_classification, test_job_failure_hides_internals,
         test_value_error_handler, test_reviews_budget, test_review_endpoint, test_shares_and_failures_still_work]


if __name__ == "__main__":
    for fn in TESTS:
        fn()
        print(f"{fn.__name__}: ok")
    print("all passed")
