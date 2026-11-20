"""Follow-ups: LiDAR index limit, generic search errors, closed sqlite connections, STAC cache cap.

Run: python tests/test_followups.py
"""
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from maptomesh import device_failures, pointcloud, reviews, shares, usgs3dep  # noqa: E402


def test_find_sources_passes_limit():
    seen = {}

    class Idx:
        def query(self, *bbox, limit=None):
            seen["limit"] = limit
            return []

    pointcloud.find_sources((-60.0, 45.0, -59.9, 45.1), Idx(), max_items=77)
    assert seen["limit"] == 77


def test_search_errors_are_generic():
    secret = "C:/secret/path?token=abc123"
    real = usgs3dep.tiles

    def boom(*a, **k):
        raise OSError(secret)

    usgs3dep.tiles = boom
    try:
        _, _, errors = pointcloud.find_sources((-77.0, 38.9, -76.9, 39.0), None, "planetary")
    finally:
        usgs3dep.tiles = real
    assert errors and all(secret not in e and "token" not in e for e in errors), errors
    assert "error id" in errors[0]


def _deletable(path):
    for suffix in ("", "-wal", "-shm"):
        if os.path.exists(path + suffix):
            os.remove(path + suffix)       # fails on Windows while a connection is open


def test_connections_closed():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "r.sqlite")
        st = reviews.ReviewStore(p)
        assert st.count_for_job("x") == 0 and st.recent() == []
        _deletable(p)
        p = os.path.join(d, "s.sqlite")
        st = shares.ShareStore(p)
        sid = st.add({"a": 1})
        assert st.get(sid) == {"a": 1}
        _deletable(p)
        p = os.path.join(d, "f.sqlite")
        st = device_failures.FailureStore(p)
        st.add(kind="error", message="m", trace="", stage="", options=None, env={}, user_agent="", app_version="")
        assert len(st.recent()) == 1
        _deletable(p)


def test_stac_cache_trim():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        now = time.time()
        for i in range(6):
            f = d / f"pc_{i}.json"
            f.write_text("x" * 1000)
            os.utime(f, (now - 100 + i, now - 100 + i))
        (d / "other.txt").write_text("keep")
        usgs3dep._trim_cache(d, keep_files=4, keep_mb=1)
        assert sorted(p.name for p in d.glob("pc_*.json")) == ["pc_2.json", "pc_3.json", "pc_4.json", "pc_5.json"]
        usgs3dep._trim_cache(d, keep_files=100, keep_mb=0.0025)     # 2.5 kB: two files fit
        assert sorted(p.name for p in d.glob("pc_*.json")) == ["pc_4.json", "pc_5.json"]
        assert (d / "other.txt").exists()


if __name__ == "__main__":
    for fn in (test_find_sources_passes_limit, test_search_errors_are_generic, test_connections_closed,
               test_stac_cache_trim):
        fn()
        print(f"{fn.__name__}: ok")
