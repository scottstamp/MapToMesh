"""Device-failure reports (maptomesh/device_failures.py): stored, pruned, listed.

    .venv/Scripts/python tests/test_device_failures.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from maptomesh import device_failures as df
from maptomesh.options import Options


def test_store():
    keep = df.KEEP
    with tempfile.TemporaryDirectory() as d:
        st = df.FailureStore(os.path.join(d, "f.sqlite"))
        o = Options(lat=47.56, lon=-52.71, width_m=2000, height_m=1500).model_dump()
        a = st.add(kind="memory", message="Out of memory", trace="MemoryError", stage="Building the surface",
                   options=o, env={"phone": True, "memory_mb": 1200}, user_agent="UA", app_version="0.1")
        b = st.add(kind="startup", message="The build worker could not start", trace="", stage="", options=None,
                   env={}, user_agent="UA", app_version="0.1")
        rows = st.recent(5)
        assert [r["id"] for r in rows] == [b, a]
        assert rows[1]["kind"] == "memory" and '"lat": 47.56' in rows[1]["options_json"]
        df.KEEP = 2
        for _ in range(3):
            st.add(kind="error", message="x", trace="", stage="", options=None, env={}, user_agent="", app_version="")
        st._added = 0   # prune on the next add
        st.add(kind="error", message="last", trace="", stage="", options=None, env={}, user_agent="", app_version="")
        assert len(st.recent(10)) <= 3
    df.KEEP = keep


def test_rate_limit():
    real = df._LIMIT
    df._LIMIT = df.limits.RateLimit(2)
    assert not df.rate_limited("a") and not df.rate_limited("a") and df.rate_limited("a")
    df._LIMIT = real


def test_cli_safe_strips_control_bytes():
    from maptomesh.device_failures import _safe
    s = _safe("a\x1b[31mred\x9b\x07\nb")
    assert "\x1b" not in s and "\x9b" not in s and "\x07" not in s and "\n" not in s
    t = _safe("x\x1b]0;t\x07\n  at f()", keep_newlines=True)
    assert "\x1b" not in t and "\n  at f()" in t


def test_cli_env_values_escaped():
    """Stored env values are printed through _safe, and malformed ones (older rows, a report that
    bypassed the endpoint's checks) don't crash the listing."""
    import contextlib
    import io
    real = df._STORE
    with tempfile.TemporaryDirectory() as d:
        df._STORE = df.FailureStore(os.path.join(d, "f.sqlite"))
        try:
            for env in ({"memory_mb": "\x1b]0;pwned\x07\x1b[2J", "python": {"ready": "\x1b[31m"}},
                        {"python": "x"}, {"cores": ["\x1b"]}):
                df._STORE.add(kind="error", message="m", trace="", stage="", options={"lat": "x"}, env=env,
                              user_agent="", app_version="t")
            con = df._STORE._connect()                         # a row whose env isn't an object
            try:
                con.execute("UPDATE failures SET env_json = '[1]' WHERE id = 3")
            finally:
                con.close()
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                df._main(["10"])
            out = buf.getvalue()
            assert "\x1b" not in out and "\x07" not in out and out.count("#") == 3, out
        finally:
            df._STORE = real


def test_global_rate_limit():
    real = df._GLOBAL
    df._GLOBAL = df.limits.RateLimit(2, group_factor=0)
    try:
        assert [df.rate_limited(f"198.51.100.{i}") for i in range(4)] == [False, False, True, True]
        assert not df.rate_limited(df.limits.LOCAL)
    finally:
        df._GLOBAL = real


if __name__ == "__main__":
    for fn in (test_store, test_rate_limit, test_cli_safe_strips_control_bytes, test_cli_env_values_escaped,
               test_global_rate_limit):
        fn()
        print(f"{fn.__name__}: ok")
    print("all passed")
