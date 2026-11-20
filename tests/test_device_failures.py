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
    per = df.PER_HOUR
    df.PER_HOUR = 2
    df._TIMES.clear()
    assert not df.rate_limited("a") and not df.rate_limited("a") and df.rate_limited("a")
    df.PER_HOUR = per


if __name__ == "__main__":
    for fn in (test_store, test_rate_limit):
        fn()
        print(f"{fn.__name__}: ok")
    print("all passed")
