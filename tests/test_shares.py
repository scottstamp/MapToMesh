"""Share links (maptomesh/shares.py): settings stored under a short id, deduplicated, pruned.

    .venv/Scripts/python tests/test_shares.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from maptomesh import shares
from maptomesh.options import Options


def _opts(**kw) -> dict:
    return Options(lat=47.56, lon=-52.71, width_m=2000, height_m=1500, **kw).model_dump()


def test_add_get_dedupe():
    with tempfile.TemporaryDirectory() as d:
        st = shares.ShareStore(os.path.join(d, "s.sqlite"))
        a = st.add(_opts(name="Signal Hill"))
        assert shares.valid_id(a) and len(a) == 10
        assert st.add(_opts(name="Signal Hill")) == a            # same settings, same link
        b = st.add(_opts(name="Quidi Vidi"))
        assert b != a
        assert st.get(a)["name"] == "Signal Hill"
        assert Options(**st.get(b)).name == "Quidi Vidi"          # stored settings stay valid
        assert st.get("zzzzzzzzzz") is None
        assert st.get("../etc/passwd") is None and st.get("ABCDEFGHIJ") is None


def test_prune():
    keep = shares.KEEP
    shares.KEEP = 3
    with tempfile.TemporaryDirectory() as d:
        st = shares.ShareStore(os.path.join(d, "s.sqlite"))
        ids = [st.add(_opts(grid_cells=300 + 50 * i)) for i in range(4)]
        st.get(ids[0])                                            # opened: recently used
        with st._connect() as con:
            st._prune(con)
        assert st.get(ids[0]) is not None and st.get(ids[1], opened=False) is None
        assert all(st.get(i, opened=False) is not None for i in ids[2:])
    shares.KEEP = keep


def test_rate_limit():
    per = shares.PER_HOUR
    shares.PER_HOUR = 2
    shares._TIMES.clear()
    assert not shares.rate_limited("1.2.3.4") and not shares.rate_limited("1.2.3.4")
    assert shares.rate_limited("1.2.3.4") and not shares.rate_limited("5.6.7.8")
    shares.PER_HOUR = per


if __name__ == "__main__":
    for fn in (test_add_get_dedupe, test_prune, test_rate_limit):
        fn()
        print(f"{fn.__name__}: ok")
    print("all passed")
