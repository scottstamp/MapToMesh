"""Script-injection hardening of the web page (web/, maptomesh/server.py): the Content-Security-Policy
on the page, the build worker and the API; the hashes of the page's inline scripts; Subresource
Integrity on third-party files; markup built only through escaping templates in web/app.js.
Offline (in-process ASGI calls, static checks of the files).

    .venv/Scripts/python tests/test_web_security.py
"""

import base64
import hashlib
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from test_server_security import call, server  # noqa: E402  (sets up the temporary cache / data dirs)

WEB = HERE.parent / "web"


def csp(header: str) -> dict[str, list[str]]:
    out = {}
    for part in header.split(";"):
        words = part.split()
        if words:
            out[words[0]] = words[1:]
    return out


def inline_scripts(page: str) -> list[str]:
    return re.findall(r"<script\b(?![^>]*\bsrc\s*=)[^>]*>(.*?)</script\s*>", page, re.S | re.I)


def sha256(text: str) -> str:
    return f"'sha256-{base64.b64encode(hashlib.sha256(text.encode('utf-8')).digest()).decode()}'"


# ---------------------------------------------------------------- headers

def test_page_csp():
    st, h, body = call("GET", "/")
    assert st == 200
    p = csp(h["content-security-policy"])
    page = body.decode("utf-8")
    scripts = inline_scripts(page)
    assert len(scripts) == 2                                     # colour theme, import map
    for s in scripts:                                            # every inline script allowed by its hash
        assert sha256(s) in p["script-src"], s[:60]
    assert not any(v in p["script-src"] for v in ("'unsafe-inline'", "'unsafe-eval'", "https:", "*", "blob:", "data:"))
    assert set(p["script-src"]) - {sha256(s) for s in scripts} == {
        "'self'", "https://unpkg.com/leaflet@1.9.4/dist/leaflet.js", "https://unpkg.com/three@0.170.0/"}
    assert p["style-src"] == ["'self'", "https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"]
    assert p["object-src"] == ["'none'"] and p["base-uri"] == ["'self'"] and p["form-action"] == ["'self'"]
    assert p["frame-ancestors"] == ["'none'"]
    assert "https://tile.openstreetmap.org" in p["img-src"] and "blob:" in p["img-src"]
    assert p["connect-src"] == ["'self'", "https:", "blob:"]
    assert p["worker-src"] == ["'self'", "blob:"]
    # the same policy for the page's file under /static
    st, h2, _ = call("GET", "/static/index.html")
    assert st == 200 and h2["content-security-policy"] == h["content-security-policy"]


def test_worker_and_static_csp():
    st, h, _ = call("GET", "/static/browser-worker.js")
    assert st == 200
    p = csp(h["content-security-policy"])
    assert p["default-src"] == ["'none'"]
    assert p["script-src"] == ["'self'", "blob:", "'wasm-unsafe-eval'"]       # no 'unsafe-eval'
    assert p["connect-src"] == ["'self'", "https:"]
    st, h, _ = call("GET", "/static/app.js")
    assert st == 200 and "content-security-policy" in h and h["x-content-type-options"] == "nosniff"
    st, h, _ = call("GET", "/static/browser.html")                            # the test page: no inline code
    assert st == 200 and "'unsafe-inline'" not in h["content-security-policy"]
    assert csp(h["content-security-policy"])["script-src"][0] == "'self'"


def test_api_csp_and_proxy_kept():
    st, h, _ = call("GET", "/api/status")
    assert st == 200 and h["content-security-policy"] == server.API_CSP
    # a route's own policy wins (the data proxy's sandbox)
    from fastapi.responses import Response

    def own():
        return Response("x", headers={"Content-Security-Policy": "default-src 'none'; sandbox"})
    server.app.add_api_route("/api/_test_own_csp", own, methods=["GET"])
    st, h, _ = call("GET", "/api/_test_own_csp")
    assert st == 200 and h["content-security-policy"] == "default-src 'none'; sandbox"


def test_500_has_headers():
    def crash():
        raise KeyError("boom")
    server.app.add_api_route("/api/_test_crash_headers", crash, methods=["GET"])
    st, h, b = call("GET", "/api/_test_crash_headers")
    assert st == 500 and b"error id" in b
    assert h["x-content-type-options"] == "nosniff" and h["x-frame-options"] == "DENY"
    assert h["referrer-policy"] == "strict-origin-when-cross-origin"
    assert h["content-security-policy"] == server.API_CSP


def test_413_has_headers():
    st, h, _ = call("POST", "/api/shares", headers={"content-type": "application/json"}, body=b"{" + b" " * 200_000 + b"}")
    assert st == 413 and h["x-frame-options"] == "DENY" and "content-security-policy" in h


# ---------------------------------------------------------------- the files

def test_third_party_pinned():
    page = (WEB / "index.html").read_text(encoding="utf-8")
    # every external script / stylesheet carries a sha384 integrity hash
    tags = re.findall(r'<script\b[^>]*\bsrc="https?://[^>]*>|<link\b[^>]*rel="stylesheet"[^>]*href="https?://[^>]*>', page)
    assert len(tags) == 2, tags                                  # Leaflet's script and stylesheet
    for tag in tags:
        assert re.search(r'integrity="sha384-[A-Za-z0-9+/=]{64}"', tag), tag
        assert 'crossorigin="anonymous"' in tag, tag
    # the import map pins each three.js module app.js imports
    imap = json.loads(next(s for s in inline_scripts(page) if '"imports"' in s))
    app = (WEB / "app.js").read_text(encoding="utf-8")
    for spec in re.findall(r"^import .* from '(three[^']*)';", app, re.M):
        url = imap["imports"].get(spec) or next(imap["imports"][k] + spec[len(k):] for k in imap["imports"]
                                                  if k.endswith("/") and spec.startswith(k))
        assert re.fullmatch(r"sha384-[A-Za-z0-9+/=]{64}", imap["integrity"][url]), url
    # Pyodide: no static import from the CDN; each core file pinned
    worker = (WEB / "browser-worker.js").read_text(encoding="utf-8")
    assert not re.search(r"^\s*import\b[^;]*https?://", worker, re.M)
    for name in ("pyodide.mjs", "pyodide.asm.mjs", "pyodide.asm.wasm", "python_stdlib.zip", "pyodide-lock.json"):
        assert re.search(rf'"{re.escape(name)}": "sha384-[A-Za-z0-9+/=]{{64}}"', worker), name


def test_no_inline_code_or_styles():
    for f in WEB.glob("*.html"):
        page = f.read_text(encoding="utf-8")
        assert not re.search(r"\son[a-z]+\s*=", page), f"{f.name}: inline event handler"
        assert "javascript:" not in page.lower(), f.name
        assert not re.search(r"<style\b|\sstyle\s*=", page), f"{f.name}: inline style (blocked by the CSP)"
        if f.name != "index.html":
            assert not inline_scripts(page), f"{f.name}: inline script"


def test_markup_escaped():
    """web/app.js: markup goes into the page only through setHtml(el, html`...`) (each ${} escaped);
    innerHTML otherwise only cleared; no other HTML sinks; Leaflet tooltips get escaped text."""
    app = (WEB / "app.js").read_text(encoding="utf-8")
    code = re.sub(r"//[^\n]*", "", app)
    sinks = re.findall(r"\.innerHTML\s*=\s*([^;\n]*)", code)
    assert sinks.count("h.s") == 1, sinks                   # in setHtml itself
    assert all(s in ("h.s", "''") for s in sinks), sinks
    assert not re.search(r"insertAdjacentHTML|outerHTML|document\.write|\.srcdoc|new Function|\beval\(", code)
    uses = re.findall(r"(?<!function )setHtml\(([^,]+),\s*(\S{0,6})", code)
    assert len(uses) >= 8, uses
    for use in uses:
        assert use[1].startswith("html`"), use
    for m in re.finditer(r"bind(?:Tooltip|Popup)\(([^,]+),", code):
        arg = m.group(1).strip()
        assert arg.startswith("'") or arg.startswith("tooltipText("), arg
    assert "style=" not in re.sub(r"data-swatch", "", code)   # swatch colours go through the CSSOM
    for f in ("browser-worker.js", "opfs-cache.js"):
        assert "innerHTML" not in (WEB / f).read_text(encoding="utf-8"), f
    page_js = (WEB / "browser-page.js").read_text(encoding="utf-8")   # the test page
    for v in ('download="${esc(name)}"', "<td>${esc(k)}</td><td>${esc(v)}</td>", "Failed: ${esc(m.message)}"):
        assert v in page_js, v


TESTS = [test_page_csp, test_worker_and_static_csp, test_api_csp_and_proxy_kept, test_500_has_headers,
         test_413_has_headers, test_third_party_pinned, test_no_inline_code_or_styles, test_markup_escaped]


if __name__ == "__main__":
    os.chdir(HERE.parent)
    for fn in TESTS:
        fn()
        print(f"{fn.__name__}: ok")
    print("all passed")
