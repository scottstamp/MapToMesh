"""Start the MapToMesh web app:  python run.py [--port 8000] [--no-browser] [--local-unlimited]

Served through Cloudflare: per-client limits count against the visitor address Cloudflare sends
in ``CF-Connecting-IP``, believed only from Cloudflare's address ranges or this machine (a
cloudflared tunnel); see ``maptomesh.limits.client_key``. Requests from this machine without that
header count against 127.0.0.1 like any visitor's, unless ``--local-unlimited`` (or
MAPTOMESH_LOCAL_UNLIMITED=1) exempts them from every limit: for development and the integration
tests, never behind a local reverse proxy (everything it forwarded would be unlimited). uvicorn's
own proxy-header handling stays off: it would replace the peer address that check relies on with
an X-Forwarded-For value.
"""

import argparse
import logging
import os
import threading
import webbrowser

import uvicorn


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--local-unlimited", action="store_true",
                    help="no request limits for requests from this machine (development, tests)")
    args = ap.parse_args()
    if args.local_unlimited:
        os.environ["MAPTOMESH_LOCAL_UNLIMITED"] = "1"      # read when maptomesh.limits is imported
    url = f"http://{args.host}:{args.port}/"
    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    # errors with their id (the id is all users see of them; see maptomesh.server._internal_error)
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    print(f"MapToMesh running at {url}" + (" (requests from this machine: no limits)" if args.local_unlimited else ""))
    uvicorn.run("maptomesh.server:app", host=args.host, port=args.port, log_level="warning",
                proxy_headers=False, server_header=False)


if __name__ == "__main__":
    main()
