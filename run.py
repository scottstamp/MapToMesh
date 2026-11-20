"""Start the MapToMesh web app:  python run.py [--port 8000] [--no-browser]

Behind a reverse proxy (nginx, Caddy, a tunnel), set ``MAPTOMESH_FORWARDED_ALLOW_IPS`` to the
proxy's address(es) - comma-separated IPs or CIDR ranges - so per-client limits use the client
address from ``X-Forwarded-For``; only requests from those addresses may set it. The default,
``127.0.0.1``, suits a proxy on the same machine. Never use ``*`` on a server that is reachable
directly: anyone could then pick the address their requests are counted against.
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
    args = ap.parse_args()
    url = f"http://{args.host}:{args.port}/"
    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    # errors with their id (the id is all users see of them; see maptomesh.server._internal_error)
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    print(f"MapToMesh running at {url}")
    uvicorn.run("maptomesh.server:app", host=args.host, port=args.port, log_level="warning",
                proxy_headers=True,
                forwarded_allow_ips=os.environ.get("MAPTOMESH_FORWARDED_ALLOW_IPS", "127.0.0.1"),
                server_header=False)


if __name__ == "__main__":
    main()
