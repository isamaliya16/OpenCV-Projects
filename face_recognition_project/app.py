"""Start the FaceKey face sign-in web app.

    python run_web.py                # http://127.0.0.1:5000
    python run_web.py --port 8080

The browser only allows camera access on https:// pages or on localhost, so
open http://localhost:5000 (not your LAN IP) when testing on your own machine.
For a public deployment see docs/SYSTEM_DESIGN.md (section 9).
"""
import argparse

from webapp import create_app

app = create_app()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    # threaded=True: one request per camera frame, served concurrently
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
