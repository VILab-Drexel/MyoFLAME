"""Serve the interactive viewer (viewer/) on http://localhost:8000 and open it in a browser.

The viewer is a static page; browsers refuse to load its data files from file://,
so it needs a local web server. The FLAME part of its data is written from your own
FLAME 2020 file by demos/build_viewer_data.py; this script runs it first if needed.

    python demos/serve_viewer.py [--port 8000] [--no-browser] [--flame PATH]
"""
import argparse
import functools
import http.server
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VIEWER = ROOT / "viewer"
DEFAULT_FLAME = ROOT / "model" / "FLAME2020" / "generic_model.pkl"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--flame", type=Path, default=DEFAULT_FLAME, help="FLAME 2020 generic_model.pkl")
    args = parser.parse_args(argv)
    if not all((VIEWER / "data" / name).is_file() for name in ("flame.bin", "verify.bin")):
        if args.flame.is_file():
            import importlib.util
            spec = importlib.util.spec_from_file_location("build_viewer_data", ROOT / "demos" / "build_viewer_data.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            module.main(["--flame", str(args.flame)])
        else:
            print(f"FLAME 2020 not found at {args.flame}; the viewer will ask for its FLAME data. "
                  "Copy generic_model.pkl into model/FLAME2020/ (see the README) and run this again.")
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(VIEWER))
    with http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
        url = f"http://localhost:{args.port}/"
        print(f"MyoFLAME viewer at {url} (Ctrl+C to stop)")
        if not args.no_browser:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
