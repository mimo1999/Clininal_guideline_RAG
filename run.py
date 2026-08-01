#!/usr/bin/env python3
"""Starts the Clinical RAG webapp (chat UI + RAG backend, one process).

    python run.py

Run this every time you want to use the system -- unlike setup.py (installs,
index restore, model downloads), this does no slow one-off work, so it's
meant to be fast and repeatable. Run `python setup.py` first if you haven't
already (or after a `git pull` that changes requirements/corpus); this script
only checks for and warns about missing prerequisites, it doesn't fix them.
"""
from __future__ import annotations

import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
CHROMA_DIR = REPO_ROOT / "data_corpus" / "vector_store" / "chroma"
PROCESSED_DIR = REPO_ROOT / "data_corpus" / "processed"
WEBAPP_URL = "http://127.0.0.1:8080"


def _check_prerequisites() -> None:
    """Warn, don't block -- the webapp itself degrades gracefully (starts,
    just fails actual queries) if these are missing, and setup.py is the
    place that actually fixes them, not this script."""
    missing = []
    if not (CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir())):
        missing.append(f"vector index ({CHROMA_DIR})")
    if not (PROCESSED_DIR.exists() and any(PROCESSED_DIR.glob("*/chunks.jsonl"))):
        missing.append(f"chunk/router records ({PROCESSED_DIR})")
    if missing:
        print(f"WARNING: {' and '.join(missing)} not found -- retrieval will fail until you "
              f"run `python setup.py` first. Starting anyway.\n")


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    _check_prerequisites()

    proc = subprocess.Popen([sys.executable, "-m", "webapp.run_webapp"], cwd=REPO_ROOT)

    print(f"Waiting for {WEBAPP_URL} to come up (this includes a model warm-up query, "
          f"can take a minute)...")
    ready = False
    for _ in range(60):
        if proc.poll() is not None:
            print(f"\nWebapp process exited early (code {proc.returncode}) -- see output above.")
            return
        try:
            urllib.request.urlopen(WEBAPP_URL, timeout=2)
            ready = True
            break
        except (urllib.error.URLError, ConnectionError, OSError):
            time.sleep(2)

    if ready:
        print(f"\nReady -- open {WEBAPP_URL} in your browser.\n(Ctrl+C here stops the server.)")
    else:
        print(f"\nWARNING: {WEBAPP_URL} didn't respond within the timeout -- it may still be "
              f"starting up (check the log output above). It's still running; try the URL "
              f"in a browser directly.")

    try:
        proc.wait()
    except KeyboardInterrupt:
        print("\nStopping...")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
