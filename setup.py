#!/usr/bin/env python3
"""Single entrypoint to set up and run the Clinical RAG system end-to-end.

    python setup.py

Steps, in order:
  1. Install Python dependencies (requirements.txt)              [--skip-install to skip]
  2. Unzip the pre-built Chroma vector index if not already present
     (data_corpus/vector_store/chroma_db.zip -> .../chroma/)
  3. Restore the per-document chunk/router records if not already present
     (data_corpus/processed.zip -> .../processed/) -- required at runtime
     (webapp + eval), NOT just for building the index: Chroma only stores
     minimal filterable metadata, so retrieval hydrates full chunk text and
     guideline/document router text straight from these files on every
     query. If the zip is missing too, falls back to running ingestion +
     chunking from data_corpus/pdf/ from scratch               [--skip-ingest to skip]
     (the BM25 sparse index builds itself automatically on first
     retrieval call -- no separate step needed)
  4. Best-effort start Langfuse tracing (OPTIONAL -- skipped
     cleanly, with a one-line note, if Podman/Docker or a
     langfuse_v2 checkout aren't available; never blocks the
     steps below)
  5. Only relevant if CLINICAL_RAG_GENERATOR_MODEL opts into an
     Ollama tag (the default generator/judge are transformers
     models that download automatically via huggingface_hub on
     first use, same as the retrieval models -- no separate step
     needed for those): ensures Ollama is installed (auto-installs
     it if missing) and pulls that tag. Warns rather than aborting
     on any failure.
  6. Start the webapp and print the localhost URL once it's ready

Re-running this script is safe and cheap: each step checks whether its
output already exists before doing any real work.
"""
from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
CHROMA_DIR = REPO_ROOT / "data_corpus" / "vector_store" / "chroma"
CHROMA_ZIP = REPO_ROOT / "data_corpus" / "vector_store" / "chroma_db.zip"
PDF_DIR = REPO_ROOT / "data_corpus" / "pdf"
PROCESSED_DIR = REPO_ROOT / "data_corpus" / "processed"
PROCESSED_ZIP = REPO_ROOT / "data_corpus" / "processed.zip"
WEBAPP_URL = "http://127.0.0.1:8080"
OLLAMA_TAGS_URL = "http://127.0.0.1:11434/api/tags"
# Only relevant if CLINICAL_RAG_GENERATOR_MODEL opts into an Ollama tag --
# the default generator/judge (generation/llm.py's GENERATOR_MODEL_NAME /
# JUDGE_MODEL_NAME) are transformers models needing no Ollama at all
# (dev_logs.md Entry 18). Read from the env var here (not imported from
# generation/llm.py) so this check works even before step 1 has installed
# anything.
OLLAMA_GENERATOR_MODEL = os.environ.get("CLINICAL_RAG_GENERATOR_MODEL")


def _step(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def _run(args: list[str]) -> int:
    """Runs a subprocess with output streamed live (not captured), so the
    ProgressTracker/print output from ingestion, chunking, etc. is visible
    as it happens rather than silently buffered until the step finishes."""
    print(f"$ {' '.join(args)}")
    return subprocess.run(args, cwd=REPO_ROOT).returncode


def step_install(skip: bool) -> None:
    _step("1) Python dependencies")
    if skip:
        print("Skipped (--skip-install).")
        return
    rc = _run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    if rc != 0:
        print(f"WARNING: pip install exited with code {rc} -- continuing anyway; "
              f"some steps below may fail if a required package is missing.")


def step_unzip_chroma() -> None:
    _step("2) Vector index (Chroma)")
    if CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir()):
        print(f"Already present at {CHROMA_DIR} -- skipping.")
        return
    if not CHROMA_ZIP.exists():
        print(f"WARNING: {CHROMA_ZIP} not found and {CHROMA_DIR} is missing/empty. "
              f"The vector index will be rebuilt from scratch during ingestion instead "
              f"(slower, but the system will still work).")
        return
    print(f"Unzipping {CHROMA_ZIP.name} -> {CHROMA_DIR.parent}/ ...")
    CHROMA_DIR.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(CHROMA_ZIP) as zf:
        zf.extractall(CHROMA_DIR.parent)
    print("Done.")


def _has_any_chunks() -> bool:
    return PROCESSED_DIR.exists() and any(PROCESSED_DIR.glob("*/chunks.jsonl"))


def step_ingest_and_chunk(skip: bool) -> None:
    _step("3) Ingestion + chunking")
    if skip:
        print("Skipped (--skip-ingest).")
        return
    if _has_any_chunks():
        print(f"{PROCESSED_DIR} already has chunked documents -- skipping. "
              f"(Delete data_corpus/processed/ and re-run to force a rebuild.)")
        return

    # Restoring from the shipped zip (chunks.jsonl / metadata.json /
    # router_text.txt per document, guideline.json / section_titles_summary.txt
    # per guideline -- the runtime-read subset, not the full processed/ tree,
    # see dev_logs.md Entry 22) is what the vector index alone can't provide:
    # Chroma only carries minimal filterable metadata, so this must exist for
    # the webapp/eval to hydrate real chunk text at query time -- distinct
    # from, and required in addition to, chroma_db.zip. Previously this step
    # unconditionally re-ran full Docling PDF parsing + chunking on every
    # fresh clone even though the vector index was already restored, since
    # nothing shipped these records -- confirmed as the actual cause of a
    # from-scratch setup re-parsing all PDFs despite the index being present.
    if PROCESSED_ZIP.exists():
        print(f"Restoring chunk/router records from {PROCESSED_ZIP.name} -> {PROCESSED_DIR}/ ...")
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(PROCESSED_ZIP) as zf:
            zf.extractall(PROCESSED_DIR.parent)
        print("Done.")
        return

    if not PDF_DIR.exists() or not any(PDF_DIR.iterdir()):
        print(f"WARNING: {PDF_DIR} is empty or missing -- nothing to ingest. "
              f"Place guideline PDFs under data_corpus/pdf/<guideline_id>/ and re-run.")
        return

    print("No chunked documents found -- running ingestion then chunking. "
          "This parses every PDF (Docling) and can take a while the first time.")
    rc = _run([sys.executable, "-m", "ingestion.run_ingest", "--input", str(PDF_DIR)])
    if rc != 0:
        print(f"WARNING: ingestion exited with code {rc} -- some documents may have failed. "
              f"Continuing to chunking with whatever succeeded.")
    rc = _run([sys.executable, "-m", "chunking.build_chunks"])
    if rc != 0:
        print(f"WARNING: chunking exited with code {rc}.")


def step_langfuse(langfuse_dir: Path | None) -> None:
    _step("4) Langfuse tracing (optional)")
    candidate = langfuse_dir or (REPO_ROOT.parent / "langfuse_v2")
    if not (candidate / "docker-compose.yml").exists():
        print(f"No Langfuse checkout found at {candidate} -- skipping tracing. "
              f"(Optional: clone https://github.com/langfuse/langfuse there, or pass "
              f"--langfuse-dir, to enable it. The webapp works fine without it.)")
        return

    compose_cmd = None
    for candidate_cmd in (["podman-compose"], ["docker-compose"], ["docker", "compose"]):
        if shutil.which(candidate_cmd[0]):
            compose_cmd = candidate_cmd
            break
    if compose_cmd is None:
        print("No podman-compose/docker-compose/docker found on PATH -- skipping tracing. "
              "(Optional: install Podman or Docker to enable it.)")
        return

    print(f"Starting Langfuse via {' '.join(compose_cmd)} in {candidate} ...")
    try:
        rc = subprocess.run(compose_cmd + ["up", "-d"], cwd=candidate, timeout=120).returncode
        if rc == 0:
            print("Langfuse started -- traces will appear at http://localhost:3000")
        else:
            print(f"WARNING: Langfuse compose exited with code {rc} -- continuing without tracing.")
    except Exception as e:
        # Langfuse is explicitly optional -- any failure here (missing
        # binary, compose file error, timeout, permissions) must never block
        # the steps below, which is why this is caught broadly rather than
        # narrowed to a specific exception type.
        print(f"WARNING: could not start Langfuse ({type(e).__name__}: {e}) -- continuing without tracing.")


def _ollama_tags() -> list[str] | None:
    """None if Ollama isn't reachable at all; a (possibly empty) tag list
    otherwise."""
    try:
        with urllib.request.urlopen(OLLAMA_TAGS_URL, timeout=3) as resp:
            import json
            return [m["name"] for m in json.loads(resp.read())["models"]]
    except Exception:
        return None


def _install_ollama() -> bool:
    """Auto-installs Ollama for the current OS. Returns True if it appears
    to have worked (the API becomes reachable afterward). This is what
    actually satisfies "runs with no Ollama pre-installed" -- Ollama itself
    is a single lightweight installer/binary, not Docker; it needs no
    account for local (non-`-cloud`) models, which is all this project's
    default config uses.

    Each platform branch below is implemented against that platform's
    documented, standard install method, but only ever exercised on Windows
    in this project's own development (where Ollama was already installed,
    so the auto-install branch itself was never actually triggered here) --
    worth a real test on a genuinely Ollama-less machine of each OS before
    fully trusting this in the field, not just trusting it from reading the
    code."""
    system = platform.system()
    print(f"Ollama not found -- attempting to install it automatically ({system})...")
    try:
        if system == "Windows":
            if shutil.which("winget"):
                subprocess.run(
                    ["winget", "install", "-e", "--id", "Ollama.Ollama", "--silent",
                     "--accept-package-agreements", "--accept-source-agreements"],
                    timeout=600, check=True,
                )
            else:
                installer = Path(tempfile.gettempdir()) / "OllamaSetup.exe"
                urllib.request.urlretrieve("https://ollama.com/download/OllamaSetup.exe", installer)
                # No documented silent-install flag for this installer as of
                # writing -- runs it and waits, which may show a UI the user
                # needs to click through once.
                subprocess.run([str(installer)], timeout=600, check=True)
        elif system == "Darwin":
            if shutil.which("brew"):
                subprocess.run(["brew", "install", "ollama"], timeout=600, check=True)
            else:
                print("WARNING: Homebrew not found -- install Ollama manually from "
                      "https://ollama.com/download/mac, or install brew first.")
                return False
        elif system == "Linux":
            # The standard, documented Linux install command.
            subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", shell=True, timeout=600, check=True)
        else:
            print(f"WARNING: unrecognized platform '{system}' -- install Ollama manually from https://ollama.com")
            return False
    except Exception as e:
        print(f"WARNING: automatic Ollama install failed ({type(e).__name__}: {e}). "
              f"Install manually from https://ollama.com and re-run.")
        return False

    # The installer may need a moment to actually start the Ollama service
    # (or the user's PATH may need refreshing to see the newly-installed
    # `ollama` binary in THIS process -- can't help that mid-process, but the
    # server reachability check below doesn't depend on PATH).
    for _ in range(15):
        if _ollama_tags() is not None:
            return True
        time.sleep(2)
    return False


def step_ensure_ollama() -> None:
    _step("5) Ollama (only relevant if CLINICAL_RAG_GENERATOR_MODEL is set)")
    if not OLLAMA_GENERATOR_MODEL:
        print("CLINICAL_RAG_GENERATOR_MODEL isn't set -- using the default transformers "
              "generator/judge (generation/llm.py), which need no Ollama at all. Skipping. "
              "(Their weights download automatically via huggingface_hub the first time the "
              "webapp actually generates/judges something, same as the retrieval models.)")
        return

    tags = _ollama_tags()
    if tags is None:
        if not _install_ollama():
            print(f"WARNING: could not get Ollama running automatically. The UI will start, "
                  f"but answer generation will fail until Ollama is running with "
                  f"'{OLLAMA_GENERATOR_MODEL}' pulled. "
                  f"Install from https://ollama.com, then `ollama pull {OLLAMA_GENERATOR_MODEL}`.")
            return
        tags = _ollama_tags() or []

    if OLLAMA_GENERATOR_MODEL in tags:
        print(f"'{OLLAMA_GENERATOR_MODEL}' already pulled.")
        return
    print(f"Pulling '{OLLAMA_GENERATOR_MODEL}' (one-time download, size varies by model)...")
    rc = _run(["ollama", "pull", OLLAMA_GENERATOR_MODEL])
    if rc != 0:
        print(f"WARNING: `ollama pull {OLLAMA_GENERATOR_MODEL}` exited with code {rc} -- "
              f"generation will fail until it's pulled successfully.")


def step_start_webapp() -> None:
    _step("6) Starting the webapp")
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


def main() -> None:
    # Line-buffer stdout even when redirected to a file/log (default is
    # block-buffered in that case) -- otherwise progress from this script's
    # own print() calls can sit invisible for minutes behind whatever a
    # long-running subprocess step is doing.
    sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-install", action="store_true", help="Skip pip install -r requirements.txt")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip ingestion/chunking even if data_corpus/processed/ is empty")
    parser.add_argument("--langfuse-dir", type=Path, default=None, help="Path to a langfuse_v2 checkout (default: ../langfuse_v2 relative to this repo)")
    args = parser.parse_args()

    step_install(args.skip_install)
    step_unzip_chroma()
    step_ingest_and_chunk(args.skip_ingest)
    step_langfuse(args.langfuse_dir)
    step_ensure_ollama()
    step_start_webapp()


if __name__ == "__main__":
    main()
