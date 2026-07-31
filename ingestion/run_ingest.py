"""CLI entrypoint for ingestion. A "guideline folder" is any directory
containing PDFs (data_corpus/pdf/<guideline_id>/) -- its <guideline_id>.txt
sidecar, if present, is used as the primary metadata source (see
build_document.build_guideline). --input can be a single guideline folder,
a parent directory of several guideline folders (batch mode -- how this
actually runs at the 10-15 guideline target), or a single PDF file.

Usage:
    python -m ingestion.run_ingest --input "data_corpus/pdf/015-027OL"
    python -m ingestion.run_ingest --input "data_corpus/pdf/"
    python -m ingestion.run_ingest --input "data_corpus/pdf/015-027OL/015-027OLl_....pdf"
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from common.progress import ProgressTracker

from .build_document import build_document, build_guideline


def _is_guideline_folder(path: Path) -> bool:
    return path.is_dir() and any(path.glob("*.pdf"))


def _iter_guideline_folders(input_path: Path):
    if _is_guideline_folder(input_path):
        yield input_path
        return
    for sub in sorted(input_path.iterdir()):
        if _is_guideline_folder(sub):
            yield sub


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest guideline PDF(s) into data_corpus/processed/")
    parser.add_argument("--input", required=True, help="Path to a single PDF, a guideline folder, or a parent of several guideline folders")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input path does not exist: {input_path}", file=sys.stderr)
        return 1

    if input_path.is_file():
        print(f"Ingesting single file: {input_path}")
        try:
            out_dir = build_document(input_path)
            print(f"  -> {out_dir}")
            return 0
        except Exception:
            traceback.print_exc()
            return 1

    guideline_folders = list(_iter_guideline_folders(input_path))
    if not guideline_folders:
        print(f"No guideline folders (directories containing PDFs) found under {input_path}", file=sys.stderr)
        return 1

    total_pdfs = sum(len(list(folder.glob("*.pdf"))) for folder in guideline_folders)
    tracker = ProgressTracker("ingest", total=total_pdfs, stage="starting")

    total_docs, failures = 0, 0
    for folder in guideline_folders:
        print(f"Ingesting guideline: {folder.name}")
        tracker.set_stage(f"guideline:{folder.name}")

        def _on_pdf_done(fname: str, ok: bool, g: str = folder.name) -> None:
            tracker.set_stage(f"guideline:{g} pdf:{fname}")
            tracker.advance(1)

        try:
            out_dirs = build_guideline(folder, on_pdf_done=_on_pdf_done)
            total_docs += len(out_dirs)
            for out_dir in out_dirs:
                print(f"  -> {out_dir}")
        except Exception:
            failures += 1
            print(f"  FAILED: {folder}", file=sys.stderr)
            traceback.print_exc()

    tracker.finish(error=f"{failures} guideline(s) failed" if failures else None)
    print(f"\nDone: {len(guideline_folders) - failures}/{len(guideline_folders)} guidelines succeeded, {total_docs} documents ingested.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
