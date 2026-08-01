"""Tracks which source PDFs have already been ingested, keyed by content hash
(not just filename/mtime) so a PDF replaced in place with different content
under the same name is still detected as changed. This is what lets
run_ingest.py skip re-parsing (the expensive Docling step) any PDF that's
already been ingested and hasn't changed, instead of reprocessing the whole
corpus every time one new guideline is added.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

PROCESSED_DIR = Path(__file__).parent.parent / "data_corpus" / "processed"
MANIFEST_PATH = PROCESSED_DIR / "ingested_manifest.json"


def pdf_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest() -> dict[str, dict]:
    if not MANIFEST_PATH.exists():
        return {}
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def save_manifest(manifest: dict[str, dict]) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def manifest_key(guideline_id: str, pdf_filename: str) -> str:
    return f"{guideline_id}/{pdf_filename}"


def is_up_to_date(manifest: dict, guideline_id: str, pdf_path: Path) -> bool:
    entry = manifest.get(manifest_key(guideline_id, pdf_path.name))
    return entry is not None and entry.get("sha256") == pdf_hash(pdf_path)


def record(manifest: dict, guideline_id: str, pdf_path: Path, doc_id: str) -> None:
    manifest[manifest_key(guideline_id, pdf_path.name)] = {
        "sha256": pdf_hash(pdf_path),
        "doc_id": doc_id,
    }
