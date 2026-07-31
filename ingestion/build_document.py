"""Orchestrates one guideline folder (data_corpus/pdf/<guideline_id>/, holding
several PDFs + a <guideline_id>.txt sidecar) -> one {parsed Markdown,
metadata.json} per PDF in data_corpus/processed/<doc_id>/, plus one
guideline.json capturing the shared guideline-level metadata.

Pipeline per PDF: parse (Docling) -> resolve doc_type (sidecar manifest label
first, filename-suffix fallback) -> cover-page regex -> AWMF registry lookup +
detail scrape (fallback only, since the sidecar already has this) -> language
ID -> deterministic reconciliation (sidecar > scrape > regex > missing) ->
write outputs.

Every stage degrades gracefully: a scrape failure or missing sidecar field
doesn't stop ingestion, it just means more fields fall through to
regex/missing (visible via field_sources and ingestion_issues.jsonl) rather
than the document being dropped -- there's no human review step to catch a
hard failure instead.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Optional

from .guideline_schema import GuidelineMetadata
from .langid import detect_language
from .parse_guideline_sidecar import SidecarResult, parse_sidecar
from .parse_pdf import parse_pdf_isolated
from .reconcile import _log_issue, reconcile
from .registry_awmf import build_registry
from .rules import detect_doc_type_from_label, extract_filename_suffix, try_extract as regex_try_extract
from .scrape_awmf import scrape_detail

PROCESSED_DIR = Path(__file__).parent.parent / "data_corpus" / "processed"

# Corpus scope for this pass is DGGG guidelines; a document from a different
# Fachgesellschaft would need its society name passed in -- see build_document(society_hint=).
DEFAULT_SOCIETY_HINT = "Gynäkologie und Geburtshilfe"

_FILENAME_REGNR_RE = re.compile(r"^(\d{3}-\d{3}(?:OL)?)")


def _strip_markdown_artifacts(markdown: str) -> str:
    return re.sub(r"<!--\s*image\s*-->", " ", markdown)


def derive_register_number(filename: str) -> str | None:
    m = _FILENAME_REGNR_RE.match(filename)
    return m.group(1) if m else None


class _JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (date, datetime)):
            return o.isoformat()
        return super().default(o)


def _guideline_metadata_from_sidecar(guideline_id: str, sidecar_path: Path, sidecar: SidecarResult) -> GuidelineMetadata:
    field_sources = {
        f: "sidecar" for f in ("title", "version", "stand_date", "valid_until", "publishing_organization")
        if getattr(sidecar, f, None) not in (None, "")
    }
    return GuidelineMetadata(
        guideline_id=guideline_id,
        title=sidecar.title,
        version=sidecar.version,
        stand_date=date.fromisoformat(sidecar.stand_date) if sidecar.stand_date else None,
        valid_until=date.fromisoformat(sidecar.valid_until_date) if sidecar.valid_until_date else None,
        last_change_note=sidecar.last_change_note,
        publishing_organization=sidecar.publishing_organization,
        purpose_text=sidecar.purpose_text,
        related_guidelines=sidecar.related_guidelines,
        document_manifest=sidecar.document_manifest,
        source_file=str(sidecar_path),
        field_sources=field_sources,
    )


def build_document(
    pdf_path: str | Path,
    society_hint: str = DEFAULT_SOCIETY_HINT,
    sidecar: Optional[SidecarResult] = None,
    guideline_id: Optional[str] = None,
    doc_type_hint: Optional[str] = None,
    doc_id_suffix: Optional[str] = None,
) -> Path:
    pdf_path = Path(pdf_path)
    filename = pdf_path.name

    parsed = parse_pdf_isolated(pdf_path)

    register_number = guideline_id or derive_register_number(filename)
    regex_fields = regex_try_extract(filename, parsed.first_page_text, doc_type=doc_type_hint)

    doc_id = f"{register_number or pdf_path.stem}_{regex_fields.get('doc_type', 'unknown')}"
    if doc_id_suffix:
        # Two PDFs in one guideline can normalize to the same doc_type
        # category (e.g. Evidenzbericht and Leitlinienreport both ->
        # "methodenreport") -- without this they'd collide on doc_id and one
        # would silently overwrite the other's output directory. The caller
        # (build_guideline) only passes this when it has detected such a
        # collision, so a guideline with no duplicate doc_type keeps the
        # plain "{register}_{doc_type}" id other code (eval questions, etc.)
        # already expects.
        doc_id = f"{doc_id}-{doc_id_suffix}"

    scraped = None
    # The sidecar already carries the register-page facts for this guideline
    # -- only fall back to a live scrape when there's no sidecar to rely on.
    if sidecar is None and register_number:
        try:
            registry = build_registry(society_hint)
            entry = registry.get(register_number)
            scraped = scrape_detail(register_number, detail_url=entry.detail_url if entry else None)
        except Exception as e:
            _log_issue(doc_id, "scrape_failed", f"register_number={register_number}: {e}")
    elif sidecar is None:
        _log_issue(doc_id, "register_number_unresolved", f"filename={filename!r} did not match the expected pattern")

    lang_result = detect_language(_strip_markdown_artifacts(parsed.markdown))

    metadata = reconcile(
        doc_id=doc_id,
        source_file=str(pdf_path),
        scraped=scraped,
        regex_fields=regex_fields,
        lang_result=lang_result,
        sidecar=sidecar,
        guideline_id=register_number,
    )

    out_dir = PROCESSED_DIR / doc_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "parsed.md").write_text(parsed.markdown, encoding="utf-8")
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata.model_dump(), cls=_JSONEncoder, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return out_dir


def build_guideline(
    guideline_dir: str | Path,
    society_hint: str = DEFAULT_SOCIETY_HINT,
    on_pdf_done: Optional[Callable[[str, bool], None]] = None,
) -> list[Path]:
    """Ingest every PDF in one guideline folder (data_corpus/pdf/<guideline_id>/),
    using the folder's <guideline_id>.txt sidecar as the primary metadata
    source. Returns the list of per-document output dirs.

    on_pdf_done(filename, success), if given, fires after each PDF (success or
    failure) -- the progress-tracking hook for run_ingest.py; this module has
    no opinion on what the callback does with that signal."""
    guideline_dir = Path(guideline_dir)
    guideline_id = guideline_dir.name
    sidecar_path = guideline_dir / f"{guideline_id}.txt"

    sidecar = None
    if sidecar_path.exists():
        sidecar = parse_sidecar(sidecar_path, guideline_dir)
        if sidecar.unmatched_labels:
            _log_issue(
                guideline_id, "sidecar_unmatched_label",
                f"manifest label(s) with no matching PDF on disk: {sidecar.unmatched_labels}",
            )
        if sidecar.unlisted_files:
            _log_issue(
                guideline_id, "sidecar_unlisted_file",
                f"PDF(s) on disk not mentioned in the sidecar's document list: {sidecar.unlisted_files}",
            )

        guideline_meta = _guideline_metadata_from_sidecar(guideline_id, sidecar_path, sidecar)
        guideline_out_dir = PROCESSED_DIR / f"_guideline_{guideline_id}"
        guideline_out_dir.mkdir(parents=True, exist_ok=True)
        (guideline_out_dir / "guideline.json").write_text(
            json.dumps(guideline_meta.model_dump(), cls=_JSONEncoder, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    else:
        _log_issue(guideline_id, "sidecar_missing", f"no {sidecar_path.name} found in {guideline_dir}")

    # invert the manifest (filename -> label) so each PDF can look up its own doc-type label
    filename_to_label = {}
    if sidecar:
        for label, fname in sidecar.document_manifest.items():
            filename_to_label[fname.lower()] = label

    pdf_paths = sorted(guideline_dir.glob("*.pdf"))
    doc_type_hints = {}
    for pdf_path in pdf_paths:
        label = filename_to_label.get(pdf_path.name.lower())
        doc_type_hints[pdf_path] = detect_doc_type_from_label(label) if label else None

    # A resolved doc_type can repeat within one guideline (e.g. Evidenzbericht
    # and Leitlinienreport both normalize to "methodenreport") -- doc_id is
    # built from doc_type, so any doc_type used by >1 PDF here needs a
    # per-file disambiguator or the later PDF overwrites the earlier one's
    # output directory on disk.
    resolved_types = [
        doc_type_hints[p] or "unknown" for p in pdf_paths
    ]
    type_counts = Counter(resolved_types)

    out_dirs = []
    for pdf_path in pdf_paths:
        doc_type_hint = doc_type_hints[pdf_path]
        resolved_type = doc_type_hint or "unknown"
        doc_id_suffix = extract_filename_suffix(pdf_path.name) if type_counts[resolved_type] > 1 else None
        try:
            out_dirs.append(build_document(
                pdf_path, society_hint=society_hint, sidecar=sidecar,
                guideline_id=guideline_id, doc_type_hint=doc_type_hint,
                doc_id_suffix=doc_id_suffix,
            ))
            if on_pdf_done:
                on_pdf_done(pdf_path.name, True)
        except Exception as e:
            # Isolated per-PDF (subprocess) conversion means one PDF crashing
            # (e.g. std::bad_alloc under memory pressure) can't corrupt the
            # parent process -- log and continue with the rest of the guideline
            # instead of losing every remaining document to one bad PDF.
            _log_issue(guideline_id, "pdf_conversion_failed", f"{pdf_path.name}: {e}")
            if on_pdf_done:
                on_pdf_done(pdf_path.name, False)
    return out_dirs
