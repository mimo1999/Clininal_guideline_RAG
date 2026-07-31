"""Deterministic merge of guideline-sidecar, AWMF-scrape, cover-page-regex,
and langid signals into a single DocumentMetadata record -- no LLM, no human
step. Priority cascade: sidecar > scrape > regex > missing (sidecar added in
the multi-guideline phase -- it's a locally-available, pre-fetched copy of
the same register-page facts scrape_awmf.py fetches live, so it wins when
present and the live scrape becomes the fallback for guidelines added
without one). Every field gets a provenance tag, and any gap or
cross-source disagreement is appended to an audit log for later visibility
rather than blocking ingestion.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from .langid import LanguageResult
from .parse_guideline_sidecar import SidecarResult
from .schema import DocumentMetadata
from .scrape_awmf import ScrapedDetail

ISSUES_LOG_PATH = Path(__file__).parent.parent / "data_corpus" / "processed" / "ingestion_issues.jsonl"


def _log_issue(doc_id: str, issue_type: str, detail: str) -> None:
    ISSUES_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "doc_id": doc_id,
        "type": issue_type,
        "detail": detail,
        "logged_at": datetime.utcnow().isoformat(),
    }
    with ISSUES_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _pick(doc_id: str, field: str, candidates: list[tuple[object, str]]) -> tuple[Optional[object], str]:
    """candidates: priority-ordered [(value, source_name), ...] -- first
    non-empty wins; every other non-empty, disagreeing candidate is logged."""
    chosen, chosen_source = None, "missing"
    for value, source in candidates:
        if value not in (None, ""):
            chosen, chosen_source = value, source
            break

    if chosen is None:
        _log_issue(doc_id, "field_missing", f"{field}: not resolved by any source")
        return None, "missing"

    for value, source in candidates:
        if source == chosen_source or value in (None, ""):
            continue
        if str(value) != str(chosen):
            _log_issue(
                doc_id, "field_conflict",
                f"{field}: {chosen_source}={chosen!r} vs {source}={value!r} -- kept {chosen_source} value",
            )
    return chosen, chosen_source


def reconcile(
    doc_id: str,
    source_file: str,
    scraped: Optional[ScrapedDetail],
    regex_fields: dict,
    lang_result: LanguageResult,
    sidecar: Optional[SidecarResult] = None,
    guideline_id: Optional[str] = None,
) -> DocumentMetadata:
    field_sources: dict[str, str] = {}
    sidecar_stand = sidecar.stand_date if sidecar else None
    sidecar_valid_until = sidecar.valid_until_date if sidecar else None

    title, field_sources["title"] = _pick(doc_id, "title", [
        (sidecar.title if sidecar else None, "sidecar"),
        (scraped.title if scraped else None, "scrape"),
        (regex_fields.get("title"), "regex"),
    ])
    version, field_sources["version"] = _pick(doc_id, "version", [
        (sidecar.version if sidecar else None, "sidecar"),
        (scraped.version if scraped else None, "scrape"),
        (regex_fields.get("version"), "regex"),
    ])
    stand_date, field_sources["awmf_stand_date"] = _pick(doc_id, "awmf_stand_date", [
        (sidecar_stand, "sidecar"),
        (scraped.stand_date if scraped else None, "scrape"),
    ])
    valid_until, field_sources["awmf_valid_until"] = _pick(doc_id, "awmf_valid_until", [
        (sidecar_valid_until, "sidecar"),
        (scraped.valid_until_date if scraped else None, "scrape"),
    ])
    publishing_org, field_sources["publishing_organization"] = _pick(doc_id, "publishing_organization", [
        (sidecar.publishing_organization if sidecar else None, "sidecar"),
        (scraped.publishing_organization if scraped else None, "scrape"),
    ])
    register_number, field_sources["awmf_register_number"] = _pick(doc_id, "awmf_register_number", [
        (guideline_id, "sidecar"),  # the folder name IS the register number -- always authoritative when known
        (scraped.register_number if scraped else None, "scrape"),
        (regex_fields.get("awmf_register_number"), "regex"),
    ])

    field_sources["language"] = "langid"

    pdf_cover_year_month = regex_fields.get("cover_year_month")
    if scraped and stand_date and pdf_cover_year_month:
        stand_ym = stand_date[:7]
        if stand_ym != pdf_cover_year_month:
            _log_issue(
                doc_id, "date_discrepancy",
                f"AWMF Stand ({stand_date}) differs from the PDF cover's own printed date "
                f"({pdf_cover_year_month}) -- both are recorded distinctly, do not treat "
                f"either alone as 'the' publish date",
            )

    from .guideline_schema import SOURCE_PRIORITY

    doc_type = regex_fields.get("doc_type") or "unknown"

    return DocumentMetadata(
        doc_id=doc_id,
        title=title,
        awmf_register_number=register_number,
        version=version,
        awmf_stand_date=date.fromisoformat(stand_date) if stand_date else None,
        awmf_valid_until=date.fromisoformat(valid_until) if valid_until else None,
        awmf_last_change_note=(sidecar.last_change_note if sidecar else None) or (scraped.last_change_note if scraped else None),
        publishing_organization=publishing_org,
        language=lang_result.language,
        language_confidence=lang_result.confidence,
        pdf_cover_year_month=pdf_cover_year_month,
        source_file=source_file,
        source_url=scraped.source_url if scraped else None,
        ingestion_timestamp=datetime.utcnow(),
        doc_type=doc_type,
        source_priority=SOURCE_PRIORITY.get(doc_type, 3),
        field_sources=field_sources,
    )
