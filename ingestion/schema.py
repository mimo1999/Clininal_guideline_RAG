"""Metadata schema for ingested guideline documents."""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field

FieldSource = Literal["sidecar", "scrape", "regex", "langid", "missing"]

# Fields whose provenance we track and report on.
TRACKED_FIELDS = (
    "title",
    "version",
    "awmf_stand_date",
    "awmf_valid_until",
    "publishing_organization",
    "language",
)


class DocumentMetadata(BaseModel):
    doc_id: str
    title: Optional[str] = None
    awmf_register_number: Optional[str] = None
    version: Optional[str] = None
    awmf_stand_date: Optional[date] = None
    awmf_valid_until: Optional[date] = None
    awmf_last_change_note: Optional[str] = None
    publishing_organization: Optional[str] = None
    language: Optional[str] = None
    language_confidence: Optional[float] = None
    # The PDF's own printed cover-page date (e.g. "Langversion 1.1 -Maerz 2020"),
    # kept distinct from awmf_stand_date -- these can legitimately disagree (see
    # 015-027OL: register Stand is 2017-12, but the long-version PDF was last
    # replaced 2020-03). Year-month precision only; no day is printed on covers.
    pdf_cover_year_month: Optional[str] = None

    source_file: str
    source_url: Optional[str] = None
    ingestion_timestamp: datetime

    doc_type: Optional[str] = None  # langfassung / methodenreport / kurzfassung / patientenversion / english / other / unknown
    # Authority ranking among a guideline's document types, for rerank
    # tie-breaking/dedup (lower = more authoritative) -- see
    # guideline_schema.SOURCE_PRIORITY. Never used to exclude a document,
    # only to prefer one when near-duplicate content surfaces from several.
    source_priority: int = 3

    field_sources: dict[str, FieldSource] = Field(default_factory=dict)

    class Config:
        json_encoders = {
            date: lambda d: d.isoformat(),
            datetime: lambda d: d.isoformat(),
        }
