"""Guideline-level metadata -- one level above DocumentMetadata in the new
Corpus -> Guideline -> Document -> Section -> Chunk hierarchy. A guideline
(one AWMF register number, one folder under data_corpus/pdf/) can have
several documents (Langfassung, Kurzfassung, Methodenreport, Patientenversion,
English, ...); this captures what's shared across all of them.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from pydantic import BaseModel, Field

from .schema import FieldSource

# Authority ranking for DocumentMetadata.source_priority -- lower is more
# authoritative. Used only for rerank tie-breaking/near-duplicate dedup
# (retrieval/hybrid_search.py), never to exclude a document from indexing.
# Matched against normalized doc_type strings from rules/__init__.py.
SOURCE_PRIORITY = {
    "langfassung": 0,
    "methodenreport": 1,
    "kurzfassung": 1,
    "english": 2,
    "patientenversion": 2,
    "kurzinformation": 3,
    "other": 3,
    "unknown": 3,
}


class GuidelineMetadata(BaseModel):
    guideline_id: str  # AWMF register number, e.g. "015-027OL" -- also the folder name
    title: Optional[str] = None
    version: Optional[str] = None
    stand_date: Optional[date] = None
    valid_until: Optional[date] = None
    last_change_note: Optional[str] = None
    publishing_organization: Optional[str] = None
    purpose_text: Optional[str] = None  # "Zielorientierung der Leitlinie"
    related_guidelines: list[str] = Field(default_factory=list)
    # doc-type label (as printed in the sidecar) -> PDF filename
    document_manifest: dict[str, str] = Field(default_factory=dict)

    source_file: str  # path to the .txt sidecar
    field_sources: dict[str, FieldSource] = Field(default_factory=dict)

    class Config:
        json_encoders = {date: lambda d: d.isoformat()}
