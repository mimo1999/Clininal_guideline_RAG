"""Chunk metadata schema -- the record stored per chunk in the vector store."""

from __future__ import annotations

from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field

ChunkType = Literal["text", "table", "section_summary"]
FieldSource = Literal["scrape", "regex", "synthetic", "missing"]


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    chunk_type: ChunkType

    text: str  # what gets embedded (packed paragraphs / table embedded_text / section summary)
    markdown_table: Optional[str] = None  # table chunks only, for citation display
    flattened_table_text: Optional[str] = None  # table chunks only

    heading_path: list[str] = Field(default_factory=list)
    section_number: str = ""
    section_title: str = ""
    section_id: str = ""
    page_numbers: list[int] = Field(default_factory=list)
    token_count: int = 0

    topic: list[str] = Field(default_factory=list)
    recommendation_id: Optional[str] = None
    evidence_grade: Optional[str] = None
    evidence_grade_source: FieldSource = "missing"

    parent_chunk_id: Optional[str] = None
    previous_chunk_id: Optional[str] = None
    next_chunk_id: Optional[str] = None

    # Denormalized doc-level fields -- avoids a join at query/citation time.
    doc_title: Optional[str] = None
    doc_type: Optional[str] = None  # langfassung / kurzfassung / methodenreport / patientenversion / english / other
    # Authority ranking among a guideline's document types (lower = more
    # authoritative), for rerank tie-breaking / near-duplicate dedup -- see
    # ingestion.guideline_schema.SOURCE_PRIORITY. Never used to exclude a chunk.
    source_priority: int = 3
    awmf_register_number: Optional[str] = None
    doc_version: Optional[str] = None
    doc_language: Optional[str] = None
    publishing_organization: Optional[str] = None
    awmf_stand_date: Optional[date] = None

    # Denormalized guideline-level fields (one level above doc_*) -- the new
    # Corpus -> Guideline -> Document -> Section -> Chunk hierarchy.
    guideline_id: Optional[str] = None
    guideline_title: Optional[str] = None
