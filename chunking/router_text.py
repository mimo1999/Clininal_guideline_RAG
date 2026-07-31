"""Builds deterministic router text for the guideline- and document-level
routers (retrieval/guideline_router.py, retrieval/document_router.py) --
explicitly NOT LLM-generated: reproducible from the already-parsed structure
tree alone (top headings + first paragraph + section titles), no hallucination
risk, no LLM in the routing critical path.
"""

from __future__ import annotations

import re

from .structure import Section

_HEADING_LINE_RE = re.compile(r"^## .+$\n?", re.MULTILINE)

# Router text is a coarse routing signal, not a corpus index -- embedding it
# should stay well inside bge-m3's efficient range. Without a cap, a
# Langfassung with 100+ sections (or a 7-document guideline) produces a
# multi-thousand-token string; embedding a near-max-length (8192) sequence is
# dramatically more expensive than a short one (attention is O(n^2)), and it
# was measured taking 79-151s for a single guideline/document router call on
# this hardware -- confirmed the cause of an apparent hang, not a real one.
# 2000 chars (~300-500 tokens) keeps routing embeddings fast without losing
# the distinguishing early titles that actually carry the topic signal.
MAX_TITLES_CHARS = 2000


def _first_paragraph(sections: list[Section], max_chars: int = 500) -> str:
    for s in sections:
        body = _HEADING_LINE_RE.sub("", s.body, count=1).strip()
        if len(body) > 40:  # skip near-empty leaf sections (e.g. bare title stubs)
            return body[:max_chars]
    return ""


def build_document_router_text(doc_title: str | None, doc_type: str | None, leaves: list[Section]) -> str:
    """One document's router text: its own title/type + first substantial
    paragraph + its own section titles."""
    parts = []
    header = " ".join(p for p in (doc_type, doc_title) if p)
    if header:
        parts.append(header)
    first_para = _first_paragraph(leaves)
    if first_para:
        parts.append(first_para)
    titles = [s.section_title for s in leaves if s.section_title]
    if titles:
        parts.append(_join_titles_capped(titles))
    return "\n".join(parts)


def _join_titles_capped(titles: list[str], max_chars: int = MAX_TITLES_CHARS) -> str:
    joined_parts: list[str] = []
    total = 0
    for t in titles:
        added = len(t) + (3 if joined_parts else 0)  # " | " separator
        if total + added > max_chars:
            break
        joined_parts.append(t)
        total += added
    return " | ".join(joined_parts)


def build_guideline_section_titles_summary(all_document_section_titles: list[str]) -> str:
    """The "summary" component of a guideline's 3-way router embedding (see
    retrieval/guideline_router.py) -- deduplicated section titles pooled
    across all of the guideline's documents. Title and purpose_text are read
    directly from guideline.json by the router, not duplicated here -- this
    function's only job is the part that isn't already a single field."""
    seen: set[str] = set()
    deduped = [t for t in all_document_section_titles if t and not (t in seen or seen.add(t))]
    return _join_titles_capped(deduped)
