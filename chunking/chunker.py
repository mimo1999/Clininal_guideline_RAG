"""Structure-aware hierarchical chunking (renamed from "semantic chunking" --
this is document-hierarchy-driven paragraph/token packing, not embedding-
based split-point detection).

For each leaf section (from structure.py): strip the heading line and any
table blocks (handled by tables.py), split remaining prose into paragraphs,
and greedily pack paragraphs into token-range chunks with overlap for
oversized paragraphs and merging for undersized trailing fragments. Table
chunks are appended after a section's text chunks, atomic and never split.
Every chunk gets linked-list fields (previous/next within its section,
parent pointing at the section's summary chunk) for retrieval-time neighbor
expansion.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from .structure import Section
from .tables import extract_tables

TARGET_MIN_TOKENS = 300
TARGET_MAX_TOKENS = 500
HARD_CAP_TOKENS = 600
OVERLAP_RATIO = 0.15
MERGE_THRESHOLD_TOKENS = 100

PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n")
HEADING_LINE_RE = re.compile(r"^## .+$\n?", re.MULTILINE)
TABLE_BLOCK_RE = re.compile(r"(?:^\|.*\|\s*$\n?)+", re.MULTILINE)


@dataclass
class TextPiece:
    text: str
    token_count: int


@lru_cache(maxsize=1)
def _tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("BAAI/bge-m3")


def count_tokens(text: str) -> int:
    return len(_tokenizer().encode(text, add_special_tokens=False))


def _strip_headings_and_tables(body: str) -> str:
    body = HEADING_LINE_RE.sub("", body, count=1)  # only the section's own heading
    body = TABLE_BLOCK_RE.sub("", body)
    return body


def _split_paragraphs(text: str) -> list[str]:
    parts = [p.strip() for p in PARAGRAPH_SPLIT_RE.split(text)]
    return [p for p in parts if p]


def _split_oversized_paragraph(paragraph: str) -> list[str]:
    tok = _tokenizer()
    ids = tok.encode(paragraph, add_special_tokens=False)
    if len(ids) <= HARD_CAP_TOKENS:
        return [paragraph]

    step = int(TARGET_MAX_TOKENS * (1 - OVERLAP_RATIO))
    pieces = []
    start = 0
    while start < len(ids):
        end = min(start + TARGET_MAX_TOKENS, len(ids))
        pieces.append(tok.decode(ids[start:end]))
        if end == len(ids):
            break
        start += step
    return pieces


def _pack_paragraphs(paragraphs: list[str]) -> list[TextPiece]:
    pieces: list[str] = []
    for p in paragraphs:
        pieces.extend(_split_oversized_paragraph(p))

    chunks: list[TextPiece] = []
    buffer_parts: list[str] = []
    buffer_tokens = 0

    def flush():
        nonlocal buffer_parts, buffer_tokens
        if buffer_parts:
            text = "\n\n".join(buffer_parts)
            chunks.append(TextPiece(text=text, token_count=buffer_tokens))
        buffer_parts, buffer_tokens = [], 0

    for piece in pieces:
        piece_tokens = count_tokens(piece)
        if buffer_tokens and buffer_tokens + piece_tokens > HARD_CAP_TOKENS:
            flush()
        buffer_parts.append(piece)
        buffer_tokens += piece_tokens
        if buffer_tokens >= TARGET_MAX_TOKENS:
            flush()
    flush()

    # Merge an undersized trailing chunk into the previous one within this section.
    if len(chunks) >= 2 and chunks[-1].token_count < MERGE_THRESHOLD_TOKENS:
        last = chunks.pop()
        prev = chunks.pop()
        merged_text = prev.text + "\n\n" + last.text
        chunks.append(TextPiece(text=merged_text, token_count=count_tokens(merged_text)))

    return chunks


def chunk_section(section: Section) -> tuple[list[TextPiece], list]:
    """Returns (text_pieces, table_chunks) for one leaf section."""
    prose = _strip_headings_and_tables(section.body)
    paragraphs = _split_paragraphs(prose)
    text_pieces = _pack_paragraphs(paragraphs) if paragraphs else []
    table_chunks = extract_tables(section.body)
    return text_pieces, table_chunks
