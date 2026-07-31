"""Table extraction: pulls genuine content tables (not AWMF recommendation
boxes, which structure.py handles separately) out of a section's Markdown,
keeping three representations per table:

- markdown_table: original, for citation display.
- flattened_text: one clause per row ("<header>: <value>; ..."), since row/
  column structure is exactly what small local LLMs struggle to parse.
- embedded_text: caption + flattened_text -- the caption (e.g. "Tabelle 3.1
  Klassifizierung von HPV-Typen...") carries real semantic signal that's
  lost if only the flattened rows are embedded.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .structure import RECOMMENDATION_RE

# chunker.py imports extract_tables from this module, so count_tokens is
# imported lazily inside the functions that need it (below) rather than at
# module level, to avoid a circular import.

TABLE_BLOCK_RE = re.compile(r"(?:^\|.*\|\s*$\n?)+", re.MULTILINE)
CAPTION_RE = re.compile(r"^Tabelle\s+[\d.]+\s*:?\s*(.+)$", re.MULTILINE)
SEPARATOR_ROW_RE = re.compile(r"^\|[\s:|-]+\|\s*$")

# "Atomic, never split" (see module docstring) assumed every table was
# reasonably sized. A real guideline document broke that assumption hard: one
# table produced a single ~49,000-token chunk (195KB), and several others hit
# 4,000-8,000 tokens -- confirmed via a direct scan of chunks.jsonl after this
# bug caused an apparent multi-minute hang in the reranker/embedder (attention
# cost is O(n^2), so a near-max-length sequence is drastically more expensive
# than a short one, quite apart from being useless as a citation -- nobody
# cites a 195KB "passage"). Tables now split by rows once they'd exceed this,
# matching the text-chunking target range instead of being unbounded.
TABLE_CHUNK_MAX_TOKENS = 500


@dataclass
class TableChunk:
    caption: str | None
    markdown_table: str
    flattened_text: str
    embedded_text: str
    start: int
    end: int


def _parse_rows(table_block: str) -> tuple[list[list[str]], list[str], str]:
    """Returns (rows, raw_data_lines, separator_line). Raw lines (header +
    data, excluding the separator) let an oversized table's split chunks
    reassemble valid markdown -- header + separator + a row subset -- without
    re-serializing cells and risking a formatting mismatch with the source."""
    rows: list[list[str]] = []
    raw_lines: list[str] = []
    separator_line = ""
    for line in table_block.strip().split("\n"):
        stripped = line.strip()
        if SEPARATOR_ROW_RE.match(stripped):
            separator_line = stripped
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        rows.append(cells)
        raw_lines.append(stripped)
    return rows, raw_lines, separator_line


def _split_rows_by_token_budget(
    header: list[str], data_rows: list[list[str]], max_tokens: int = TABLE_CHUNK_MAX_TOKENS
) -> list[list[list[str]]]:
    """O(n): each row's token cost is measured once (not recomputed per
    candidate group), so this stays linear even for a table with thousands
    of rows -- the exact scenario that needed splitting in the first place."""
    from .chunker import count_tokens

    if not data_rows:
        return []
    row_tokens = [count_tokens(_flatten([header, row])) for row in data_rows]
    groups: list[list[list[str]]] = []
    current: list[list[str]] = []
    current_tokens = 0
    for row, tok in zip(data_rows, row_tokens):
        if current and current_tokens + tok > max_tokens:
            groups.append(current)
            current, current_tokens = [row], tok
        else:
            current.append(row)
            current_tokens += tok
    if current:
        groups.append(current)
    return groups


def _is_recommendation_table(table_block: str) -> bool:
    return bool(RECOMMENDATION_RE.search(table_block))


def _flatten(rows: list[list[str]]) -> str:
    if len(rows) < 2:
        return ""
    header = rows[0]
    lines = []
    for row in rows[1:]:
        pairs = [f"{h}: {v}" for h, v in zip(header, row) if v]
        if pairs:
            lines.append("; ".join(pairs))
    return "\n".join(lines)


def _find_caption(text: str, table_start: int) -> str | None:
    preceding = text[:table_start]
    matches = list(CAPTION_RE.finditer(preceding))
    if not matches:
        return None
    last = matches[-1]
    # only accept a caption that's "close" to the table (within ~300 chars,
    # i.e. not some unrelated earlier table's caption)
    if table_start - last.end() > 300:
        return None
    return last.group(0).strip()


def extract_tables(section_body: str) -> list[TableChunk]:
    from .chunker import count_tokens

    tables = []
    for m in TABLE_BLOCK_RE.finditer(section_body):
        block = m.group(0)
        if _is_recommendation_table(block):
            continue
        rows, raw_lines, separator_line = _parse_rows(block)
        flattened = _flatten(rows)
        if not flattened:
            continue
        caption = _find_caption(section_body, m.start())
        embedded_text = f"{caption}\n{flattened}" if caption else flattened

        if count_tokens(embedded_text) <= TABLE_CHUNK_MAX_TOKENS:
            tables.append(TableChunk(
                caption=caption,
                markdown_table=block.strip(),
                flattened_text=flattened,
                embedded_text=embedded_text,
                start=m.start(),
                end=m.end(),
            ))
            continue

        header, header_line = rows[0], raw_lines[0]
        groups = _split_rows_by_token_budget(header, rows[1:])
        data_lines = raw_lines[1:]  # aligned 1:1 with rows[1:]
        row_offset = 0
        for gi, group_rows in enumerate(groups):
            group_flat = _flatten([header] + group_rows)
            group_lines = data_lines[row_offset:row_offset + len(group_rows)]
            row_offset += len(group_rows)
            if not group_flat:
                continue
            part_caption = f"{caption} (Teil {gi + 1}/{len(groups)})" if caption else None
            group_embedded = f"{part_caption}\n{group_flat}" if part_caption else group_flat
            group_markdown = "\n".join([header_line, separator_line, *group_lines])
            tables.append(TableChunk(
                caption=part_caption,
                markdown_table=group_markdown,
                flattened_text=group_flat,
                embedded_text=group_embedded,
                start=m.start(),
                end=m.end(),
            ))
    return tables
