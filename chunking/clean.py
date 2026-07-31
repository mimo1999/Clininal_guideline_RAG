"""Boilerplate removal on Docling's Markdown output, before chunking.

Two independent mechanisms, both generalizable across the corpus (not
hand-tuned to one document):

1. A generic table-of-contents/index detector: any Markdown table whose cells
   are mostly dot-leader text ("....... 42") is structural front-matter, not
   content, regardless of which document it came from.
2. A heading-name blocklist for known non-informative section *types*
   (bibliography, list of tables/figures, abbreviation lists, committee
   rosters) -- extend the list as new corpus documents surface new patterns.

Docling emits all headings as "## <numbered heading>" regardless of nesting
depth -- hierarchy lives in the numeric prefix (e.g. "3.1.2."), not the
Markdown heading level. This module treats "## " as the universal section
boundary; structure.py is what interprets the numeric nesting.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

HEADING_RE = re.compile(r"^## (.+)$", re.MULTILINE)
DOT_LEADER_RE = re.compile(r"\.{4,}\s*\d*\s*$")
TABLE_ROW_RE = re.compile(r"^\|.*\|\s*$", re.MULTILINE)

# Case-insensitive substring match against heading text. Content under a
# matching heading is excluded until the next heading at or above depth.
BOILERPLATE_HEADING_PATTERNS = [
    "literaturverzeichnis",
    "abkürzungsverzeichnis",
    "tabellenverzeichnis",
    "abbildungsverzeichnis",
    "inhaltsverzeichnis",
    "beteiligte fachgesellschaften",
    "interessenkonflikt",
]

BIBLIOGRAPHY_HEADING_PATTERNS = ["literaturverzeichnis"]


@dataclass
class CleanResult:
    cleaned_markdown: str
    references_text: str


def _heading_depth(heading_text: str) -> int:
    m = re.match(r"^([\d.]+)\.?\s", heading_text + " ")
    if not m:
        return 0
    return heading_text.split(" ", 1)[0].rstrip(".").count(".") + 1


def _is_boilerplate_heading(heading_text: str, patterns: list[str]) -> bool:
    lowered = heading_text.lower()
    return any(p in lowered for p in patterns)


def _strip_toc_tables(text: str) -> str:
    """Remove any Markdown table block where most cells look like dot-leader
    index entries (title ....... 42)."""
    lines = text.split("\n")
    out_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith("|"):
            table_lines = []
            j = i
            while j < len(lines) and lines[j].strip().startswith("|"):
                table_lines.append(lines[j])
                j += 1
            cells = [c for row in table_lines for c in row.split("|")]
            dot_leader_cells = [c for c in cells if DOT_LEADER_RE.search(c.strip())]
            if cells and len(dot_leader_cells) / max(len(cells), 1) > 0.15:
                i = j  # drop this table
                continue
            out_lines.extend(table_lines)
            i = j
            continue
        out_lines.append(line)
        i += 1
    return "\n".join(out_lines)


_REAL_CHAPTER_HEADING_RE = re.compile(r"^## \d+\.\s+\S", re.MULTILINE)
_TABLE_LINE_RE = re.compile(r"^\|.*\|\s*$", re.MULTILINE)


def _find_bibliography_span(markdown: str) -> tuple[int, int] | None:
    """The bibliography sometimes isn't captured as a proper '## ' heading by
    Docling (observed: its layout gets misread as a table row rather than a
    heading style) -- fall back to locating it by content instead of
    heading structure. Once found, everything up to the next *real* numbered
    chapter heading (which filters out stray reference-URL lines Docling
    sometimes also mis-tags as headings) is treated as bibliography."""
    for m in _TABLE_LINE_RE.finditer(markdown):
        line = m.group(0)
        if "literaturverzeichnis" not in line.lower():
            continue
        cells = line.split("|")
        is_toc_row = any(DOT_LEADER_RE.search(c.strip()) for c in cells)
        if not is_toc_row:
            start = m.start()
            next_chapter = _REAL_CHAPTER_HEADING_RE.search(markdown, pos=start + len(line))
            end = next_chapter.start() if next_chapter else len(markdown)
            return start, end
    return None


def clean(markdown: str) -> CleanResult:
    references_from_body = ""
    biblio_span = _find_bibliography_span(markdown)
    if biblio_span:
        start, end = biblio_span
        references_from_body = markdown[start:end]
        markdown = markdown[:start] + markdown[end:]

    matches = list(HEADING_RE.finditer(markdown))

    segments: list[tuple[str | None, str]] = []
    if matches:
        if matches[0].start() > 0:
            segments.append((None, markdown[: matches[0].start()]))
        for idx, m in enumerate(matches):
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(markdown)
            segments.append((m.group(1).strip(), markdown[m.start():end]))
    else:
        segments.append((None, markdown))

    kept: list[str] = []
    references_parts: list[str] = []

    skip_until_depth: int | None = None
    in_bibliography = False
    for heading_text, block in segments:
        if heading_text is None:
            kept.append(_strip_toc_tables(block))
            continue

        depth = _heading_depth(heading_text)

        if skip_until_depth is not None:
            if depth <= skip_until_depth:
                skip_until_depth = None
                in_bibliography = False
            else:
                if in_bibliography:
                    references_parts.append(block)
                continue

        if _is_boilerplate_heading(heading_text, BOILERPLATE_HEADING_PATTERNS):
            skip_until_depth = depth
            in_bibliography = _is_boilerplate_heading(heading_text, BIBLIOGRAPHY_HEADING_PATTERNS)
            if in_bibliography:
                references_parts.append(block)
            continue

        kept.append(_strip_toc_tables(block))

    all_references = "\n".join(references_parts)
    if references_from_body:
        all_references = f"{all_references}\n{references_from_body}" if all_references else references_from_body

    return CleanResult(
        cleaned_markdown="\n".join(kept),
        references_text=all_references,
    )
