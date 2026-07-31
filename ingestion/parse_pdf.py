"""Docling-based PDF parsing: structured Markdown + first-page text."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption


@dataclass
class ParsedPdf:
    markdown: str
    first_page_text: str
    page_count: int


@lru_cache(maxsize=1)
def _converter() -> DocumentConverter:
    # OCR defaults to on in Docling, which loads and runs a torch-backed OCR
    # pass (RapidOCR) on every page -- expensive, and unnecessary for this
    # corpus: AWMF guideline PDFs are digitally-native with a real text
    # layer (confirmed -- first_page_text extraction has always worked
    # directly off the text layer, never needed OCR fallback). Disabling it
    # was the fix for repeated `std::bad_alloc` crashes mid-conversion on
    # this machine's available RAM -- see dev_logs.md Entry 5. Table
    # structure recognition stays on; that's load-bearing for chunking/tables.py.
    options = PdfPipelineOptions()
    options.do_ocr = False
    options.do_table_structure = True
    options.generate_page_images = False
    options.generate_picture_images = False
    return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)})


def parse_pdf(pdf_path: str | Path) -> ParsedPdf:
    pdf_path = Path(pdf_path)
    result = _converter().convert(str(pdf_path))
    doc = result.document

    markdown = doc.export_to_markdown()
    page_count = len(doc.pages) if doc.pages else 0

    first_page_text = _extract_first_page_text(doc)

    return ParsedPdf(markdown=markdown, first_page_text=first_page_text, page_count=page_count)


def parse_pdf_isolated(pdf_path: str | Path) -> ParsedPdf:
    """Runs parse_pdf() in a fresh subprocess per call. Docling's per-conversion
    memory (torch/C++ allocator arenas) doesn't fully return to the OS between
    conversions, so calling parse_pdf() repeatedly in one long-lived process
    accumulates resident memory across a multi-PDF batch until std::bad_alloc --
    confirmed by crashes happening progressively later (more PDFs converted
    first) across retries of the same batch. A fresh process per PDF gives each
    conversion a clean memory budget, reclaimed unconditionally on exit."""
    import os

    pdf_path = Path(pdf_path)
    fd, tmp_name = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    out_path = Path(tmp_name)
    try:
        subprocess.run(
            [sys.executable, "-m", "ingestion.parse_pdf", str(pdf_path), str(out_path)],
            check=True,
            cwd=str(Path(__file__).parent.parent),
        )
        data = json.loads(out_path.read_text(encoding="utf-8"))
        return ParsedPdf(**data)
    finally:
        out_path.unlink(missing_ok=True)


def _extract_first_page_text(doc) -> str:
    """Pull text confined to page 1 for cover-page metadata parsing."""
    parts = []
    for item, _level in doc.iterate_items():
        prov = getattr(item, "prov", None)
        if not prov:
            continue
        page_no = prov[0].page_no
        if page_no != 1:
            continue
        text = getattr(item, "text", None)
        if text:
            parts.append(text)
    if parts:
        return "\n".join(parts)
    # Fallback: first ~2000 chars of the full markdown if page-level provenance is unavailable.
    return doc.export_to_markdown()[:2000]


if __name__ == "__main__":
    # Entrypoint for parse_pdf_isolated()'s subprocess: parse one PDF, write
    # the result as JSON, exit -- the OS reclaims all memory on process exit.
    pdf_arg, out_arg = Path(sys.argv[1]), Path(sys.argv[2])
    result = parse_pdf(pdf_arg)
    out_arg.write_text(
        json.dumps({
            "markdown": result.markdown,
            "first_page_text": result.first_page_text,
            "page_count": result.page_count,
        }),
        encoding="utf-8",
    )
