"""Orchestrator: data_corpus/processed/<doc_id>/{parsed.md, metadata.json}
-> clean -> structure extract -> chunk (text/recommendation/table/summary)
-> data_corpus/processed/<doc_id>/chunks.jsonl (+ references.txt, router_text.txt).

Also rolls up one guideline-level router text per guideline (see
build_chunks_for_guideline), pooling section titles across all of a
guideline's documents -- used by retrieval/guideline_router.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable, Optional

from common.progress import ProgressTracker

from .chunker import chunk_section, count_tokens
from .clean import clean
from .router_text import build_document_router_text, build_guideline_section_titles_summary
from .schema import Chunk
from .structure import build_sections, leaf_sections
from .summarize import summarize_section

PROCESSED_DIR = Path(__file__).parent.parent / "data_corpus" / "processed"


def _load_doc_metadata(doc_dir: Path) -> dict:
    return json.loads((doc_dir / "metadata.json").read_text(encoding="utf-8"))


def _load_guideline_metadata(guideline_id: str) -> dict:
    path = PROCESSED_DIR / f"_guideline_{guideline_id}" / "guideline.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def build_chunks(doc_id: str) -> Path:
    doc_dir = PROCESSED_DIR / doc_id
    markdown = (doc_dir / "parsed.md").read_text(encoding="utf-8")
    doc_meta = _load_doc_metadata(doc_dir)
    guideline_id = doc_meta.get("awmf_register_number")
    guideline_meta = _load_guideline_metadata(guideline_id) if guideline_id else {}

    clean_result = clean(markdown)
    if clean_result.references_text:
        (doc_dir / "references.txt").write_text(clean_result.references_text, encoding="utf-8")

    sections = build_sections(clean_result.cleaned_markdown)
    leaves = leaf_sections(sections)

    denorm = dict(
        doc_title=doc_meta.get("title"),
        doc_type=doc_meta.get("doc_type"),
        source_priority=doc_meta.get("source_priority", 3),
        awmf_register_number=guideline_id,
        doc_version=doc_meta.get("version"),
        doc_language=doc_meta.get("language"),
        publishing_organization=doc_meta.get("publishing_organization"),
        awmf_stand_date=doc_meta.get("awmf_stand_date"),
        guideline_id=guideline_id,
        guideline_title=guideline_meta.get("title"),
    )

    router_text = build_document_router_text(doc_meta.get("title"), doc_meta.get("doc_type"), leaves)
    (doc_dir / "router_text.txt").write_text(router_text, encoding="utf-8")

    chunks: list[Chunk] = []
    rec_counter = 0

    for sec_idx, section in enumerate(leaves):
        text_pieces, table_chunks = chunk_section(section)
        section_id = section.section_id

        summary_input = "\n\n".join(p.text for p in text_pieces) or section.section_title
        summary_text = summarize_section(section.section_title, summary_input)
        summary_chunk_id = f"{doc_id}::sec-{sec_idx}::summary"
        chunks.append(Chunk(
            chunk_id=summary_chunk_id, doc_id=doc_id, chunk_type="section_summary",
            text=summary_text, heading_path=section.heading_path,
            section_number=section.section_number, section_title=section.section_title,
            section_id=section_id, token_count=count_tokens(summary_text),
            topic=section.topics, **denorm,
        ))

        section_chunk_indices: list[int] = []

        for i, piece in enumerate(text_pieces):
            chunks.append(Chunk(
                chunk_id=f"{doc_id}::sec-{sec_idx}::text-{i}", doc_id=doc_id, chunk_type="text",
                text=piece.text, heading_path=section.heading_path,
                section_number=section.section_number, section_title=section.section_title,
                section_id=section_id, token_count=piece.token_count, topic=section.topics,
                parent_chunk_id=summary_chunk_id, **denorm,
            ))
            section_chunk_indices.append(len(chunks) - 1)

        for i, rec in enumerate(section.recommendations):
            rec_counter += 1
            chunks.append(Chunk(
                chunk_id=f"{doc_id}::sec-{sec_idx}::rec-{i}", doc_id=doc_id, chunk_type="text",
                text=rec.text, heading_path=section.heading_path,
                section_number=section.section_number, section_title=section.section_title,
                section_id=section_id, token_count=count_tokens(rec.text), topic=section.topics,
                recommendation_id=f"R{rec_counter}", evidence_grade=rec.evidence_grade,
                evidence_grade_source="regex", parent_chunk_id=summary_chunk_id, **denorm,
            ))
            section_chunk_indices.append(len(chunks) - 1)

        for i, table in enumerate(table_chunks):
            chunks.append(Chunk(
                chunk_id=f"{doc_id}::sec-{sec_idx}::table-{i}", doc_id=doc_id, chunk_type="table",
                text=table.embedded_text, markdown_table=table.markdown_table,
                flattened_table_text=table.flattened_text, heading_path=section.heading_path,
                section_number=section.section_number, section_title=section.section_title,
                section_id=section_id, token_count=count_tokens(table.embedded_text),
                topic=section.topics, parent_chunk_id=summary_chunk_id, **denorm,
            ))
            section_chunk_indices.append(len(chunks) - 1)

        for pos, idx in enumerate(section_chunk_indices):
            if pos > 0:
                chunks[idx].previous_chunk_id = chunks[section_chunk_indices[pos - 1]].chunk_id
            if pos < len(section_chunk_indices) - 1:
                chunks[idx].next_chunk_id = chunks[section_chunk_indices[pos + 1]].chunk_id

    out_path = doc_dir / "chunks.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(c.model_dump_json() + "\n")

    return out_path


def _guideline_ids_present() -> list[str]:
    ids = set()
    for doc_dir in PROCESSED_DIR.iterdir():
        if not doc_dir.is_dir() or doc_dir.name.startswith("_guideline_"):
            continue
        meta_path = doc_dir / "metadata.json"
        if not meta_path.exists():
            continue
        reg_num = _load_doc_metadata(doc_dir).get("awmf_register_number")
        if reg_num:
            ids.add(reg_num)
    return sorted(ids)


def _chunks_up_to_date(doc_dir: Path) -> bool:
    """True if chunks.jsonl already exists and is at least as new as the
    parsed.md it's derived from -- i.e. this document was already chunked
    since its last (re-)ingestion, so re-chunking it now would just
    reproduce the same output. mtime-based, not a content hash (unlike
    ingestion/manifest.py's PDF-level check): parsed.md is only ever
    rewritten by build_document() when a PDF is actually re-ingested (see
    ingestion/manifest.py), so its mtime alone is a reliable signal here --
    no need for a second, separately-maintained manifest at this layer."""
    chunks_path = doc_dir / "chunks.jsonl"
    parsed_path = doc_dir / "parsed.md"
    return chunks_path.exists() and parsed_path.exists() and chunks_path.stat().st_mtime >= parsed_path.stat().st_mtime


def build_chunks_for_guideline(
    guideline_id: str, on_doc_done: Optional[Callable[[str], None]] = None, force: bool = False,
) -> list[Path]:
    doc_dirs = [
        d for d in sorted(PROCESSED_DIR.iterdir())
        if d.is_dir() and not d.name.startswith("_guideline_")
        and (d / "metadata.json").exists()
        and _load_doc_metadata(d).get("awmf_register_number") == guideline_id
    ]

    outputs = []
    for d in doc_dirs:
        if not force and _chunks_up_to_date(d):
            outputs.append(d / "chunks.jsonl")
        else:
            outputs.append(build_chunks(d.name))
        if on_doc_done:
            on_doc_done(d.name)

    guideline_meta = _load_guideline_metadata(guideline_id)
    all_titles: list[str] = []
    for out_path in outputs:
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            c = json.loads(line)
            if c.get("chunk_type") == "section_summary" and c.get("section_title"):
                all_titles.append(c["section_title"])

    section_titles_summary = build_guideline_section_titles_summary(all_titles)
    guideline_dir = PROCESSED_DIR / f"_guideline_{guideline_id}"
    guideline_dir.mkdir(parents=True, exist_ok=True)
    # title/purpose_text are already in guideline.json -- this file is just
    # the pooled-section-titles component of the 3-way router embedding.
    (guideline_dir / "section_titles_summary.txt").write_text(section_titles_summary, encoding="utf-8")

    return outputs


def _total_doc_count() -> int:
    count = 0
    for d in PROCESSED_DIR.iterdir():
        if d.is_dir() and not d.name.startswith("_guideline_") and (d / "metadata.json").exists():
            count += 1
    return count


def build_chunks_for_all(force: bool = False) -> list[Path]:
    tracker = ProgressTracker("chunk_build", total=_total_doc_count(), stage="starting")
    outputs = []
    try:
        for guideline_id in _guideline_ids_present():
            print(f"Chunking guideline: {guideline_id}")
            tracker.set_stage(f"guideline:{guideline_id}")

            def _on_doc_done(doc_id: str, g: str = guideline_id) -> None:
                tracker.set_stage(f"guideline:{g} doc:{doc_id}")
                tracker.advance(1)

            outputs.extend(build_chunks_for_guideline(guideline_id, on_doc_done=_on_doc_done, force=force))
    except Exception as e:
        tracker.finish(error=str(e))
        raise
    tracker.finish()
    return outputs


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] != "--force":
        print(build_chunks(sys.argv[1]))
    else:
        for p in build_chunks_for_all(force="--force" in sys.argv):
            print(p)
