"""Per-section summary, purely extractive -- no LLM call. A prior version
used a local Qwen2.5-1.5B model to generate one summary per leaf section
(~150-300+ sections per document), which was CPU-bound and too expensive to
run at the ~10-15 guideline / 4-5 PDFs each target scale. Guideline-level
description (title, purpose) already comes straight from the AWMF sidecar
.txt files (guideline_schema.py/build_document.py), not from this module --
this only ever covered the section-level summary chunk used for coarse-to-
fine retrieval, and a truncated extract of the section's own text serves
that role adequately without any model compute.
"""

from __future__ import annotations

from .chunker import count_tokens

SHORT_SECTION_TOKEN_THRESHOLD = 150  # below this, the section text itself is used as its own "summary"
MAX_SUMMARY_CHARS = 500  # ~100-130 tokens, roughly matching the old model's MAX_SUMMARY_TOKENS budget


def summarize_section(heading: str, text: str) -> str:
    text = text.strip()
    if not text:
        return heading

    if count_tokens(text) < SHORT_SECTION_TOKEN_THRESHOLD:
        return text

    truncated = text[:MAX_SUMMARY_CHARS].strip()
    # Trim to the last full sentence/clause boundary so the extract doesn't
    # end mid-word -- falls back to a hard cut only if no boundary is found.
    for boundary in (". ", ".\n", "; "):
        idx = truncated.rfind(boundary)
        if idx > MAX_SUMMARY_CHARS * 0.5:  # don't over-trim a short lead sentence
            return truncated[:idx + 1].strip()
    return truncated.rstrip() + "..."
