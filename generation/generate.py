"""Turns (question, retrieval.hybrid_search.SearchResult) into a grounded,
cited answer (in the query's own language -- German or English, see
prompt.py) or a refusal.

Three refusal paths, all reported uniformly via `refusal_reason`:
- "guideline_router" / "empty_retrieval" / "low_rerank_confidence" -- from
  hybrid_search, short-circuits before the LLM is even called.
- "llm_grounding_refusal" -- retrieval succeeded but the LLM itself
  determined the given context doesn't answer the specific question, and
  emitted the fixed REFUSAL_STRING as instructed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .llm import generate as llm_generate
from .prompt import (
    REFUSAL_STRINGS,
    build_context_block,
    build_messages,
    citation_source_pattern,
    detect_query_language,
    sources_label,
)


@dataclass
class Answer:
    text: str
    citations: list[str] = field(default_factory=list)
    refused: bool = False
    refusal_reason: str | None = None


def _looks_like_refusal(text: str, refusal_string: str) -> bool:
    return refusal_string.lower() in text.strip().lower()


def answer_question(question: str, search_result, max_new_tokens: int = 3000) -> Answer:
    language = detect_query_language(question)
    refusal_string = REFUSAL_STRINGS[language]

    if search_result.refused:
        return Answer(text=refusal_string, refused=True, refusal_reason=search_result.refusal_reason)

    chunks = search_result.chunks
    _, tags = build_context_block(chunks, language)
    messages = build_messages(question, chunks, language)

    raw = llm_generate(messages, max_new_tokens=max_new_tokens)

    if _looks_like_refusal(raw, refusal_string):
        return Answer(text=refusal_string, refused=True, refusal_reason="llm_grounding_refusal")

    used_indices = sorted(set(int(m) for m in citation_source_pattern(language).findall(raw)))
    used_tags = [tags[i - 1] for i in used_indices if 1 <= i <= len(tags)]

    # The small local model doesn't reliably emit inline "(Quelle N)" tags
    # despite the instruction (observed in testing) -- falling back to citing
    # every chunk actually given to the model is more honest than silently
    # dropping citations, since those chunks were already the deterministic
    # output of retrieval, not a guess.
    if not used_tags:
        used_tags = tags

    answer_text = raw
    if used_tags:
        answer_text = f"{raw}\n\n{sources_label(language)}:\n" + "\n".join(used_tags)

    return Answer(text=answer_text, citations=used_tags, refused=False, refusal_reason=None)
