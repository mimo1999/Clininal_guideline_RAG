"""Document-level router -- one level below guideline_router.py. Within the
guideline(s) already selected, scores each candidate *document* (Langfassung,
Kurzfassung, Methodenreport, Patientenversion, English, ...) against the
query using its own deterministic router text (chunking/router_text.py:
doc_type/title + first paragraph + its own section titles).

Same threshold+margin mechanism as the guideline router, but degrades
gracefully instead of refusing: if nothing clears the document-level
threshold, fall back to searching all documents in the selected guideline(s)
-- a query can be guideline-appropriate without cleanly matching one
document's particular style (e.g. a query that's topically clear but
doesn't lean toward "simple explanation" or "methodology detail" specifically).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .embed import EMBED_DIM, embed_query, embed_texts

PROCESSED_DIR = Path(__file__).parent.parent / "data_corpus" / "processed"

ABSOLUTE_THRESHOLD = 0.3
RELATIVE_MARGIN = 0.1


@dataclass
class DocumentCandidate:
    doc_id: str
    doc_type: str | None
    score: float


def _load_document_texts(guideline_ids: tuple[str, ...]) -> dict[str, dict]:
    docs: dict[str, dict] = {}
    if not PROCESSED_DIR.exists():
        return docs
    guideline_set = set(guideline_ids)
    for d in PROCESSED_DIR.iterdir():
        if not d.is_dir() or d.name.startswith("_guideline_"):
            continue
        meta_path, router_path = d / "metadata.json", d / "router_text.txt"
        if not (meta_path.exists() and router_path.exists()):
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("awmf_register_number") not in guideline_set:
            continue
        router_text = router_path.read_text(encoding="utf-8").strip()
        if not router_text:
            continue
        docs[d.name] = {
            "guideline_id": meta.get("awmf_register_number"),
            "doc_type": meta.get("doc_type"),
            "router_text": router_text,
        }
    return docs


@lru_cache(maxsize=32)
def _document_embeddings(guideline_ids: tuple[str, ...]):
    docs = _load_document_texts(guideline_ids)
    doc_ids = list(docs.keys())
    texts = [docs[did]["router_text"] for did in doc_ids]
    embeddings = embed_texts(texts) if texts else np.empty((0, EMBED_DIM))
    return doc_ids, embeddings, docs


def clear_cache() -> None:
    _document_embeddings.cache_clear()


def route(
    query: str,
    guideline_ids: list[str],
    absolute_threshold: float = ABSOLUTE_THRESHOLD,
    relative_margin: float = RELATIVE_MARGIN,
) -> list[DocumentCandidate]:
    """Returns the selected document candidates. Empty list means "no
    document-level preference" -- caller should fall back to searching every
    document in the given guidelines, not treat this as a refusal."""
    doc_ids, embeddings, docs = _document_embeddings(tuple(sorted(guideline_ids)))
    if not doc_ids:
        return []

    q_emb = embed_query(query)
    scores = embeddings @ q_emb
    order = np.argsort(-scores)
    best_score = float(scores[order[0]])

    if best_score < absolute_threshold:
        return []  # graceful fallback -- not a refusal

    selected = []
    for idx in order:
        score = float(scores[idx])
        if score < absolute_threshold or (best_score - score) > relative_margin:
            break
        did = doc_ids[idx]
        selected.append(DocumentCandidate(doc_id=did, doc_type=docs[did]["doc_type"], score=score))
    return selected
