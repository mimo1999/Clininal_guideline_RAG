"""Guideline-level router -- replaces the old standalone topic_gate.py. The
router is already a semantic classifier over the whole corpus, so a separate
topic gate was redundant; its refusal job is now done here (see `route`'s
absolute_threshold check).

Scores each guideline against the query via 3 SEPARATE embeddings (title,
purpose statement, pooled section-titles summary) rather than one blended
embedding -- a query like "CIN III follow-up" may never appear in a title or
purpose statement, but could match a section title directly, so collapsing
everything into one embedding risks missing that. Combined via a weighted
score.

Selection requires BOTH `score > absolute_threshold` AND `score within
relative_margin of the best score` -- margin alone would route to the
"least bad" guideline even when every score is mediocre (i.e. an
out-of-domain query), so the absolute floor is what actually gates refusal;
the margin just allows more than one guideline through when several are
genuinely close.

Router text is built deterministically (chunking/router_text.py) from the
already-parsed structure tree -- no LLM, reproducible.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .embed import EMBED_DIM, embed_query, embed_texts

PROCESSED_DIR = Path(__file__).parent.parent / "data_corpus" / "processed"

# Section-titles summary weighted highest -- most specific/content-dense
# signal. Title and purpose lower but still contribute (a query may match
# the guideline's stated purpose without matching any single section title).
WEIGHTS = {"title": 0.2, "purpose": 0.2, "summary": 0.6}

# Candidate guidelines are selected using a similarity threshold with a
# relative margin to the highest-scoring guideline. This avoids fixed top-k
# routing while still allowing multiple closely related guidelines to be
# searched when appropriate.
#
# ABSOLUTE_THRESHOLD calibrated against real score data from all 12 of the
# brief's questions on the actual 2-guideline corpus (see dev_logs.md Entry
# 5): real questions' BEST-guideline score ranged 0.472-0.579, traps' best
# score 0.388-0.458. Raising this to 0.42 looked safe from the best-score
# view alone, but broke real recall: Q2's second guideline (015-027OL, which
# actually holds the expected section) scored below the new floor/margin band
# and got excluded outright -- hit@5 went from True to False, confirmed via a
# direct per-question recall check, not assumed. The per-guideline score that
# actually matters for recall isn't visible in the single "best score" number
# a quick calibration pass looks at, so 0.35 (empirically already achieving
# refusal_correctness_rate=1.0 on the full evaluation run -- see Entry 5) is
# kept rather than traded for a nice-to-have earlier refusal on 2 of 3 traps
# that are already caught correctly one layer downstream regardless.
ABSOLUTE_THRESHOLD = 0.35
RELATIVE_MARGIN = 0.08

_route_cache: dict[str, tuple[bool, list["GuidelineCandidate"]]] = {}


@dataclass
class GuidelineCandidate:
    guideline_id: str
    title: str | None
    score: float


def _load_guideline_texts() -> dict[str, dict[str, str]]:
    texts: dict[str, dict[str, str]] = {}
    if not PROCESSED_DIR.exists():
        return texts
    for d in PROCESSED_DIR.iterdir():
        if not d.is_dir() or not d.name.startswith("_guideline_"):
            continue
        guideline_id = d.name[len("_guideline_"):]
        meta_path = d / "guideline.json"
        summary_path = d / "section_titles_summary.txt"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        title = meta.get("title") or ""
        texts[guideline_id] = {
            "title": title,
            "purpose": meta.get("purpose_text") or "",
            "summary": summary_path.read_text(encoding="utf-8").strip() if summary_path.exists() else "",
        }
        # embed_texts can't take an empty string meaningfully -- fall back to
        # the title (never empty if the guideline was ingested at all) so
        # every component always has *something* to embed.
        for component in ("purpose", "summary"):
            if not texts[guideline_id][component]:
                texts[guideline_id][component] = title or guideline_id
    return texts


@lru_cache(maxsize=1)
def _guideline_embeddings():
    texts = _load_guideline_texts()
    guideline_ids = list(texts.keys())
    embeddings = {}
    for component in ("title", "purpose", "summary"):
        component_texts = [texts[gid][component] for gid in guideline_ids]
        embeddings[component] = embed_texts(component_texts) if component_texts else np.empty((0, EMBED_DIM))
    return guideline_ids, embeddings, texts


def clear_cache() -> None:
    """Call after re-ingesting/re-chunking so the router picks up new/changed
    guidelines instead of stale cached embeddings."""
    _guideline_embeddings.cache_clear()
    _route_cache.clear()


def route(
    query: str,
    absolute_threshold: float = ABSOLUTE_THRESHOLD,
    relative_margin: float = RELATIVE_MARGIN,
) -> tuple[bool, list[GuidelineCandidate]]:
    """Returns (in_domain, candidates). candidates is sorted by score desc,
    already filtered to the selected set. in_domain=False means refuse --
    this subsumes the old topic_gate's job."""
    cache_key = query.strip().lower()
    if cache_key in _route_cache:
        return _route_cache[cache_key]

    guideline_ids, embeddings, texts = _guideline_embeddings()
    if not guideline_ids:
        result = (True, [])  # nothing indexed yet -- fail open, downstream retrieval finds nothing anyway
        _route_cache[cache_key] = result
        return result

    q_emb = embed_query(query)
    scores = np.zeros(len(guideline_ids))
    for component, weight in WEIGHTS.items():
        sims = embeddings[component] @ q_emb
        scores = scores + weight * sims

    order = np.argsort(-scores)
    best_score = float(scores[order[0]])

    if best_score < absolute_threshold:
        result = (False, [])
        _route_cache[cache_key] = result
        return result

    selected = []
    for idx in order:
        score = float(scores[idx])
        if score < absolute_threshold or (best_score - score) > relative_margin:
            break
        gid = guideline_ids[idx]
        selected.append(GuidelineCandidate(guideline_id=gid, title=texts[gid]["title"], score=score))

    result = (True, selected)
    _route_cache[cache_key] = result
    return result
