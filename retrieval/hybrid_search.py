"""Orchestrates the full layered retrieval pipeline:

    Guideline router -> Document router -> Scoped Dense + BM25 -> RRF fusion
    -> Cross-encoder rerank (+ language bonus, + source-priority dedup)
    -> Dynamic window expansion

The guideline router is the primary refusal gate (in-domain/out-of-domain --
subsumes the old standalone topic_gate.py, deleted). The document router
narrows further within the selected guideline(s) but degrades gracefully
(falls back to searching every document in-guideline) rather than refusing --
see retrieval/document_router.py. A downstream low-rerank-confidence check is
a second, independent refusal trigger (defense in depth).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ingestion.langid import detect_language

from . import document_router, guideline_router
from .embed import embed_query, embed_texts_for_dedup
from .index_store import _tokenize, build_indexes
from .rerank import rerank as cross_encoder_rerank

RRF_K = 60
# Widened from 20 -- confirmed too narrow via a real miss: Q3's expected
# section (10.6) never appeared anywhere in a 20-candidate dense+sparse fetch
# against the 5,833-chunk corpus (~0.34% of it), so no amount of reranking
# could have surfaced it -- the candidate was never in the pool to begin with.
# A cross-encoder reranker is specifically good at picking the true best out
# of a wide, noisy net; a narrow initial fetch defeats that. ~3x the final
# generator count (RERANK_TOP_K) is the target ratio.
DENSE_TOP_K = 40
SPARSE_TOP_K = 40
# Narrowed from 40 -> 25 after the granite-embedding-278m-multilingual swap
# (see embed.py's MODEL_NAME comment / dev_logs.md Entry 10): a live
# fused+rerank funnel test against all 18 silver questions, using the REAL
# BM25 index and the REAL reranker, found the worst-case fused rank was 18
# (bge-m3 previously had a total miss at this stage, plus a rank-20
# straggler) -- 25 keeps a real safety margin over that observed max on a
# 30-question sample, not a value tuned to exactly match it.
FUSED_TOP_K = 25
# Narrowed from 10 -> 8 alongside the same granite swap -- the funnel test's
# post-rerank ranks for previously-hard cross-lingual questions all landed at
# 1-4, so the extra headroom the wider window was compensating for isn't
# needed as urgently; still deduped (see DEDUP_SIMILARITY_THRESHOLD below).
RERANK_TOP_K = 8
# The cross-encoder actually reranks this many candidates (used for the
# Recall@10/NDCG@10 evaluation metrics AND as the pre-dedup pool RERANK_TOP_K
# is drawn from -- must stay >= RERANK_TOP_K plus some slack for whatever
# dedup prunes); only the first RERANK_TOP_K of the resulting order are kept
# in `chunks` and fed to the generator.
RERANK_DIAGNOSTIC_K = 15
# Provisional, set from a handful of manual probes (see dev_logs.md) -- the
# cross-encoder's top score after reranking is what actually separates
# in-domain from out-of-domain queries cleanly (traps scored ~0.001-0.03 vs
# 0.85-0.99 for real answers), tighter than either router's own threshold.
RERANK_CONFIDENCE_THRESHOLD = 0.1
# Single retrieval pass, not a hard language filter -- a chunk in the query's
# detected language gets a flat score bump at rerank time so it wins ties
# against an equally-relevant chunk in another language, but a dramatically
# better cross-language match can still outrank it.
LANGUAGE_BONUS = 0.15
# Two chunks whose own dense embeddings cosine-similarity clears this are
# treated as near-duplicate restatements of the same fact (e.g. a Langfassung
# passage and its near-identical Kurzfassung counterpart, or a translated
# passage restating the same fact in another language). Language match is
# checked first: a same-query-language duplicate is always kept over a
# cross-language one, even if the cross-language copy has higher source
# authority -- the language bonus already pushed same-language matches ahead
# at rerank time, and letting dedup silently override that with an authority
# tie-break (the previous behavior) undermined it whenever a higher-authority
# document happened to also be in the "wrong" language. Only when both
# candidates agree on language-match status does source_priority arbitrate.
DEDUP_SIMILARITY_THRESHOLD = 0.92


def _release_gpu_memory() -> None:
    """Called after the heavy rerank/dedup stages -- a live server keeps
    calling search() in the SAME long-lived process (unlike the earlier
    isolated single-call benchmarks each run in a fresh process), and
    without this, PyTorch's caching allocator can leave enough
    reserved-but-unused memory behind each call that a 4GB card runs
    increasingly short of contiguous free space, making later calls
    progressively slower even with no leak in the literal sense -- the same
    class of issue diagnosed for bge-m3's .half() cast earlier (see
    dev_logs.md), just from repeated calls this time instead of one cast."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


@dataclass
class SearchResult:
    refused: bool
    refusal_reason: str | None
    chunks: list[dict] = field(default_factory=list)
    topic_gate_similarity: float | None = None  # best guideline_router score, field name kept for eval-harness compatibility
    routed_guideline_ids: list[str] = field(default_factory=list)
    routed_doc_ids: list[str] = field(default_factory=list)
    # Diagnostic fields for the evaluation harness's Recall@k ablation
    # (pre-rerank vs. post-rerank, before window expansion pads the count).
    fused_chunk_ids: list[str] = field(default_factory=list)
    reranked_chunk_ids: list[str] = field(default_factory=list)
    # Per-stage wall-clock seconds -- added specifically to answer "which
    # step took how long" without guessing; populated unconditionally (not
    # just in a debug mode) since the cost of a few time.perf_counter() calls
    # is negligible next to the stages being measured.
    timings: dict[str, float] = field(default_factory=dict)


class HybridSearcher:
    def __init__(self, force_rebuild: bool = False):
        self.collection, self.bm25, self.chunk_ids, self.chunks = build_indexes(force_rebuild=force_rebuild)
        self._chunk_index = {cid: i for i, cid in enumerate(self.chunk_ids)}

    def _dense_search(self, query: str, top_k: int, where: dict | None) -> list[str]:
        q_emb = embed_query(query)
        result = self.collection.query(query_embeddings=[q_emb.tolist()], n_results=top_k, where=where)
        return result["ids"][0] if result["ids"] else []

    def _sparse_search(self, query: str, top_k: int, allowed_ids: set[str] | None) -> list[str]:
        if not self.chunk_ids:
            return []
        tokens = _tokenize(query)
        scores = self.bm25.get_scores(tokens)
        indices = range(len(scores)) if allowed_ids is None else (
            self._chunk_index[cid] for cid in allowed_ids if cid in self._chunk_index
        )
        ranked = sorted(indices, key=lambda i: scores[i], reverse=True)[:top_k]
        return [self.chunk_ids[i] for i in ranked]

    @staticmethod
    def _rrf_fuse(dense_ids: list[str], sparse_ids: list[str], k: int = RRF_K) -> list[str]:
        scores: dict[str, float] = {}
        for rank, cid in enumerate(dense_ids):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
        for rank, cid in enumerate(sparse_ids):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
        return sorted(scores, key=lambda cid: scores[cid], reverse=True)

    def _apply_language_bonus(self, candidates: list[dict], query_language: str) -> list[dict]:
        if query_language == "unknown":
            return candidates
        adjusted = [
            {**c, "rerank_score": c["rerank_score"] + (LANGUAGE_BONUS if c.get("doc_language") == query_language else 0.0)}
            for c in candidates
        ]
        adjusted.sort(key=lambda c: c["rerank_score"], reverse=True)
        return adjusted

    def _dedup_by_source_priority(self, candidates: list[dict], query_language: str) -> list[dict]:
        """Suppress near-duplicate chunks (by content similarity). Language
        match is decided first (a same-query-language duplicate always wins
        over a cross-language one); source_priority only arbitrates when both
        candidates agree on language-match status. O(n^2) over a ~15-candidate
        window -- cheap."""
        if len(candidates) < 2:
            return candidates

        def _matches_lang(c: dict) -> bool:
            return query_language != "unknown" and c.get("doc_language") == query_language

        # Character-truncate (not model-level max_seq_length -- bge-m3 is a
        # SHARED, cached model instance also used at index time, so capping
        # it globally would silently change stored chunk embeddings and
        # require a full re-index) just the text fed into this ad-hoc
        # similarity check. A long outlier candidate (confirmed up to
        # ~900 tokens / 3600 chars against real retrieved candidates) forces
        # SentenceTransformer.encode()'s batch to pad every candidate to that
        # length -- measured as a real ~20-27s cost on a live server, not
        # hypothetical. Near-duplicate detection doesn't need full-chunk
        # fidelity; ~2000 chars (~500 tokens) is ample for that.
        dedup_texts = [(c["text"] or "")[:2000] for c in candidates]
        embeddings = embed_texts_for_dedup(dedup_texts)
        kept: list[dict] = []
        kept_embeddings: list[np.ndarray] = []
        for c, emb in zip(candidates, embeddings):
            dup_idx = None
            for i, kept_emb in enumerate(kept_embeddings):
                if float(emb @ kept_emb) > DEDUP_SIMILARITY_THRESHOLD:
                    dup_idx = i
                    break
            if dup_idx is None:
                kept.append(c)
                kept_embeddings.append(emb)
                continue

            existing = kept[dup_idx]
            c_matches = _matches_lang(c)
            existing_matches = _matches_lang(existing)
            if c_matches != existing_matches:
                replace = c_matches and not existing_matches
            else:
                replace = c.get("source_priority", 3) < existing.get("source_priority", 3)
            if replace:
                kept[dup_idx] = c
                kept_embeddings[dup_idx] = emb
        return kept

    def _expand_window(self, chunk_dicts: list[dict]) -> list[dict]:
        """Pull in a ±1 neighbor when a top result looks truncated at a
        section boundary -- catches recommendations whose conclusion spills
        into the next chunk (e.g. "...repeat after..." / "...12 months.")."""
        expanded = list(chunk_dicts)
        seen = {c["chunk_id"] for c in chunk_dicts}
        for c in chunk_dicts:
            text = (c.get("text") or "").strip()
            looks_truncated = bool(text) and (text[0].islower() or text[-1] not in ".!?\"'“”")
            if not looks_truncated:
                continue
            for neighbor_key in ("previous_chunk_id", "next_chunk_id"):
                nid = c.get(neighbor_key)
                if nid and nid not in seen and nid in self.chunks:
                    expanded.append(self.chunks[nid])
                    seen.add(nid)
        return expanded

    def search(self, query: str, top_k: int = RERANK_TOP_K) -> SearchResult:
        timings: dict[str, float] = {}
        t0 = time.perf_counter()

        in_domain, guideline_candidates = guideline_router.route(query)
        timings["guideline_router"] = time.perf_counter() - t0
        if not in_domain:
            return SearchResult(
                refused=True, refusal_reason="guideline_router", topic_gate_similarity=0.0, timings=timings,
            )

        guideline_ids = [c.guideline_id for c in guideline_candidates]
        where: dict | None = None
        doc_ids: list[str] = []
        allowed_ids: set[str] | None = None

        t1 = time.perf_counter()
        if guideline_ids:
            doc_candidates = document_router.route(query, guideline_ids)
            doc_ids = [c.doc_id for c in doc_candidates]
            if doc_ids:
                where = {"doc_id": {"$in": doc_ids}}
                allowed_ids = {cid for cid, c in self.chunks.items() if c.get("doc_id") in doc_ids}
            else:
                where = {"guideline_id": {"$in": guideline_ids}}
                allowed_ids = {cid for cid, c in self.chunks.items() if c.get("guideline_id") in guideline_ids}
        # else: nothing indexed under any guideline yet (guideline_router fails open) -- search unscoped
        timings["document_router"] = time.perf_counter() - t1

        best_guideline_score = max((c.score for c in guideline_candidates), default=0.0)

        t2 = time.perf_counter()
        dense_ids = self._dense_search(query, DENSE_TOP_K, where)
        timings["dense_search"] = time.perf_counter() - t2

        t3 = time.perf_counter()
        sparse_ids = self._sparse_search(query, SPARSE_TOP_K, allowed_ids)
        fused_ids = self._rrf_fuse(dense_ids, sparse_ids)[:FUSED_TOP_K]
        timings["sparse_search_and_fuse"] = time.perf_counter() - t3

        candidates = [self.chunks[cid] for cid in fused_ids if cid in self.chunks]
        if not candidates:
            return SearchResult(
                refused=True, refusal_reason="empty_retrieval", topic_gate_similarity=best_guideline_score,
                routed_guideline_ids=guideline_ids, routed_doc_ids=doc_ids, fused_chunk_ids=fused_ids,
                timings=timings,
            )

        t4 = time.perf_counter()
        reranked_full = cross_encoder_rerank(query, candidates, text_key="text", top_k=RERANK_DIAGNOSTIC_K)
        _release_gpu_memory()
        timings["rerank"] = time.perf_counter() - t4
        if not reranked_full or reranked_full[0].get("rerank_score", 0.0) < RERANK_CONFIDENCE_THRESHOLD:
            return SearchResult(
                refused=True, refusal_reason="low_rerank_confidence", topic_gate_similarity=best_guideline_score,
                routed_guideline_ids=guideline_ids, routed_doc_ids=doc_ids, fused_chunk_ids=fused_ids,
                reranked_chunk_ids=[c["chunk_id"] for c in reranked_full], timings=timings,
            )

        t5 = time.perf_counter()
        query_language = detect_language(query).language
        adjusted = self._apply_language_bonus(reranked_full, query_language)
        deduped = self._dedup_by_source_priority(adjusted, query_language)
        _release_gpu_memory()
        timings["language_bonus_and_dedup"] = time.perf_counter() - t5

        reranked_ids = [c["chunk_id"] for c in deduped]  # up to RERANK_DIAGNOSTIC_K, for Recall/NDCG@10
        reranked = deduped[:top_k]  # only top_k actually used for generation

        t6 = time.perf_counter()
        expanded = self._expand_window(reranked)
        timings["window_expand"] = time.perf_counter() - t6

        timings["total"] = time.perf_counter() - t0

        return SearchResult(
            refused=False, refusal_reason=None, chunks=expanded, topic_gate_similarity=best_guideline_score,
            routed_guideline_ids=guideline_ids, routed_doc_ids=doc_ids,
            fused_chunk_ids=fused_ids, reranked_chunk_ids=reranked_ids, timings=timings,
        )
