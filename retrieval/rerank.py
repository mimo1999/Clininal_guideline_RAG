"""Cross-encoder reranking, local/offline. Inserted after RRF fusion
(dense+BM25 top-40 -> rerank -> top-10) -- per review feedback, the single
biggest lever available for retrieval quality on this corpus, and the
cross-encoder is a better judge of query-passage relevance than either
retriever's own similarity score."""

from __future__ import annotations

import os
from functools import lru_cache

from common.config import RERANK_MAX_LENGTH, RERANK_MODEL_NAME_DEFAULT

# RERANK_MAX_LENGTH below was tuned against bge-reranker-v2-m3's 8192-token
# native context (see that constant's own comment); mmarco-mMiniLMv2-L12 was
# trained at 512 tokens natively, so the cap is now a no-op ceiling rather
# than an active truncation for this model, but is kept explicit so a future
# swap back doesn't silently reintroduce the original unbounded-padding cost.
#
# Switched from BAAI/bge-reranker-v2-m3 (567M params, multilingual) to this
# bilingual DE-EN, ~110M-param model after a live 12-question A/B test (see
# dev_logs.md Entry 9): rerank time dropped ~33.8s -> ~3.2s (~10x) with the
# refusal/out-of-domain gate (RERANK_CONFIDENCE_THRESHOLD below) still intact
# at the same trap_refusal_rate as the old model. Traded off deliberately,
# not free: recall@3/@5 on the golden 12 questions dropped 0.667 -> 0.556
# (one fewer answerable question keeps its expected chunk in the top-5) --
# accepted explicitly by the user given the latency win, not a silent
# regression. A second lighter candidate tested in the same run,
# cross-encoder/msmarco-MiniLM-L6-en-de-v1, was REJECTED outright: it broke
# the refusal gate entirely (trap_refusal_rate 0.667 -> 0.0 -- it stopped
# refusing out-of-domain questions altogether), because RERANK_CONFIDENCE_
# THRESHOLD was calibrated against bge-reranker-v2-m3's specific score
# distribution and doesn't transfer to every model's score range. Any future
# reranker swap must re-verify trap_refusal_rate, not just recall/NDCG.
DEFAULT_MODEL_NAME = RERANK_MODEL_NAME_DEFAULT
# bge-reranker-v2-m3 supports up to 8192 tokens by default -- with no cap,
# a single long candidate (table chunks especially can run 500-900+ tokens,
# confirmed directly against real retrieved candidates) forces the WHOLE
# batch to pad to that length, since CrossEncoder.predict() tokenizes a
# batch to a common length. On a live server this measured as the dominant
# cost of a chat response (~46-62s of a ~70-95s total, generation itself
# only ~2-5s) -- not a memory/fragmentation issue (ruled out: an added
# empty_cache() call measurably freed VRAM between calls but left timing
# unchanged). 512 tokens covers the target chunk size (300-500) with margin;
# reranking only decides ORDER, so truncating the tail of an outlier-long
# candidate for scoring purposes doesn't affect what's actually shown to the
# user or sent to the generator afterward -- that still uses the full text.
# (RERANK_MAX_LENGTH itself imported from common.config above.)


@lru_cache(maxsize=2)
def _model(model_name: str):
    import torch
    from sentence_transformers import CrossEncoder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        return CrossEncoder(model_name, device=device, max_length=RERANK_MAX_LENGTH)

    # Release whatever bge-m3's own load left reserved-but-unused before
    # requesting this model's block -- on a 4GB card, headroom is tight
    # enough that leftover allocator fragmentation from the first load can
    # starve the second (see embed.py's matching empty_cache() call).
    torch.cuda.empty_cache()
    # fp16 via model_kwargs at load time -- NOT a manual .half() cast after
    # load, which was tried previously and broke CrossEncoder's internal
    # forward wiring (BatchEncoding vs tensor confusion, per the old comment
    # here). Loading natively in fp16 avoids that: confirmed working
    # (valid scores) and drops reserved VRAM from ~2.2-2.7GB (fp32) to
    # ~1.15GB. That was necessary, not just nice-to-have -- fp32 reliably
    # produced std::bad_alloc-style CUDA OOMs on this 4GB card once bge-m3
    # was also loaded (confirmed reproducible from a clean GPU, twice).
    return CrossEncoder(
        model_name, device=device, max_length=RERANK_MAX_LENGTH,
        model_kwargs={"torch_dtype": torch.float16},
    )


def rerank(query: str, candidates: list[dict], text_key: str = "text", top_k: int = 5) -> list[dict]:
    """candidates: list of dicts each containing at least `text_key`. Returns
    the same dicts, sorted by cross-encoder score desc, truncated to top_k,
    with a `rerank_score` field added.

    Reads RERANK_MODEL_NAME from the environment on every call (not bound to
    a module-level constant at import time) -- a prior version bound it once
    at import, which meant setting the env var afterward (e.g. from an A/B
    test driver script) silently had no effect, since retrieval/hybrid_search.py
    imports this module long before any test script gets a chance to set the
    var. Cheap to re-read per call; correctness matters more here than saving
    a dict.get()."""
    if not candidates:
        return []
    model_name = os.environ.get("RERANK_MODEL_NAME", DEFAULT_MODEL_NAME)
    pairs = [(query, c[text_key]) for c in candidates]
    scores = _model(model_name).predict(pairs)
    scored = [{**c, "rerank_score": float(s)} for c, s in zip(candidates, scores)]
    scored.sort(key=lambda c: c["rerank_score"], reverse=True)
    return scored[:top_k]
