"""Central registry of the retrieval/chunking funnel's tunable hyperparameters
-- one place to see every knob and override any of them (via environment
variable, same mechanism this project already used for RERANK_MODEL_NAME/
DEDUP_MODEL_NAME before this module existed) for an A/B test or sweep,
without editing source across half a dozen files.

Deliberately NOT where the reasoning for each value lives -- that stays as an
inline comment in the module that actually uses the constant (e.g.
retrieval/hybrid_search.py's RRF_K import), since that's where a future
reader looks when asking "why is this value what it is." This module only
answers "where do I change it" and "what can I override at runtime."

Each constant is read once at import time (env var -> typed value), matching
how nearly every module in this codebase already declares its tunables as
plain module-level constants. The one existing exception -- rerank.py's
RERANK_MODEL_NAME, deliberately re-read from os.environ on every call because
an A/B driver script sets it after retrieval/hybrid_search.py has already
imported rerank.py -- is unchanged by this module; it still reads directly
from os.environ at call time, not from a constant defined here.
"""

from __future__ import annotations

import os


def _float(env_var: str, default: float) -> float:
    return float(os.environ.get(env_var, default))


def _int(env_var: str, default: int) -> int:
    return int(os.environ.get(env_var, default))


def _str(env_var: str, default: str) -> str:
    return os.environ.get(env_var, default)


# --- chunking/chunker.py -----------------------------------------------
TARGET_MIN_TOKENS = _int("TARGET_MIN_TOKENS", 300)
TARGET_MAX_TOKENS = _int("TARGET_MAX_TOKENS", 500)
HARD_CAP_TOKENS = _int("HARD_CAP_TOKENS", 600)
OVERLAP_RATIO = _float("OVERLAP_RATIO", 0.15)
MERGE_THRESHOLD_TOKENS = _int("MERGE_THRESHOLD_TOKENS", 100)

# --- retrieval/embed.py -------------------------------------------------
# EMBED_MODEL_NAME is NOT freely swappable via env var like the others below
# -- changing it changes the embedding space itself and requires a full
# corpus re-index (see embed.py's own MODEL_NAME comment), unlike a rerank/
# dedup model swap which costs nothing to try. Centralized here anyway so
# it's visible alongside everything else, but treat it as a source-level
# change (also update EMBED_DIM in retrieval/embed.py to match), not a
# runtime experiment.
EMBED_MODEL_NAME = _str("EMBED_MODEL_NAME", "ibm-granite/granite-embedding-278m-multilingual")
EMBED_MAX_SEQ_LENGTH = _int("EMBED_MAX_SEQ_LENGTH", 512)
DEDUP_MODEL_NAME_DEFAULT = _str("DEDUP_MODEL_NAME", "intfloat/multilingual-e5-small")

# --- retrieval/rerank.py -------------------------------------------------
RERANK_MODEL_NAME_DEFAULT = _str("RERANK_MODEL_NAME", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
RERANK_MAX_LENGTH = _int("RERANK_MAX_LENGTH", 512)

# --- retrieval/hybrid_search.py ------------------------------------------
RRF_K = _int("RRF_K", 60)
DENSE_TOP_K = _int("DENSE_TOP_K", 40)
# BM25 is nearly free computationally; a wide sparse net helps recover exact
# medical term matches (e.g. "Kolposkopie", "CIN 3+") that dense retrieval may miss.
SPARSE_TOP_K = _int("SPARSE_TOP_K", 60)
# Limits the reranker's input to a multiple of its expected output,
# avoiding too much noise while still capturing enough context.
RERANK_TOP_K = _int("RERANK_TOP_K", 5)
FUSED_TOP_K = _int("FUSED_TOP_K", RERANK_TOP_K * 4)
RERANK_DIAGNOSTIC_K = _int("RERANK_DIAGNOSTIC_K", 15)
RERANK_CONFIDENCE_THRESHOLD = _float("RERANK_CONFIDENCE_THRESHOLD", 0.1)
LANGUAGE_BONUS = _float("LANGUAGE_BONUS", 0.15)
DEDUP_SIMILARITY_THRESHOLD = _float("DEDUP_SIMILARITY_THRESHOLD", 0.92)

# --- retrieval/guideline_router.py ---------------------------------------
GUIDELINE_ABSOLUTE_THRESHOLD = _float("GUIDELINE_ABSOLUTE_THRESHOLD", 0.35)
GUIDELINE_RELATIVE_MARGIN = _float("GUIDELINE_RELATIVE_MARGIN", 0.08)
# A normalized 3-way weight triple, not a single scalar -- no env-var
# override (a sweep would need to vary all three together and renormalize,
# which doesn't fit the single-value _float()/_int() pattern above). Change
# directly here if experimenting with these.
# Title is high-precision when it matches (e.g. query contains "Zervixkarzinom"
# which is literally the guideline's title).
GUIDELINE_ROUTER_WEIGHTS = {"title": 0.3, "purpose": 0.2, "summary": 0.5}

# --- retrieval/document_router.py ----------------------------------------
# A higher threshold means it falls back to "search all documents in this guideline"
# more often, which is the safe behavior per the docstring (graceful fallback,
# rather than risking routing to a Kurzfassung/Methodenreport that lacks the expected chunk).
DOCUMENT_ABSOLUTE_THRESHOLD = _float("DOCUMENT_ABSOLUTE_THRESHOLD", 0.45)
DOCUMENT_RELATIVE_MARGIN = _float("DOCUMENT_RELATIVE_MARGIN", 0.1)
