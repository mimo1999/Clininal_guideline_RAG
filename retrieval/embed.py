"""Dense embedding, local/offline. Used both to embed chunks at index time
and to embed queries at retrieval time."""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np

from common.config import DEDUP_MODEL_NAME_DEFAULT, EMBED_MAX_SEQ_LENGTH, EMBED_MODEL_NAME

# Switched from BAAI/bge-m3 (567M params, 1024-dim, 8192-token context) to
# this smaller (278M params, 768-dim, 512-token-capped) model after a live
# silver-question comparison (dev_logs.md Entry 10): granite matched bge-m3
# on monolingual (de_de/en_en) retrieval and DECISIVELY beat it on the
# cross-lingual (de_en) questions that were bge-m3's real weak point --
# recovered a total-miss question (Q17, previously not found by dense OR
# sparse OR fused at all) and cut the worst cross-lingual fused rank from 20
# (bge-m3) to a max of 18 across all 18 silver questions with the real BM25
# + reranker pipeline, letting FUSED_TOP_K shrink from 40 to 25 without
# losing coverage. Requires a full corpus re-index (embedding dimension
# changed 1024 -> 768, and the vector space itself is different) -- the prior
# bge-m3 Chroma collection is preserved at
# data_corpus/vector_store/chroma_bge-m3_backup for rollback. Value itself
# lives in common/config.py now (not freely swappable via env var like the
# other config.py entries -- see that module's comment on EMBED_MODEL_NAME).
MODEL_NAME = EMBED_MODEL_NAME
EMBED_DIM = 768
# Corpus token-length check (dev_logs.md) found p99=593, max=1224 out of
# 5,833 chunks under bge-m3's tokenizer -- granite's is comparable. Capping
# avoids the same whole-batch-padding trap already fixed for the reranker
# (rerank.py's RERANK_MAX_LENGTH) and dedup (embed_texts_for_dedup below).
MAX_SEQ_LENGTH = EMBED_MAX_SEQ_LENGTH
# Dedup's near-duplicate check (hybrid_search.py's _dedup_by_source_priority)
# is a symmetric, ad-hoc similarity check over a small (~15) candidate window
# -- it never needs to share an embedding space with the persisted Chroma
# index, unlike embed_texts()/embed_query() below. That means it can use a
# DIFFERENT, lighter model with zero re-index cost, unlike swapping
# MODEL_NAME itself (which would require re-embedding the whole corpus).
# Overridable for A/B testing (DEDUP_MODEL_NAME env var, read per-call in
# embed_texts_for_dedup() below); default value lives in common/config.py.


@lru_cache(maxsize=1)
def _model():
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(MODEL_NAME, device=device)
    model.max_seq_length = MAX_SEQ_LENGTH
    if device == "cuda":
        model = model.half()
        # The fp32->fp16 cast transiently peaks at ~2.4x the final resident
        # size (both precisions briefly coexist), and PyTorch's caching
        # allocator doesn't return that freed peak back to the driver on its
        # own -- confirmed via torch.cuda.memory_summary(): ~2.68GB reserved
        # after this call vs. ~1.1GB actually allocated. On a 4GB card that
        # leftover ~1.6GB of dead reservation is enough to starve the
        # reranker's later load (see rerank.py) of a contiguous ~2.2GB
        # block, causing a real OOM even though nothing is using that memory.
        torch.cuda.empty_cache()
    return model


@lru_cache(maxsize=2)
def _dedup_model(model_name: str):
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    if device == "cuda":
        model = model.half()
        torch.cuda.empty_cache()
    return model


def embed_texts(texts: list[str], batch_size: int = 16) -> np.ndarray:
    if not texts:
        return np.empty((0, EMBED_DIM), dtype=np.float32)
    return _model().encode(texts, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)


def embed_query(query: str) -> np.ndarray:
    return embed_texts([query])[0]


def embed_texts_for_dedup(texts: list[str], batch_size: int = 16) -> np.ndarray:
    """Separate, swappable model for the dedup near-duplicate check only --
    see DEDUP_MODEL_NAME_DEFAULT above for why this is safe to change without
    a re-index. e5 models are trained with a "query: " prefix; for a
    symmetric document-vs-document comparison (not asymmetric query-vs-passage
    retrieval) the E5 authors' own guidance is to prefix both sides with
    "query: " uniformly, not "query: "/"passage: " -- confirmed against the
    intfloat/e5 model cards' usage examples for STS-style tasks."""
    if not texts:
        return np.empty((0,), dtype=np.float32)
    model_name = os.environ.get("DEDUP_MODEL_NAME", DEDUP_MODEL_NAME_DEFAULT)
    prefixed = [f"query: {t}" for t in texts]
    return _dedup_model(model_name).encode(prefixed, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)
