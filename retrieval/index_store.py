"""Builds/loads the ChromaDB dense-vector collection and the BM25 sparse
index from every ingested document's chunks.jsonl. Chroma only carries the
minimal metadata needed for filtering (list-typed fields like heading_path
and topic aren't valid Chroma metadata values) -- full chunk records are
kept in an in-memory dict, hydrated straight from the JSONL files, and used
to enrich whatever chunk_ids either index returns.
"""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from common.progress import ProgressTracker

PROCESSED_DIR = Path(__file__).parent.parent / "data_corpus" / "processed"
STORE_DIR = Path(__file__).parent.parent / "data_corpus" / "vector_store"
CHROMA_PATH = STORE_DIR / "chroma"
BM25_PATH = STORE_DIR / "bm25.pkl"
COLLECTION_NAME = "guideline_chunks"

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def load_all_chunks() -> dict[str, dict]:
    chunks: dict[str, dict] = {}
    if not PROCESSED_DIR.exists():
        return chunks
    for doc_dir in sorted(PROCESSED_DIR.iterdir()):
        jsonl_path = doc_dir / "chunks.jsonl"
        if not jsonl_path.exists():
            continue
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            chunks[record["chunk_id"]] = record
    return chunks


def _chroma_metadata(chunk: dict) -> dict:
    return {
        "doc_id": chunk.get("doc_id") or "",
        "chunk_type": chunk.get("chunk_type") or "",
        "section_id": chunk.get("section_id") or "",
        "section_number": chunk.get("section_number") or "",
        # Added for the guideline/document router scoping (hybrid_search.py
        # filters dense+BM25 candidates to the routed guideline/document set
        # via Chroma `where`) and the language bonus at rerank time.
        "guideline_id": chunk.get("guideline_id") or "",
        "doc_type": chunk.get("doc_type") or "",
        "doc_language": chunk.get("doc_language") or "",
        "source_priority": chunk.get("source_priority", 3),
    }


def _load_or_build_bm25(chunks: dict, chunk_ids: list[str], force_rebuild: bool) -> BM25Okapi:
    if not force_rebuild and BM25_PATH.exists():
        with BM25_PATH.open("rb") as f:
            saved_ids, bm25 = pickle.load(f)
        if saved_ids == chunk_ids:
            return bm25

    tokenized = [_tokenize(chunks[cid]["text"]) for cid in chunk_ids]
    bm25 = BM25Okapi(tokenized)
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    with BM25_PATH.open("wb") as f:
        pickle.dump((chunk_ids, bm25), f)
    return bm25


def build_indexes(force_rebuild: bool = False):
    import chromadb

    from .embed import embed_texts

    chunks = load_all_chunks()
    chunk_ids = list(chunks.keys())
    desired_ids = set(chunk_ids)

    STORE_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    if force_rebuild:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
    collection = client.get_or_create_collection(COLLECTION_NAME)

    # Diff against what's already embedded rather than the old all-or-nothing
    # "count mismatch -> delete + re-embed everything" check -- that made
    # adding a single new guideline re-embed the ENTIRE corpus (thousands of
    # chunks) just because the total count changed. include=[] skips pulling
    # back embeddings/documents/metadatas for this check, just ids -- cheap
    # even at several thousand chunks.
    try:
        existing_ids = set(collection.get(include=[])["ids"])
    except Exception:
        existing_ids = set(collection.get()["ids"])

    stale_ids = existing_ids - desired_ids
    if stale_ids:
        # A chunk_id no longer produced by the current corpus (doc removed,
        # or re-chunked under a different id scheme) -- drop it so it can't
        # still surface in search results for content that no longer exists.
        collection.delete(ids=list(stale_ids))

    missing_ids = [cid for cid in chunk_ids if cid not in existing_ids]
    if missing_ids:
        tracker = ProgressTracker("index_build", total=len(missing_ids), stage="embedding")

        # Embed and write one batch at a time (not embed-everything-then-add)
        # so a kill mid-run loses at most one batch's GPU compute instead of
        # the whole run's -- embedding thousands of chunks on a constrained
        # GPU takes long enough that an all-or-nothing call isn't safely
        # interruptible. Resuming after a partial run: re-invoking
        # build_indexes() re-diffs existing vs. desired ids from scratch, so
        # already-added ids from the interrupted run are correctly excluded
        # from missing_ids and never re-embedded.
        batch = 256
        try:
            for i in range(0, len(missing_ids), batch):
                batch_ids = missing_ids[i:i + batch]
                batch_texts = [chunks[cid]["text"] for cid in batch_ids]
                batch_embeddings = embed_texts(batch_texts)
                batch_metadatas = [_chroma_metadata(chunks[cid]) for cid in batch_ids]
                collection.add(
                    ids=batch_ids,
                    embeddings=batch_embeddings.tolist(),
                    documents=batch_texts,
                    metadatas=batch_metadatas,
                )
                tracker.set_stage(f"batch {i // batch + 1}/{(len(missing_ids) - 1) // batch + 1}")
                tracker.advance(len(batch_ids))
        except Exception as e:
            tracker.finish(error=str(e))
            raise
        tracker.finish()

    bm25 = _load_or_build_bm25(chunks, chunk_ids, force_rebuild)

    return collection, bm25, chunk_ids, chunks
