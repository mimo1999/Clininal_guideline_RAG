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

    STORE_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    existing_names = [c.name for c in client.list_collections()]
    collection = None
    if not force_rebuild and COLLECTION_NAME in existing_names:
        collection = client.get_collection(COLLECTION_NAME)
        if collection.count() != len(chunk_ids):
            collection = None  # stale -- rebuild

    if collection is None:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        collection = client.get_or_create_collection(COLLECTION_NAME)

        tracker = ProgressTracker("index_build", total=len(chunk_ids), stage="embedding")

        # Embed and write one batch at a time (not embed-everything-then-add)
        # so a kill mid-run loses at most one batch's GPU compute instead of
        # the whole corpus's -- embedding 5000+ chunks on a 4GB GPU takes long
        # enough that an all-or-nothing call isn't safely interruptible.
        # Resuming after a partial run: re-invoking build_indexes() sees
        # collection.count() != len(chunk_ids) (stale, from the `collection
        # is None` check above) and starts over from batch 0 -- already-added
        # ids just get overwritten by Chroma's upsert-on-add semantics, so
        # this is correct, just not maximally efficient. A true resume would
        # need to diff existing ids first; not worth the complexity yet at
        # this corpus size.
        batch = 256
        try:
            for i in range(0, len(chunk_ids), batch):
                batch_ids = chunk_ids[i:i + batch]
                batch_texts = [chunks[cid]["text"] for cid in batch_ids]
                batch_embeddings = embed_texts(batch_texts)
                batch_metadatas = [_chroma_metadata(chunks[cid]) for cid in batch_ids]
                collection.add(
                    ids=batch_ids,
                    embeddings=batch_embeddings.tolist(),
                    documents=batch_texts,
                    metadatas=batch_metadatas,
                )
                tracker.set_stage(f"batch {i // batch + 1}/{(len(chunk_ids) - 1) // batch + 1}")
                tracker.advance(len(batch_ids))
        except Exception as e:
            tracker.finish(error=str(e))
            raise
        tracker.finish()

    bm25 = _load_or_build_bm25(chunks, chunk_ids, force_rebuild)

    return collection, bm25, chunk_ids, chunks
