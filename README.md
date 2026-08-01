# Clinical RAG — DGGG/AWMF Guideline Assistant

A retrieval-augmented chatbot over German gynecology clinical guidelines (DGGG/AWMF), built for the Nixi AI take-home task (Option B).

## Quick start

```bash
python setup.py
```

Installs dependencies, restores the pre-built vector index + chunk data, best-effort starts Langfuse (skipped cleanly if unavailable), and starts the webapp at `http://127.0.0.1:8080`. Nothing needs to be pre-installed except **Python 3.10+** — no Ollama, no Docker, no accounts. Generation/judging run locally via `transformers` (`google/gemma-4-E2B-it` / `Qwen/Qwen3.5-2B`), downloading weights on first use. Safe to re-run.

Flags: `--skip-install`, `--skip-ingest`, `--langfuse-dir PATH`.

## The 3 entrypoints, in order

| # | Command | Does |
|---|---|---|
| 1 | `python setup.py` | Setup + launches the webapp. Run this first, always. |
| 2 | `python -m webapp.run_webapp` | Relaunch just the chat UI + backend (skip if you just ran `setup.py`). |
| 3 | `python -m evaluation.run_eval` | Benchmark the pipeline against the 12 golden/trap questions. |

Step 1 must run once before 2 or 3 (both need the index/chunks it builds). After that, 2 and 3 are independent and repeatable in any order.

## Benchmarking

```bash
python -m evaluation.run_eval
```

Runs all 12 questions (9 answerable, 3 traps) through retrieval → generation → judging, writing to `data_corpus/`:

- **`eval_report.csv`** — the report to check: `metric, k, value, definition` per rate:
  - **Recall@3 / Recall@5** — fraction of answerable questions (1–9) whose correct evidence appears in the top-k reranked chunks.
  - **Answer accuracy rate** — fraction judged correct *and* grounded (no hallucination) by a local LLM-as-judge (`Qwen/Qwen3.5-2B`, a different model from the generator to avoid self-judging bias — validated against an independent cloud judge, see dev_logs.md Entry 14).
  - **Refusal correctness rate** — fraction of trap questions (10–12) correctly refused. Answering a trap confidently fails this regardless of other scores.
  - **NDCG@5** — rewards ranking the right passage *higher*, not just hit/miss.
- **`eval_results.md`** — same rates + full per-question table.
- **`eval_results_full.json`** — raw per-question output, for debugging.

## What's shipped vs. regenerated

Only `data_corpus/vector_store/chroma_db.zip` (vector index), `data_corpus/processed.zip` (chunk/router records), and `data_corpus/pdf/` (source PDFs) are committed. `setup.py` restores everything else:

| Path | Source |
|---|---|
| `data_corpus/vector_store/chroma/` | Unzipped from `chroma_db.zip` |
| `data_corpus/processed/` | Unzipped from `processed.zip` (or rebuilt from `pdf/` if the zip's missing) |
| `data_corpus/vector_store/bm25.pkl` | Built on first retrieval call |

Both zips are required, not just the index: Chroma only stores minimal filter metadata — full chunk text and router text are read straight from `processed/*/chunks.jsonl`/`metadata.json`/`router_text.txt` at query time (`retrieval/index_store.py`, `guideline_router.py`, `document_router.py`).

## Other options

- **Ollama instead of transformers**: set `CLINICAL_RAG_GENERATOR_MODEL` to an Ollama tag before running `setup.py` (it'll install Ollama and pull the tag). See dev_logs.md Entries 16-18.
- **GPU optional**: every model auto-detects CUDA, falls back to CPU.
- **Langfuse optional**: auto-starts if a `langfuse_v2` checkout + Docker/Podman are present; skipped otherwise.
- **Manual/standalone stages**:
  ```bash
  python -m ingestion.run_ingest --input data_corpus/pdf/
  python -m chunking.build_chunks
  python -m evaluation.run_eval_cloud_diagnostic   # dev_logs.md Entry 4
  python -m evaluation.run_eval_silver
  ```

## Repository layout

- `ingestion/` — PDF parsing (Docling), AWMF metadata, language ID
- `chunking/` — structure extraction, section/table/recommendation chunking
- `retrieval/` — hybrid dense+BM25 search, guideline/document routing, reranking
- `generation/` — prompts + `llm.py` (transformers, default) / `ollama_llm.py` (Ollama alt.)
- `evaluation/` — golden-12/silver-18 harnesses, LLM-as-judge
- `webapp/` — FastAPI chat UI, Langfuse tracing
