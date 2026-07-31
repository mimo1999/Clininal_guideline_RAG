# Clinical RAG — DGGG/AWMF Guideline Assistant

A retrieval-augmented chatbot over German gynecology clinical guidelines (DGGG/AWMF), built for the Nixi AI take-home task (Option B).

## Quick start

```bash
python setup.py
```

That single command: installs dependencies, unpacks the pre-built vector index, ingests+chunks the guideline PDFs if needed, best-effort starts Langfuse tracing (skipped cleanly if not available), and starts the webapp — printing the `http://127.0.0.1:8080` link once it's ready.

**Nothing needs to be pre-installed except Python.** No Ollama, no Docker/Podman, no account of any kind required for the default setup — generation and judging run entirely in-process via `transformers` (`google/gemma-4-E2B-it` / `Qwen/Qwen3.5-2B`), downloading their own weights automatically on first use, the same way the retrieval models already do.

Re-running `python setup.py` is safe — every step checks whether its output already exists before doing real work again.

**Flags**: `--skip-install` (skip pip install), `--skip-ingest` (skip PDF ingestion/chunking even if empty), `--langfuse-dir PATH` (point at a Langfuse checkout other than `../langfuse_v2`).

## Running the project — 3 entrypoints, in order

There are exactly three executable entrypoints. Run them in this order:

| # | Command | What it does |
|---|---|---|
| 1 | `python setup.py` | One-time (or idempotent re-run) project setup: installs dependencies, unpacks the pre-built vector index, ingests/chunks the guideline PDFs if needed, best-effort starts Langfuse. **Also launches the webapp itself at the end** — see step 2 for launching it standalone instead. |
| 2 | `python -m webapp.run_webapp` | Launches the chat UI **and** the RAG backend that answers it, together in a single process, at `http://127.0.0.1:8080`. Only needed on its own if you already ran `setup.py` once before and just want to relaunch the app without redoing setup. |
| 3 | `python -m evaluation.run_eval` | Runs the pipeline against the golden/trap question set (12 questions) and writes the benchmark report. |

**Step 1 must be run at least once before step 2 or step 3** — both depend on the vector index and chunked corpus that `setup.py` produces. After that, steps 2 and 3 can be run independently, any number of times, in either order (step 2 to use the chatbot interactively, step 3 to (re-)benchmark the pipeline — they don't depend on each other).

### Benchmarking the pipeline (step 3, in detail)

```bash
python setup.py               # once, if not already done — builds the index + chunked corpus run_eval needs
python -m evaluation.run_eval
```

`run_eval` runs all 12 questions (9 answerable, 3 traps) through retrieval → generation → judging, and writes three files to `data_corpus/`:

- **`eval_report.csv`** — the benchmark report: one row per rate (`metric, k, value, definition`), each with a one-line description of how it was computed. This is the file to check for pipeline performance:
  - **Retrieval Recall@3 / Recall@5** — fraction of answerable questions (1–9) whose correct evidence (expected guideline section) appears in the top-3 / top-5 post-rerank retrieved chunks.
  - **Answer accuracy rate** — fraction of answerable questions answered correctly *and* grounded in retrieved text (no hallucination). Judged automatically by a local LLM-as-judge (`Qwen/Qwen3.5-2B`, deliberately a different model from the generator to avoid self-judging bias). The judge's reliability was validated by cross-checking its verdicts against an independent cloud judge on an earlier run — see dev_logs.md Entry 14, where a same-model self-judge was caught rating clearly wrong/fabricated answers as correct on 6 of 9 questions, which is why generator and judge are required to be separate models here.
  - **Refusal correctness rate** — fraction of trap questions (10–12, no answer exists in the corpus) correctly refused. A system that confidently answers a trap question fails this check regardless of every other score.
  - **NDCG@5** — an additional honest/informative rate: rewards ranking the correct passage *higher* within the top-5, rather than treating every top-k position as equivalent the way plain Recall@k does.
- **`eval_results.md`** — the same rates plus a full per-question table (human-readable).
- **`eval_results_full.json`** — every question's raw retrieval/generation/judge output (for debugging a specific question's result).

Re-running `evaluation.run_eval` is safe and idempotent with respect to the corpus/index — it only re-does the generation + judging pass each time (a few minutes on GPU, longer on CPU).

## Prerequisites

- **Python 3.10+** — that's it for a from-scratch machine. `setup.py` (and `transformers` itself, on first use) installs/downloads everything else.
- **An NVIDIA GPU is recommended but not required** — every model in this project (retrieval, generation, judge) auto-detects CUDA and falls back to CPU otherwise (slower, not broken).
- **Langfuse tracing is entirely optional.** If you have a `langfuse_v2` checkout (https://github.com/langfuse/langfuse) as a sibling directory to this repo, and Podman or Docker installed, `setup.py` starts it automatically. If not, it's skipped with a one-line note — the webapp works identically either way, just without trace visibility.
- **Want to use Ollama instead?** Set `CLINICAL_RAG_GENERATOR_MODEL` to any Ollama tag (local or `-cloud`) before running `setup.py` — this routes generation through `generation/ollama_llm.py` instead, and `setup.py` will install Ollama and pull that tag for you. Cloud tags need an Ollama account; local tags don't. See dev_logs.md Entries 16-18 for the full history of why the default ended up on transformers rather than Ollama.

## What's shipped vs. regenerated

Per `.gitignore`, only `data_corpus/vector_store/chroma_db.zip` (the pre-built vector index, zipped) and `data_corpus/pdf/` (source guideline PDFs) are committed from `data_corpus/`. Everything else is regenerated locally by `setup.py`:

| Path | Source |
|---|---|
| `data_corpus/vector_store/chroma/` | Unzipped from `chroma_db.zip` (or rebuilt from scratch if the zip is missing) |
| `data_corpus/processed/` | Regenerated from `data_corpus/pdf/` via `ingestion.run_ingest` + `chunking.build_chunks` |
| `data_corpus/vector_store/bm25.pkl` | Built automatically on first retrieval call |

This keeps the repo small (PDFs + a compressed vector index, not the much larger set of intermediate parsed/chunked/cached artifacts) while still being fully reproducible from source.

## Manual / partial runs

Each pipeline stage is also runnable standalone, if you don't want the full `setup.py` flow:

```bash
python -m ingestion.run_ingest --input data_corpus/pdf/    # PDF -> parsed + metadata
python -m chunking.build_chunks                             # parsed -> chunks.jsonl
python -m webapp.run_webapp                                  # start the UI directly
python -m evaluation.run_eval                                 # local (transformers) eval against the golden 12
python -m evaluation.run_eval_cloud_diagnostic                # cloud-diagnostic eval (see dev_logs.md Entry 4)
python -m evaluation.run_eval_silver                           # self-generated silver-set validation pass
```

## Repository layout

- `ingestion/` — PDF parsing (Docling), AWMF metadata reconciliation, language ID
- `chunking/` — structure extraction, section/table/recommendation chunking
- `retrieval/` — dense (bge-m3-family) + sparse (BM25) hybrid search, guideline/document routing, cross-encoder reranking
- `generation/` — prompt construction; `llm.py` (default, transformers-based local generator + judge) and `ollama_llm.py` (Ollama-based alternative, used when `CLINICAL_RAG_GENERATOR_MODEL` opts in, and by the cloud-diagnostic eval path)
- `evaluation/` — golden-12, silver-18, and cross-lingual eval harnesses; LLM-as-judge (local + cloud + ragas)
- `webapp/` — FastAPI chat UI, Langfuse tracing integration
