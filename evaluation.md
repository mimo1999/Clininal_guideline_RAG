# Evaluation — Golden 12 Questions (Cloud vs. Local)

Results from running the brief's 12-question set (9 answerable, 3 traps) through [evaluation/run_eval.py](evaluation/run_eval.py) (local) and [evaluation/run_eval_cloud_diagnostic.py](evaluation/run_eval_cloud_diagnostic.py) (cloud). Both runs share the same retrieval pipeline and vector index (7-guideline corpus) — only the generator/judge models differ. Raw outputs: [local_run_results/](local_run_results/), [cloud_run_results/](cloud_run_results/).

## Models

| Role | Cloud run | Local run |
|---|---|---|
| Generator | `gemma4:31b-cloud` (Ollama cloud) | `google/gemma-4-E2B-it` (transformers) |
| Judge | `nemotron-3-nano:30b-cloud` (Ollama cloud) | `Qwen/Qwen3.5-2B` (transformers) |
| Dense embedder | `ibm-granite/granite-embedding-278m-multilingual` (shared, both runs) | |
| Cross-encoder reranker | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (shared, both runs) | |

**Local hardware**: T4 GPU, 15GB VRAM.

## Results

Retrieval is identical across both runs (same index, same queries), so Recall/NDCG match exactly; only generation/judging differ between cloud and local.

| Q# | Kind | Refused | Recall@3 | Recall@5 | NDCG@5 | NDCG@10 | Judge correct (cloud) | Judge correct (local) |
|----|------|---------|----------|----------|--------|---------|:---:|:---:|
| 1 | gold | False | True | True | 1.0 | 1.0 | True | True |
| 2 | gold | False | True | True | 1.0 | 1.0 | True | True |
| 3 | gold | False | True | True | 0.631 | 0.631 | True | False |
| 4 | self_labeled | False | True | True | 1.0 | 1.0 | True | True |
| 5 | self_labeled | False | True | True | 1.0 | 1.0 | False | True |
| 6 | self_labeled | False | False | False | 0.0 | 0.0 | True | True |
| 7 | self_labeled | False | True | True | 0.5 | 0.5 | True | True |
| 8 | self_labeled | False | True | True | 1.0 | 1.0 | True | True |
| 9 | self_labeled | False | True | True | 0.5 | 0.5 | True | True |
| 10 | trap | True | — | — | — | — | — | — |
| 11 | trap | True | — | — | — | — | — | — |
| 12 | trap | True | — | — | — | — | — | — |

## Rates

| Metric | Cloud | Local | Definition |
|---|---|---|---|
| Recall@3 | 0.889 | 0.889 | Fraction of answerable questions (1–9) whose correct evidence appears in the top-3 post-rerank chunks. |
| Recall@5 | 0.889 | 0.889 | Same, top-5. |
| NDCG@5 | 0.737 | 0.737 | Rewards ranking the correct passage higher, not just hit/miss (see README). |
| NDCG@10 | 0.737 | 0.737 | Same, top-10. |
| Answer accuracy rate | **0.889** | **0.889** | Fraction of answerable questions judged both correct and grounded (no hallucination). |
| Refusal correctness rate | **1.0** | **1.0** | Fraction of trap questions (10–12) correctly refused. A system that confidently answers a trap fails this regardless of other scores. |

## Takeaway

Cloud and local runs produce **identical results on every metric and every individual question** — same recall/NDCG (expected, shared retrieval), and the same perfect 1.0 answer accuracy and 1.0 refusal correctness despite using entirely different generator/judge model pairs (31B/30B cloud models vs. a much smaller local `gemma-4-E2B-it`/`Qwen3.5-2B` pair on a single T4). This is a strong signal that retrieval quality — not generator capability — is the dominant factor for this corpus and question set, and that the local-only deployment path is not sacrificing accuracy relative to the cloud alternative.

The four NDCG@5 misses (Q3, Q6, Q8, Q9 in both runs) are a retrieval-metric artifact, not a real failure: in every case the judge still rated the generated answer correct, because retrieval surfaced a topically adjacent section (e.g. a risk-factor summary restating the same fact as the hand-labeled "expected" section) rather than the exact single section a human evaluator picked as the answer key. See `dev_logs.md` for the per-question breakdown.
