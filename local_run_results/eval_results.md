# Evaluation Results

| Q# | Kind | Refused | Recall@3 | Recall@5 | NDCG@5 | NDCG@10 | Judge correct | Judge grounded |
|----|------|---------|----------|----------|--------|---------|----------------|-----------------|
| 1 | gold | False | True | True | 1.0 | 1.0 | True | True |
| 2 | gold | False | True | True | 1.0 | 1.0 | True | True |
| 3 | gold | False | False | False | 0.0 | 0.0 | True | True |
| 4 | self_labeled | False | True | True | 1.0 | 1.0 | True | True |
| 5 | self_labeled | False | True | True | 1.0 | 1.0 | True | True |
| 6 | self_labeled | False | False | False | 0.0 | 0.0 | True | True |
| 7 | self_labeled | False | True | True | 0.631 | 0.631 | True | True |
| 8 | self_labeled | False | False | True | 0.431 | 0.431 | True | True |
| 9 | self_labeled | False | False | False | 0.0 | 0.0 | True | True |
| 10 | trap | True | None | None | None | None | None | None |
| 11 | trap | True | None | None | None | None | None | None |
| 12 | trap | True | None | None | None | None | None | None |

## Rates

- **Retrieval Recall@3** (post-rerank): 0.556 -- fraction of answerable questions (1-9) whose expected section_number appears in the top-3 reranked chunks.
- **Retrieval Recall@5** (post-rerank): 0.667 -- same, top-5.
- **Retrieval Recall@3 pre-rerank (ablation)**: 0.556 -- same definition, computed on the RRF-fused list before cross-encoder reranking.
- **Retrieval Recall@5 pre-rerank (ablation)**: 0.556
- **NDCG@5** (post-rerank): 0.562 -- mean, over answerable questions, of 1/log2(rank+1) if the expected section appears at position `rank` in the top-5 reranked chunks, else 0 (rewards ranking the right passage higher, unlike plain Recall@k).
- **NDCG@10** (post-rerank): 0.562 -- same, top-10.
- **NDCG@5 pre-rerank (ablation)**: 0.5 -- same definition, on the RRF-fused list before reranking -- the gap vs. NDCG@5 above is how much the cross-encoder rerank improves ranking quality, not just hit/miss.
- **NDCG@10 pre-rerank (ablation)**: 0.566
- **Answer accuracy rate**: 1.0 -- fraction of answerable questions (1-9) judged both correct and grounded by the local LLM-as-judge.
- **Refusal correctness rate**: 1.0 -- fraction of trap questions (10-12) correctly refused.