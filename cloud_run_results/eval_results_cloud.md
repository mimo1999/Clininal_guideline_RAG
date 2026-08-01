# Evaluation Results -- Ollama Cloud Diagnostic (gemma4:31b-cloud / nemotron-3-nano:30b-cloud)

| Q# | Kind | Refused | Recall@3 | Recall@5 | NDCG@5 | NDCG@10 | Judge correct | Judge grounded |
|----|------|---------|----------|----------|--------|---------|----------------|-----------------|
| 1 | gold | False | True | True | 1.0 | 1.0 | True | True |
| 2 | gold | False | True | True | 1.0 | 1.0 | True | True |
| 3 | gold | False | False | False | 0.0 | 0.301 | False | True |
| 4 | self_labeled | False | True | True | 1.0 | 1.0 | True | True |
| 5 | self_labeled | False | True | True | 1.0 | 1.0 | True | True |
| 6 | self_labeled | False | False | False | 0.0 | 0.0 | True | True |
| 7 | self_labeled | False | True | True | 0.631 | 0.631 | True | True |
| 8 | self_labeled | False | False | False | 0.0 | 0.356 | True | True |
| 9 | self_labeled | False | False | True | 0.387 | 0.387 | True | True |
| 10 | trap | True | None | None | None | None | None | None |
| 11 | trap | True | None | None | None | None | None | None |
| 12 | trap | True | None | None | None | None | None | None |

## Rates

- **recall_at_3**: 0.556
- **recall_at_5**: 0.667
- **fused_recall_at_3**: 0.556
- **fused_recall_at_5**: 0.556
- **ndcg_at_5**: 0.558
- **ndcg_at_10**: 0.631
- **fused_ndcg_at_5**: 0.5
- **fused_ndcg_at_10**: 0.566
- **answer_accuracy_rate**: 0.889
- **refusal_correctness_rate**: 1.0