# Silver Dataset Validation Results

Diagnostic pass only -- see silver_dataset.md for the question set. Does NOT feed into or replace the 12-brief-question accuracy numbers.

| Silver# | Category | Refused | Recall@5 | NDCG@5 | Judge correct | Judge grounded |
|---------|----------|---------|----------|--------|----------------|-----------------|
| 1 | de_de | False | True | 1.0 | True | True |
| 2 | de_de | False | True | 1.0 | True | True |
| 3 | de_de | False | True | 1.0 | True | True |
| 4 | de_de | False | True | 1.0 | True | True |
| 5 | de_de | False | True | 0.5 | True | True |
| 6 | de_de | False | True | 1.0 | True | True |
| 7 | de_en | False | True | 1.0 | True | True |
| 8 | de_en | False | True | 1.0 | True | True |
| 9 | de_en | False | True | 1.0 | True | True |
| 10 | de_en | False | True | 1.0 | True | True |
| 11 | de_en | False | True | 1.0 | True | True |
| 12 | de_en | False | True | 1.0 | True | True |

## Overall

- **recall_at_3**: 1.0
- **recall_at_5**: 1.0
- **ndcg_at_5**: 0.958
- **ndcg_at_10**: 0.958
- **answer_accuracy_rate**: 1.0
- **unexpected_refusal_rate**: 0.0

## By category

- **A: German source / German query** (n=6): recall@5=1.0, ndcg@5=0.917, answer_accuracy=1.0, unexpected_refusal_rate=0.0
- **B: German source / English query (cross-lingual)** (n=6): recall@5=1.0, ndcg@5=1.0, answer_accuracy=1.0, unexpected_refusal_rate=0.0