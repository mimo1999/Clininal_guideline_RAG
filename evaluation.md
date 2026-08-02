# Evaluation — Golden 12 Questions (Cloud vs. Local)

Results from running the brief's 12-question set (9 answerable, 3 traps) through [evaluation/run_eval.py](evaluation/run_eval.py) (local) and [evaluation/run_eval_cloud_diagnostic.py](evaluation/run_eval_cloud_diagnostic.py) (cloud). Both runs share the same retrieval pipeline and vector index (7-guideline corpus) — only the generator/judge models differ. Raw outputs: [local_run_results/](local_run_results/), [cloud_run_results/](cloud_run_results/).

## Gold Question Set & Reference Answers

| Q# | Kind | Question | Reference Answer |
|---|---|---|---|
| 1 | gold | Welche Untersuchung und in welchem Intervall wird für Frauen im Alter von 20 bis 34 Jahren im organisierten Screening empfohlen? | Zytologie (Pap-Test) auf jährlicher Basis; HPV-Testung ist in dieser Altersgruppe nicht das primäre Screeninginstrument. |
| 2 | gold | Welches Testverfahren und welches Intervall gelten für Frauen ab 35 Jahren? | Ko-Testung (HPV-Test und Zytologie) im Intervall von 3 Jahren. |
| 3 | gold | Wie ist das empfohlene Vorgehen bei einer Frau ab 35, die HPV-positiv, aber zytologisch unauffällig ist? | Wiederholung der Ko-Testung nach ca. 12 Monaten statt sofortiger Kolposkopie; bei erneuter Positivität bei der Kontrolle erfolgt dann eine Kolposkopie. |
| 4 | self_labeled | Ab welchem Alter beginnt das organisierte Zervixkarzinom-Screening in Deutschland? | Das organisierte Screening kann ab 25 Jahren beginnen (Empfehlung 8.2, Konsensbasierte Empfehlung). In Deutschland haben Frauen ab 20 Jahren weiterhin einen gesetzlichen Anspruch auf eine Screeninguntersuchung gemäß Krebsfrüherkennungs-Richtlinie (KFE-RL). |
| 5 | self_labeled | Welche Rolle spielt die HPV-Selbstabnahme (Selbstentnahme) laut Leitlinie, und für welche Gruppe? | Der HPV-Selbstabstrich soll den Frauen vorbehalten bleiben, die sich nicht an der regulären Krebsvorsorgeuntersuchung beteiligen (Non-Responder) -- Empfehlungsgrad A. Er verdoppelt annähernd die Teilnahmerate bei diesen Frauen, hat aber eine etwas niedrigere Sensitivität als ein professionell entnommener Abstrich. |
| 6 | self_labeled | Welche HPV-Typen sind für die Mehrzahl der Zervixkarzinome verantwortlich? | HPV-Typen 16 und 18 sind für 60 bis 70% aller Zervixkarzinome verantwortlich. Die acht häufigsten HPV-Typen (16, 18, 33, 45, 31, 58, 52, 35) wurden in bis zu 90% der untersuchten Tumormaterialien nachgewiesen. |
| 7 | self_labeled | Was ist der Unterschied zwischen einer zytologiebasierten und einer HPV-basierten Screening-Strategie? | Ein organisiertes HPV-basiertes Screening (ab 30 Jahren, Intervall 3-5 Jahre) führt zu einer niedrigeren Rate an Neuerkrankungen am Zervixkarzinom im Vergleich zu einem rein zytologiebasierten Screening mit 3-jährlichen Intervallen, da HPV-Tests eine höhere Sensitivität für die Detektion von CIN 3+ haben. Cytologie-basiertes Screening hat dagegen eine höhere Spezifität und weniger falsch-positive Ergebnisse, besonders bei jüngeren Frauen -- deshalb wird HPV-basiertes Screening unter 30 Jahren nicht empfohlen. |
| 8 | self_labeled | Welche Empfehlung gibt die Leitlinie zur Kolposkopie bei auffälligem Befund? | Die Indikation zur kolposkopischen Abklärung soll ab einer Post-Test-Wahrscheinlichkeit für ein durchschnittliches kumulatives CIN 3+ Risiko von 10% gestellt werden (Empfehlung 10.2, Konsensbasierte Empfehlung, EK). |
| 9 | self_labeled | Welche Bedeutung hat die HPV-Impfung im Kontext der Prävention laut Leitlinie? | Die HPV-Impfung ist primäre Prävention (im Gegensatz zum Screening als sekundäre Prävention). Die STIKO empfiehlt die Impfung aller Mädchen im Alter von 9 bis 14 Jahren; sie verhindert die Infektion mit den Impf-HPV-Typen (u.a. 16, 18) sowie die Entstehung der daraus resultierenden Krebsvorstufen. |
| 10 | trap | Welche Thrombolyse-Therapie und welches Zeitfenster gelten beim akuten ischämischen Schlaganfall? | *(Trap question — correct behavior is refusal)* |
| 11 | trap | Welches Antibiotikum ist first-line bei einer ambulant erworbenen Pneumonie? | *(Trap question — correct behavior is refusal)* |
| 12 | trap | Welche medikamentöse Erstlinientherapie wird bei arterieller Hypertonie empfohlen? | *(Trap question — correct behavior is refusal)* |

## Models

| Role | Cloud run | Local run |
|---|---|---|
| Generator | `gemma4:31b-cloud` (Ollama cloud) | `google/gemma-4-E2B-it` (transformers) |
| Judge | `nemotron-3-super:cloud` (Ollama cloud) | `Qwen/Qwen3.5-2B` (transformers) |
| Dense embedder | `ibm-granite/granite-embedding-278m-multilingual` (shared, both runs) | |
| Cross-encoder reranker | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (shared, both runs) | |

**Local hardware**: T4 GPU, 15GB VRAM.

## Results

Retrieval uses the same index and queries across both runs, with minor cross-encoder rank shifts on boundary chunks (e.g. Q9 target chunk at rank 2 in Cloud vs rank 5 in Local; Q8 target chunk unranked in top-10 in Cloud vs rank 8 in Local).

| Q# | Kind | Refused | Recall@3 (C / L) | Recall@5 | NDCG@5 (C / L) | NDCG@10 (C / L) | Judge correct (Cloud) | Judge correct (Local) |
|----|------|---------|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | gold | False | True / True | True | 1.000 / 1.000 | 1.000 / 1.000 | True | True |
| 2 | gold | False | True / True | True | 1.000 / 1.000 | 1.000 / 1.000 | True | True |
| 3 | gold | False | True / True | True | 0.631 / 0.631 | 0.631 / 0.631 | False | True |
| 4 | self_labeled | False | True / True | True | 1.000 / 1.000 | 1.000 / 1.000 | True | True |
| 5 | self_labeled | False | True / True | True | 1.000 / 1.000 | 1.000 / 1.000 | False | True |
| 6 | self_labeled | False | True / True | True | 1.000 / 1.000 | 1.000 / 1.000 | True | True |
| 7 | self_labeled | False | True / True | True | 0.500 / 0.500 | 0.500 / 0.500 | True | True |
| 8 | self_labeled | False | False / False | False | 0.000 / 0.000 | 0.000 / 0.356 | True | True |
| 9 | self_labeled | False | True / False | True | 0.500 / 0.387 | 0.500 / 0.387 | True | False |
| 10 | trap | True | — | — | — | — | — | — |
| 11 | trap | True | — | — | — | — | — | — |
| 12 | trap | True | — | — | — | — | — | — |

## Rates

| Metric | Cloud | Local | Definition |
|---|---|---|---|
| Recall@3 | 0.889 | 0.778 | Fraction of answerable questions (1–9) whose correct evidence appears in the top-3 post-rerank chunks. |
| Recall@5 | 0.889 | 0.889 | Same, top-5. |
| NDCG@5 | 0.737 | 0.724 | Rewards ranking the correct passage higher, not just hit/miss (see README). |
| NDCG@10 | 0.737 | 0.764 | Same, top-10. |
| Answer accuracy rate | **0.778** | **0.889** | Fraction of answerable questions judged both correct and grounded (no hallucination). |
| Refusal correctness rate | **1.0** | **1.0** | Fraction of trap questions (10–12) correctly refused. A system that confidently answers a trap fails this regardless of other scores. |

## Takeaway

With strictly untuned section annotations, the pipeline achieves **88.9% Recall@3/5** and an **NDCG@5 of 0.737** on the core 9 questions. Generative answer accuracy on the cloud run with the strict `nemotron-3-super:cloud` judge is **0.778** (7 out of 9 correct), and refusal correctness remains at **100%**.
