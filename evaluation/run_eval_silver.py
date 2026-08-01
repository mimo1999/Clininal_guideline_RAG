"""Silver-dataset validation pass -- runs evaluation/silver_questions.py's 12
self-generated questions through the SAME retrieval + cloud-model generation
+ judging pipeline as run_eval_cloud_diagnostic.py, but is entirely separate
from it: own result files (eval_results_silver.md / _full.json), own Langfuse
run_label ("silver", so traces never mix with "cloud"/"local" runs), and
never touches the 12 brief questions or their numbers. This is a debugging/
regression-signal pass, not the accuracy deliverable -- that remains
run_eval.py (local, submitted system) / run_eval_cloud_diagnostic.py
(cloud, diagnostic) against the 12 brief questions.

Reuses _judge()/_looks_like_refusal() from run_eval_cloud_diagnostic.py
directly rather than re-implementing them (same nemotron-3-nano token-budget
fix, same JSON-parsing logic) -- one code path for "ask the cloud judge",
not two to keep in sync.

No refusal-correctness metric here: every silver question is answerable and
grounded in a real chunk (none are traps, unlike the brief's Q10-12), so an
unexpected refusal is itself a finding worth flagging, not an expected
outcome to score against.

Known limitation, not fixed here: the judge prompt (evaluation/judge.py's
JUDGE_SYSTEM_PROMPT, and _judge()'s user-message template in
run_eval_cloud_diagnostic.py) is still German-only -- it was never in scope
for the generator-prompt bilingual fix (dev_logs.md Entry 5 Section 8). So
category B/de_en questions get judged by a German-instructed judge
evaluating English-language answers/references. Flagged here rather than
silently assumed fine, since it's a plausible confound if category B
accuracy looks worse than category A for reasons unrelated to retrieval or
generation quality.

A third category, "en_en" (English source content, English query), existed
here previously but was removed along with its underlying source document
(032-033OL's English translation, deleted -- see silver_questions.py's own
docstring and dev_logs.md). No English-language source content remains in
the corpus for that category to be grounded in.
"""

from __future__ import annotations

import json
from pathlib import Path

from evaluation import tracing
from evaluation.run_eval import RESULTS_DIR, _ndcg_at_k
from evaluation.run_eval_cloud_diagnostic import GENERATOR_MODEL, JUDGE_MODEL, _judge, _looks_like_refusal
from evaluation.silver_questions import SILVER_QUESTIONS
from generation.ollama_llm import generate_with_usage as ollama_generate
from generation.prompt import REFUSAL_STRINGS, build_messages, detect_query_language
from retrieval.hybrid_search import HybridSearcher
from retrieval.index_store import load_all_chunks

RESULTS_JSON_PATH = RESULTS_DIR / "eval_results_silver_full.json"
RESULTS_MD_PATH = RESULTS_DIR / "eval_results_silver.md"

CATEGORY_LABELS = {
    "de_de": "A: German source / German query",
    "de_en": "B: German source / English query (cross-lingual)",
}


def _rate(values) -> float | None:
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 3) if values else None


def compute_silver_rates(rows: list[dict]) -> dict:
    overall = {
        "recall_at_3": _rate(r["recall_hit_at_3"] for r in rows),
        "recall_at_5": _rate(r["recall_hit_at_5"] for r in rows),
        "ndcg_at_5": _rate(r["ndcg_at_5"] for r in rows),
        "ndcg_at_10": _rate(r["ndcg_at_10"] for r in rows),
        "answer_accuracy_rate": _rate((r["judge_correct"] and r["judge_grounded"]) for r in rows),
        # every silver question is answerable -- any refusal here is unexpected,
        # not a "should be refused" case like the brief's traps.
        "unexpected_refusal_rate": _rate(r["refused"] for r in rows),
    }
    by_category = {}
    for cat in CATEGORY_LABELS:
        cat_rows = [r for r in rows if r["category"] == cat]
        by_category[cat] = {
            "n": len(cat_rows),
            "recall_at_5": _rate(r["recall_hit_at_5"] for r in cat_rows),
            "ndcg_at_5": _rate(r["ndcg_at_5"] for r in cat_rows),
            "answer_accuracy_rate": _rate((r["judge_correct"] and r["judge_grounded"]) for r in cat_rows),
            "unexpected_refusal_rate": _rate(r["refused"] for r in cat_rows),
        }
    return {"overall": overall, "by_category": by_category}


def main() -> None:
    chunk_lookup = load_all_chunks()
    searcher = HybridSearcher()
    rows = []

    for q in SILVER_QUESTIONS:
        language = detect_query_language(q.question)
        refusal_string = REFUSAL_STRINGS[language]

        trace = tracing.get_or_create_trace(q.id, q.question, q.category, run_label="silver")
        with tracing.trace_retrieval(trace, q.question) as span:
            result = searcher.search(q.question)
            if span is not None:
                span.update(output={
                    "refused": result.refused, "refusal_reason": result.refusal_reason,
                    "routed_guideline_ids": result.routed_guideline_ids, "routed_doc_ids": result.routed_doc_ids,
                    "chunk_ids": [c.get("chunk_id") for c in result.chunks],
                })

        row = {
            "id": q.id, "category": q.category, "question": q.question,
            "source_guideline_id": q.source_guideline_id, "source_section_number": q.source_section_number,
            "source_chunk_id": q.source_chunk_id, "reference_answer": q.reference_answer,
            "refused": result.refused, "refusal_reason": result.refusal_reason,
            "answer_text": None,
            "recall_hit_at_3": None, "recall_hit_at_5": None, "ndcg_at_5": None, "ndcg_at_10": None,
            "judge_correct": None, "judge_grounded": None, "judge_reasoning": None,
        }

        if result.refused:
            tracing.set_trace_output(trace, refusal_string)
            print(f"  Silver {q.id} [{q.category}]: RETRIEVAL REFUSED ({result.refusal_reason}) "
                  f"-- unexpected, this question should be answerable")
            rows.append(row)
            continue

        reranked_sections = [chunk_lookup[cid]["section_number"] for cid in result.reranked_chunk_ids if cid in chunk_lookup]
        row["recall_hit_at_3"] = q.source_section_number in reranked_sections[:3]
        row["recall_hit_at_5"] = q.source_section_number in reranked_sections[:5]
        row["ndcg_at_5"] = _ndcg_at_k(reranked_sections, q.source_section_number, 5)
        row["ndcg_at_10"] = _ndcg_at_k(reranked_sections, q.source_section_number, 10)

        messages = build_messages(q.question, result.chunks, language)
        with tracing.trace_generation(trace, GENERATOR_MODEL, messages) as gen:
            gen_resp = ollama_generate(messages, model=GENERATOR_MODEL, max_new_tokens=3000)
            refused_by_llm = _looks_like_refusal(gen_resp.content, refusal_string)
            answer_text = refusal_string if refused_by_llm else gen_resp.content
            if gen is not None:
                gen.update(
                    output=answer_text, metadata={"refused_by_llm": refused_by_llm},
                    usage={"input": gen_resp.input_tokens, "output": gen_resp.output_tokens, "unit": "TOKENS"},
                )
        tracing.set_trace_output(trace, answer_text)

        row["refused"] = refused_by_llm
        row["refusal_reason"] = "llm_grounding_refusal" if refused_by_llm else None
        row["answer_text"] = answer_text

        if not refused_by_llm:
            retrieved_context = "\n\n---\n\n".join(c.get("text", "") for c in result.chunks)
            with tracing.trace_judge(trace, JUDGE_MODEL, q.question, q.reference_answer, answer_text) as jgen:
                verdict = _judge(q.question, q.reference_answer, answer_text, retrieved_context)
                if jgen is not None:
                    jgen.update(output=verdict, usage=verdict["usage"])
            row["judge_correct"] = verdict["correct"]
            row["judge_grounded"] = verdict["grounded"]
            row["judge_reasoning"] = verdict["begruendung"]
        else:
            row["judge_correct"] = False
            row["judge_grounded"] = False
            row["judge_reasoning"] = "Unexpected refusal of an answerable silver question (llm_grounding_refusal)"

        if trace is not None and row["judge_correct"] is not None:
            trace.score(name="correct", value=bool(row["judge_correct"]), data_type="BOOLEAN")
            trace.score(name="grounded", value=bool(row["judge_grounded"]), data_type="BOOLEAN")

        print(f"  Silver {q.id} [{q.category}]: refused={row['refused']} "
              f"judge_correct={row['judge_correct']} recall@5={row['recall_hit_at_5']}")
        rows.append(row)

    tracing.flush()
    RESULTS_JSON_PATH.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    rates = compute_silver_rates(rows)

    lines = ["# Silver Dataset Validation Results\n",
             "Diagnostic pass only -- see silver_dataset.md for the question set. "
             "Does NOT feed into or replace the 12-brief-question accuracy numbers.\n"]
    lines.append("| Silver# | Category | Refused | Recall@5 | NDCG@5 | Judge correct | Judge grounded |")
    lines.append("|---------|----------|---------|----------|--------|----------------|-----------------|")
    for r in rows:
        ndcg5 = round(r["ndcg_at_5"], 3) if r["ndcg_at_5"] is not None else None
        lines.append(
            f"| {r['id']} | {r['category']} | {r['refused']} | {r['recall_hit_at_5']} "
            f"| {ndcg5} | {r['judge_correct']} | {r['judge_grounded']} |"
        )
    lines.append("\n## Overall\n")
    for k, v in rates["overall"].items():
        lines.append(f"- **{k}**: {v}")
    lines.append("\n## By category\n")
    for cat, label in CATEGORY_LABELS.items():
        stats = rates["by_category"][cat]
        lines.append(f"- **{label}** (n={stats['n']}): recall@5={stats['recall_at_5']}, "
                     f"ndcg@5={stats['ndcg_at_5']}, answer_accuracy={stats['answer_accuracy_rate']}, "
                     f"unexpected_refusal_rate={stats['unexpected_refusal_rate']}")

    RESULTS_MD_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nResults written to {RESULTS_MD_PATH} and {RESULTS_JSON_PATH}")
    print(json.dumps(rates, indent=2))


if __name__ == "__main__":
    main()
