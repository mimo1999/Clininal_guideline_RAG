"""Diagnostic-only re-run of Phase B using Ollama cloud models
(gemma4:31b-cloud generator, nemotron-3-nano:30b-cloud judge) against the
SAME cached retrieval results and SAME prompts as the local Qwen2.5-1.5B run.
Purpose: isolate whether the local run's low answer-accuracy rate is a
model-capability ceiling or a harness/prompt bug. NOT the submitted system --
see generation/ollama_llm.py's module docstring and dev_logs.md Entry 4.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from evaluation import tracing
from evaluation.questions import QUESTIONS
from evaluation.run_eval import RESULTS_DIR, RETRIEVAL_CACHE_PATH, _ndcg_at_k, compute_rates, _is_match
from generation.ollama_llm import generate_with_usage as ollama_generate
from generation.prompt import REFUSAL_STRINGS, build_context_block, build_messages, detect_query_language
from evaluation.judge import JUDGE_SYSTEM_PROMPT, JUDGE_RESPONSE_SCHEMA
from retrieval.hybrid_search import SearchResult
from retrieval.index_store import load_all_chunks

GENERATOR_MODEL = "gemma4:31b-cloud"
JUDGE_MODEL = "nemotron-3-super:cloud"

RESULTS_JSON_PATH = RESULTS_DIR / "eval_results_cloud_full.json"
RESULTS_MD_PATH = RESULTS_DIR / "eval_results_cloud.md"


def _load_cached_result(qid: int) -> SearchResult:
    cache = json.loads(RETRIEVAL_CACHE_PATH.read_text(encoding="utf-8"))
    data = cache[str(qid)]
    return SearchResult(**data)


def _looks_like_refusal(text: str, refusal_string: str) -> bool:
    return refusal_string.lower() in text.strip().lower()


def _judge(question: str, reference_answer: str, generated_answer: str, retrieved_context: str = "") -> dict:
    # retrieved_context: same rationale as evaluation/judge.py's judge_answer()
    # -- lets `grounded` be checked against the actual retrieved evidence
    # instead of only the reference_answer string. Optional/defaults to ""
    # for backward compatibility.
    context_block = f"\n\nAbgerufener Quelltext (Basis der generierten Antwort):\n{retrieved_context}" if retrieved_context else ""
    user_prompt = f"""Frage: {question}

Referenzantwort: {reference_answer}
{context_block}

Generierte Antwort: {generated_answer}

Bewertung (nur JSON):"""
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    # nemotron-3-nano is a reasoning model -- think=False (below) suppresses
    # the <think> block, but NOT its tendency to still write verbose
    # step-by-step prose as regular content (confirmed directly for
    # qwen3:4b, dev_logs.md Entry 17 -- the same underlying mechanism
    # applies here). `format` is what actually fixes this: constrains
    # output server-side to valid JSON instead of hoping free-form prose
    # happens to end with parseable JSON.
    resp = ollama_generate(messages, model=JUDGE_MODEL, max_new_tokens=2000, think=False, format=JUDGE_RESPONSE_SCHEMA)
    raw, usage = resp.content, {"input": resp.input_tokens, "output": resp.output_tokens, "unit": "TOKENS"}
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {"correct": False, "grounded": False, "begruendung": "judge output unparseable", "raw": raw, "usage": usage}
    try:
        data = json.loads(match.group(0))
        return {
            "correct": bool(data.get("correct")),
            "grounded": bool(data.get("grounded")),
            "begruendung": str(data.get("begruendung", "")),
            "raw": raw,
            "usage": usage,
        }
    except json.JSONDecodeError:
        return {"correct": False, "grounded": False, "begruendung": "judge output invalid JSON", "raw": raw, "usage": usage}


def main(compare_ragas: bool = False):
    chunk_lookup = load_all_chunks()
    rows = []

    for q in QUESTIONS:
        result = _load_cached_result(q.id)
        language = detect_query_language(q.question)
        refusal_string = REFUSAL_STRINGS[language]
        # Own trace family ("cloud"), distinct from run_eval.py's "local" --
        # both scripts share the SAME cached retrieval (Phase A only ever
        # runs once), so re-log it here too rather than leaving this trace
        # family without retrieval context; each run's trace stays
        # self-contained and independently viewable in the UI.
        trace = tracing.get_or_create_trace(q.id, q.question, q.kind, run_label="cloud")
        if trace is not None:
            with tracing.trace_retrieval(trace, q.question) as span:
                if span is not None:
                    span.update(output={
                        "refused": result.refused, "refusal_reason": result.refusal_reason,
                        "routed_guideline_ids": result.routed_guideline_ids, "routed_doc_ids": result.routed_doc_ids,
                        "chunk_ids": [c.get("chunk_id") for c in result.chunks],
                    })

        if result.refused:
            row = {
                "id": q.id, "question": q.question, "kind": q.kind,
                "refused": True, "refusal_reason": result.refusal_reason,
                "answer_text": refusal_string, "raw_answer_text": refusal_string,
                "expected_section_number": q.expected_section_number,
                "recall_hit_at_3": None, "recall_hit_at_5": None,
                "fused_recall_hit_at_3": None, "fused_recall_hit_at_5": None,
                "ndcg_at_5": None, "ndcg_at_10": None,
                "fused_ndcg_at_5": None, "fused_ndcg_at_10": None,
                "judge_correct": None, "judge_grounded": None, "judge_reasoning": None,
                "ragas_correct": None, "ragas_grounded": None,
                "ragas_correctness_score": None, "ragas_faithfulness_score": None,
            }
            tracing.set_trace_output(trace, refusal_string)
            print(f"  Q{q.id}: refused ({result.refusal_reason}) -- skipping generation")
            rows.append(row)
            continue

        messages = build_messages(q.question, result.chunks, language)
        with tracing.trace_generation(trace, GENERATOR_MODEL, messages) as gen:
            gen_resp = ollama_generate(messages, model=GENERATOR_MODEL, max_new_tokens=3000)
            raw_answer = gen_resp.content
            refused_by_llm = _looks_like_refusal(raw_answer, refusal_string)
            answer_text = refusal_string if refused_by_llm else raw_answer
            if gen is not None:
                gen.update(
                    output=answer_text, metadata={"refused_by_llm": refused_by_llm},
                    usage={"input": gen_resp.input_tokens, "output": gen_resp.output_tokens, "unit": "TOKENS"},
                )
        tracing.set_trace_output(trace, answer_text)

        row = {
            "id": q.id,
            "question": q.question,
            "kind": q.kind,
            "refused": refused_by_llm,
            "refusal_reason": "llm_grounding_refusal" if refused_by_llm else None,
            "answer_text": answer_text,
            "raw_answer_text": raw_answer,
            "expected_section_number": q.expected_section_number,
            "recall_hit_at_3": None,
            "recall_hit_at_5": None,
            "fused_recall_hit_at_3": None,
            "fused_recall_hit_at_5": None,
            "ndcg_at_5": None,
            "ndcg_at_10": None,
            "fused_ndcg_at_5": None,
            "fused_ndcg_at_10": None,
            "judge_correct": None,
            "judge_grounded": None,
            "judge_reasoning": None,
        }

        if q.expected_section_number:
            reranked_sections = [chunk_lookup[cid]["section_number"] for cid in result.reranked_chunk_ids if cid in chunk_lookup]
            fused_sections = [chunk_lookup[cid]["section_number"] for cid in result.fused_chunk_ids if cid in chunk_lookup]
            row["recall_hit_at_3"] = any(_is_match(sec, q.expected_section_number) for sec in reranked_sections[:3])
            row["recall_hit_at_5"] = any(_is_match(sec, q.expected_section_number) for sec in reranked_sections[:5])
            row["fused_recall_hit_at_3"] = any(_is_match(sec, q.expected_section_number) for sec in fused_sections[:3])
            row["fused_recall_hit_at_5"] = any(_is_match(sec, q.expected_section_number) for sec in fused_sections[:5])
            row["ndcg_at_5"] = _ndcg_at_k(reranked_sections, q.expected_section_number, 5)
            row["ndcg_at_10"] = _ndcg_at_k(reranked_sections, q.expected_section_number, 10)
            row["fused_ndcg_at_5"] = _ndcg_at_k(fused_sections, q.expected_section_number, 5)
            row["fused_ndcg_at_10"] = _ndcg_at_k(fused_sections, q.expected_section_number, 10)

        if q.reference_answer and not refused_by_llm:
            retrieved_context = "\n\n---\n\n".join(c.get("text", "") for c in result.chunks)
            with tracing.trace_judge(trace, JUDGE_MODEL, q.question, q.reference_answer, answer_text) as jgen:
                verdict = _judge(q.question, q.reference_answer, answer_text, retrieved_context)
                if jgen is not None:
                    jgen.update(output=verdict, usage=verdict["usage"])
            row["judge_correct"] = verdict["correct"]
            row["judge_grounded"] = verdict["grounded"]
            row["judge_reasoning"] = verdict["begruendung"]
        elif q.reference_answer and refused_by_llm:
            row["judge_correct"] = False
            row["judge_grounded"] = False
            row["judge_reasoning"] = "System refused an answerable question (llm_grounding_refusal)"

        # Optional side-by-side comparison against the hand-rolled judge
        # above -- opt-in (not run by default) since ragas' two metrics each
        # issue their own LLM calls (Faithfulness decomposes the answer into
        # claims and verifies each one separately), meaningfully slower per
        # question than the single-shot judge.
        row["ragas_correct"] = None
        row["ragas_grounded"] = None
        row["ragas_correctness_score"] = None
        row["ragas_faithfulness_score"] = None
        if compare_ragas and q.reference_answer and not refused_by_llm:
            from evaluation.ragas_judge import ragas_judge_answer

            retrieved_contexts = [c.get("text", "") for c in result.chunks]
            with tracing.trace_judge(trace, "ragas:" + JUDGE_MODEL, q.question, q.reference_answer, answer_text) as rgen:
                ragas_verdict = ragas_judge_answer(
                    q.question, q.reference_answer, answer_text, retrieved_contexts,
                )
                if rgen is not None:
                    rgen.update(output={
                        "correct": ragas_verdict.correct, "grounded": ragas_verdict.grounded,
                        "correctness_score": ragas_verdict.correctness_score,
                        "faithfulness_score": ragas_verdict.faithfulness_score,
                    })
            row["ragas_correct"] = ragas_verdict.correct
            row["ragas_grounded"] = ragas_verdict.grounded
            row["ragas_correctness_score"] = ragas_verdict.correctness_score
            row["ragas_faithfulness_score"] = ragas_verdict.faithfulness_score
            if trace is not None:
                trace.score(name="ragas_correct", value=ragas_verdict.correct, data_type="BOOLEAN")
                trace.score(name="ragas_grounded", value=ragas_verdict.grounded, data_type="BOOLEAN")
                trace.score(name="ragas_correctness_score", value=ragas_verdict.correctness_score, data_type="NUMERIC")
                trace.score(name="ragas_faithfulness_score", value=ragas_verdict.faithfulness_score, data_type="NUMERIC")
            print(f"    ragas: correct={ragas_verdict.correct} grounded={ragas_verdict.grounded} "
                  f"(scores {ragas_verdict.correctness_score:.3f}/{ragas_verdict.faithfulness_score:.3f})")

        if trace is not None and row["judge_correct"] is not None:
            trace.score(name="correct", value=bool(row["judge_correct"]), data_type="BOOLEAN")
            trace.score(name="grounded", value=bool(row["judge_grounded"]), data_type="BOOLEAN")

        print(f"  Q{q.id} done: refused={refused_by_llm} judge_correct={row['judge_correct']}")
        rows.append(row)

    tracing.flush()
    RESULTS_JSON_PATH.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    rates = compute_rates(rows)

    # write to the cloud-specific report path, not the local one
    lines = ["# Evaluation Results -- Ollama Cloud Diagnostic (gemma4:31b-cloud / nemotron-3-nano:30b-cloud)\n"]
    header = "| Q# | Kind | Refused | Recall@3 | Recall@5 | NDCG@5 | NDCG@10 | Judge correct | Judge grounded |"
    sep = "|----|------|---------|----------|----------|--------|---------|----------------|-----------------|"
    if compare_ragas:
        header += " Ragas correct | Ragas grounded | Ragas scores (correctness/faithfulness) |"
        sep += "----------------|-----------------|--------------------------------------------|"
    lines.append(header)
    lines.append(sep)
    for r in rows:
        ndcg5 = round(r["ndcg_at_5"], 3) if r["ndcg_at_5"] is not None else None
        ndcg10 = round(r["ndcg_at_10"], 3) if r["ndcg_at_10"] is not None else None
        line = (
            f"| {r['id']} | {r['kind']} | {r['refused']} | {r['recall_hit_at_3']} | {r['recall_hit_at_5']} "
            f"| {ndcg5} | {ndcg10} | {r['judge_correct']} | {r['judge_grounded']} |"
        )
        if compare_ragas:
            cs = round(r["ragas_correctness_score"], 3) if r["ragas_correctness_score"] is not None else None
            fs = round(r["ragas_faithfulness_score"], 3) if r["ragas_faithfulness_score"] is not None else None
            line += f" {r['ragas_correct']} | {r['ragas_grounded']} | {cs}/{fs} |"
        lines.append(line)
    lines.append("\n## Rates\n")
    for k, v in rates.items():
        lines.append(f"- **{k}**: {v}")

    if compare_ragas:
        judged = [r for r in rows if r["judge_correct"] is not None and r["ragas_correct"] is not None]
        agree_correct = sum(1 for r in judged if r["judge_correct"] == r["ragas_correct"])
        agree_grounded = sum(1 for r in judged if r["judge_grounded"] == r["ragas_grounded"])
        lines.append("\n## Judge comparison (hand-rolled vs. ragas)\n")
        lines.append(f"- **correct agreement**: {agree_correct}/{len(judged)} questions" if judged else "- no comparable questions")
        lines.append(f"- **grounded agreement**: {agree_grounded}/{len(judged)} questions" if judged else "")

    RESULTS_MD_PATH.write_text("\n".join(lines), encoding="utf-8")

    print(f"\nResults written to {RESULTS_MD_PATH} and {RESULTS_JSON_PATH}")
    print(json.dumps(rates, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compare-ragas", action="store_true",
        help="Also judge each answer with evaluation/ragas_judge.py and report side-by-side agreement with the hand-rolled judge.",
    )
    args = parser.parse_args()
    main(compare_ragas=args.compare_ragas)
