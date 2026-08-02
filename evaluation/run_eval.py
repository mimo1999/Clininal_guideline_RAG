"""Two-phase evaluation harness for the brief's 12 questions.

Phase A (retrieval): embedding + reranker models resident on GPU, runs
hybrid_search for every question, caches each SearchResult to disk.
Phase B (generation + judging): releases the retrieval models, loads the
generation LLM alone, generates + judges every question from the cached
retrieval results.

This split exists because bge-m3 + bge-reranker-v2-m3 + Qwen2.5-1.5B-Instruct
don't fit in the GTX 1650's 4GB VRAM simultaneously -- see dev_logs.md Entry 3.
"""

from __future__ import annotations

import csv
import gc
import json
import math
from dataclasses import asdict
from pathlib import Path

from common.progress import ProgressTracker
from evaluation import tracing
from evaluation.judge import judge_answer
from evaluation.questions import QUESTIONS
from generation.generate import answer_question
from generation.llm import JUDGE_MODEL_NAME
from generation.llm import MODEL_NAME as GENERATION_MODEL_NAME, unload as unload_generation_llm
from retrieval.hybrid_search import HybridSearcher, SearchResult
from retrieval.index_store import load_all_chunks

RESULTS_DIR = Path(__file__).parent.parent / "data_corpus"
RETRIEVAL_CACHE_PATH = RESULTS_DIR / "eval_retrieval_cache.json"
RESULTS_MD_PATH = RESULTS_DIR / "eval_results.md"
RESULTS_JSON_PATH = RESULTS_DIR / "eval_results_full.json"
RESULTS_CSV_PATH = RESULTS_DIR / "eval_report.csv"


def phase_a_retrieval(force_rebuild_index: bool = False) -> None:
    print("=== Phase A: retrieval ===")
    searcher = HybridSearcher(force_rebuild=force_rebuild_index)
    tracker = ProgressTracker("eval_phase_a", total=len(QUESTIONS), stage="starting")

    cache = {}
    for q in QUESTIONS:
        tracker.set_stage(f"Q{q.id}")
        trace = tracing.get_or_create_trace(q.id, q.question, q.kind)
        with tracing.trace_retrieval(trace, q.question) as span:
            result = searcher.search(q.question)
            if span is not None:
                span.update(output={
                    "refused": result.refused, "refusal_reason": result.refusal_reason,
                    "routed_guideline_ids": result.routed_guideline_ids, "routed_doc_ids": result.routed_doc_ids,
                    "chunk_ids": [c.get("chunk_id") for c in result.chunks],
                })
        cache[str(q.id)] = asdict(result)
        status = f"refused ({result.refusal_reason})" if result.refused else f"{len(result.chunks)} chunks"
        print(f"  Q{q.id}: {status}")
        tracker.advance(1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RETRIEVAL_CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    tracker.finish()
    tracing.flush()

    # release retrieval models before Phase B loads the generation model --
    # embed._dedup_model (the separate lighter model used only for near-
    # duplicate detection, added after this cleanup was first written) was
    # missing here, so it stayed GPU-resident through all of Phase B, eating
    # into the VRAM budget Qwen needs. Confirmed the real cause of an OOM
    # crash Phase B mid-run once local generation was attempted for the
    # first time with this model in place.
    from retrieval import embed, rerank
    embed._model.cache_clear()
    embed._dedup_model.cache_clear()
    rerank._model.cache_clear()
    del searcher
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _is_match(section: str, expected: str | list[str]) -> bool:
    if not section or not expected: return False
    if isinstance(expected, str): expected = [expected]
    sec_norm = section.strip(" .")
    for exp in expected:
        if sec_norm.startswith(exp.strip(" .")):
            return True
    return False


def _ndcg_at_k(ranked_section_numbers: list[str], expected: str | list[str], k: int) -> float:
    """NDCG for a single relevant document: DCG = 1/log2(rank+1) if the
    expected section appears at 1-indexed `rank` within the top-k, else 0.
    IDCG = 1/log2(2) = 1 (best case: relevant doc at rank 1), so NDCG = DCG.
    Rewards a hit at rank 1 more than a hit at rank 5, unlike plain Recall@k
    which treats every position in the top-k as equivalent."""
    for rank, section in enumerate(ranked_section_numbers[:k], start=1):
        if _is_match(section, expected):
            return 1.0 / math.log2(rank + 1)
    return 0.0


def _load_cached_result(qid: int) -> SearchResult:
    cache = json.loads(RETRIEVAL_CACHE_PATH.read_text(encoding="utf-8"))
    data = cache[str(qid)]
    return SearchResult(**data)


def phase_b_generate_and_judge() -> list[dict]:
    print("=== Phase B: generation + judging ===")
    chunk_lookup = load_all_chunks()  # pure JSON, no GPU needed
    rows = []
    tracker = ProgressTracker("eval_phase_b", total=len(QUESTIONS), stage="starting")

    for q in QUESTIONS:
        tracker.set_stage(f"Q{q.id}:generating")
        result = _load_cached_result(q.id)
        trace = tracing.get_or_create_trace(q.id, q.question, q.kind)

        with tracing.trace_generation(trace, GENERATION_MODEL_NAME, [{"question": q.question}]) as gen:
            answer = answer_question(q.question, result)
            if gen is not None:
                gen.update(output=answer.text, metadata={"refused": answer.refused, "refusal_reason": answer.refusal_reason})
        tracing.set_trace_output(trace, answer.text)

        row = {
            "id": q.id,
            "question": q.question,
            "kind": q.kind,
            "refused": answer.refused,
            "refusal_reason": answer.refusal_reason,
            "answer_text": answer.text,
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
            reranked_sections = [
                chunk_lookup[cid]["section_number"]
                for cid in result.reranked_chunk_ids if cid in chunk_lookup
            ]
            fused_sections = [
                chunk_lookup[cid]["section_number"]
                for cid in result.fused_chunk_ids if cid in chunk_lookup
            ]
            row["recall_hit_at_3"] = any(_is_match(sec, q.expected_section_number) for sec in reranked_sections[:3])
            row["recall_hit_at_5"] = any(_is_match(sec, q.expected_section_number) for sec in reranked_sections[:5])
            row["fused_recall_hit_at_3"] = any(_is_match(sec, q.expected_section_number) for sec in fused_sections[:3])
            row["fused_recall_hit_at_5"] = any(_is_match(sec, q.expected_section_number) for sec in fused_sections[:5])
            row["ndcg_at_5"] = _ndcg_at_k(reranked_sections, q.expected_section_number, 5)
            row["ndcg_at_10"] = _ndcg_at_k(reranked_sections, q.expected_section_number, 10)
            row["fused_ndcg_at_5"] = _ndcg_at_k(fused_sections, q.expected_section_number, 5)
            row["fused_ndcg_at_10"] = _ndcg_at_k(fused_sections, q.expected_section_number, 10)

        if q.reference_answer and not answer.refused:
            tracker.set_stage(f"Q{q.id}:judging")
            retrieved_context = "\n\n---\n\n".join(c.get("text", "") for c in result.chunks)
            with tracing.trace_judge(trace, GENERATION_MODEL_NAME, q.question, q.reference_answer, answer.text) as jgen:
                verdict = judge_answer(q.question, q.reference_answer, answer.text, retrieved_context)
                if jgen is not None:
                    jgen.update(output={"correct": verdict.correct, "grounded": verdict.grounded, "reasoning": verdict.reasoning})
            row["judge_correct"] = verdict.correct
            row["judge_grounded"] = verdict.grounded
            row["judge_reasoning"] = verdict.reasoning
        elif q.reference_answer and answer.refused:
            # an answerable question that got refused is a miss, not judged as "correct"
            row["judge_correct"] = False
            row["judge_grounded"] = False
            row["judge_reasoning"] = f"System refused an answerable question ({answer.refusal_reason})"

        if trace is not None and row["judge_correct"] is not None:
            trace.score(name="correct", value=bool(row["judge_correct"]), data_type="BOOLEAN")
            trace.score(name="grounded", value=bool(row["judge_grounded"]), data_type="BOOLEAN")

        # Unload the judge model between questions -- the generator has to
        # stay resident every iteration (used every question), but the judge
        # is only actually needed for a few seconds per question. Keeping
        # BOTH models GPU-resident for the entire Phase B loop was confirmed
        # to leave too little headroom on a 14.56GB card: Q1 fit, then Q2's
        # generate() call OOM'd with only ~100MB free and ~13.89GB already
        # "allocated by PyTorch" before Q2 even started -- i.e. the combined
        # static footprint of both models alone was already consuming nearly
        # the whole budget, leaving no room for that question's KV cache.
        # Trades ~12s of judge-model reload latency per question for that
        # headroom back during every generate() call, which is what actually
        # needs it. A no-op (safe) call on questions where judging never ran
        # (e.g. Q10-12 traps, refused before reaching the judge).
        unload_generation_llm(JUDGE_MODEL_NAME)

        print(f"  Q{q.id} done: refused={answer.refused} judge_correct={row['judge_correct']}")
        rows.append(row)
        tracker.advance(1)

        # Write after every question, not just once at the end -- Phase B is
        # the slow, GPU-heavy stage (each question issues a long generate +
        # judge call), and a kill or crash mid-run previously lost every
        # completed question's answer/verdict, since they only ever existed
        # in the in-memory `rows` list until the loop finished. Confirmed
        # directly: a real multi-question local Qwen3-1.7B run got killed
        # partway through and there was no way to recover the already-done
        # results. Overwriting the same file each time is cheap at this
        # question count (<=12) and always leaves a valid, complete-so-far
        # JSON array on disk.
        RESULTS_JSON_PATH.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

        # Each question issues a long generation (up to 3000 tokens) plus a
        # long judge call (up to 2000) on the same model -- PyTorch's caching
        # allocator doesn't always return freed KV-cache memory to the driver
        # between calls (same pattern as hybrid_search.py's
        # _release_gpu_memory()), so repeated cycles on this 4GB card can
        # accumulate until a later question OOMs even though each individual
        # call would fit in isolation -- confirmed: a real OOM hit mid-run
        # here before this was added. gc.collect() before empty_cache() --
        # matching phase_a_retrieval()'s cleanup above, missing here before --
        # since Python may not have finalized freed generate()/judge_answer()
        # tensors yet, and empty_cache() alone only returns memory the
        # allocator already knows is unreferenced.
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    tracing.flush()
    unload_generation_llm()
    RESULTS_JSON_PATH.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    tracker.finish()
    return rows


def compute_rates(rows: list[dict]) -> dict:
    answerable = [r for r in rows if r["expected_section_number"]]
    traps = [r for r in rows if r["expected_section_number"] is None]

    def rate(vals):
        vals = [v for v in vals if v is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    return {
        "recall_at_3": rate(r["recall_hit_at_3"] for r in answerable),
        "recall_at_5": rate(r["recall_hit_at_5"] for r in answerable),
        "fused_recall_at_3": rate(r["fused_recall_hit_at_3"] for r in answerable),
        "fused_recall_at_5": rate(r["fused_recall_hit_at_5"] for r in answerable),
        "ndcg_at_5": rate(r["ndcg_at_5"] for r in answerable),
        "ndcg_at_10": rate(r["ndcg_at_10"] for r in answerable),
        "fused_ndcg_at_5": rate(r["fused_ndcg_at_5"] for r in answerable),
        "fused_ndcg_at_10": rate(r["fused_ndcg_at_10"] for r in answerable),
        "answer_accuracy_rate": rate(
            (r["judge_correct"] and r["judge_grounded"]) for r in answerable
        ),
        "refusal_correctness_rate": rate(r["refused"] for r in traps),
    }


def write_report(rows: list[dict], rates: dict) -> None:
    lines = ["# Evaluation Results\n"]
    lines.append("| Q# | Kind | Refused | Recall@3 | Recall@5 | NDCG@5 | NDCG@10 | Judge correct | Judge grounded |")
    lines.append("|----|------|---------|----------|----------|--------|---------|----------------|-----------------|")
    for r in rows:
        ndcg5 = round(r["ndcg_at_5"], 3) if r["ndcg_at_5"] is not None else None
        ndcg10 = round(r["ndcg_at_10"], 3) if r["ndcg_at_10"] is not None else None
        lines.append(
            f"| {r['id']} | {r['kind']} | {r['refused']} | {r['recall_hit_at_3']} | {r['recall_hit_at_5']} "
            f"| {ndcg5} | {ndcg10} | {r['judge_correct']} | {r['judge_grounded']} |"
        )

    lines.append("\n## Rates\n")
    lines.append(f"- **Retrieval Recall@3** (post-rerank): {rates['recall_at_3']} "
                 f"-- fraction of answerable questions (1-9) whose expected section_number appears in the top-3 reranked chunks.")
    lines.append(f"- **Retrieval Recall@5** (post-rerank): {rates['recall_at_5']} "
                 f"-- same, top-5.")
    lines.append(f"- **Retrieval Recall@3 pre-rerank (ablation)**: {rates['fused_recall_at_3']} "
                 f"-- same definition, computed on the RRF-fused list before cross-encoder reranking.")
    lines.append(f"- **Retrieval Recall@5 pre-rerank (ablation)**: {rates['fused_recall_at_5']}")
    lines.append(f"- **NDCG@5** (post-rerank): {rates['ndcg_at_5']} "
                 f"-- mean, over answerable questions, of 1/log2(rank+1) if the expected section appears at position `rank` in the top-5 reranked chunks, else 0 (rewards ranking the right passage higher, unlike plain Recall@k).")
    lines.append(f"- **NDCG@10** (post-rerank): {rates['ndcg_at_10']} -- same, top-10.")
    lines.append(f"- **NDCG@5 pre-rerank (ablation)**: {rates['fused_ndcg_at_5']} "
                 f"-- same definition, on the RRF-fused list before reranking -- the gap vs. NDCG@5 above is how much the cross-encoder rerank improves ranking quality, not just hit/miss.")
    lines.append(f"- **NDCG@10 pre-rerank (ablation)**: {rates['fused_ndcg_at_10']}")
    lines.append(f"- **Answer accuracy rate**: {rates['answer_accuracy_rate']} "
                 f"-- fraction of answerable questions (1-9) judged both correct and grounded by the local LLM-as-judge.")
    lines.append(f"- **Refusal correctness rate**: {rates['refusal_correctness_rate']} "
                 f"-- fraction of trap questions (10-12) correctly refused.")

    RESULTS_MD_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nResults written to {RESULTS_MD_PATH}")


def write_report_csv(rates: dict) -> None:
    """The brief's required "rates" report, one row per rate with a one-line
    definition of how it was computed -- a machine-readable counterpart to
    the prose version in write_report()'s markdown table."""
    rows = [
        {
            "metric": "Retrieval Recall@3",
            "k": 3,
            "value": rates["recall_at_3"],
            "definition": (
                "Fraction of answerable questions (1-9) whose correct evidence "
                "(expected guideline section) appears among the top-3 post-rerank "
                "retrieved chunks."
            ),
        },
        {
            "metric": "Retrieval Recall@5",
            "k": 5,
            "value": rates["recall_at_5"],
            "definition": (
                "Same as Recall@3, computed on the top-5 post-rerank retrieved chunks."
            ),
        },
        {
            "metric": "Answer accuracy rate",
            "k": "",
            "value": rates["answer_accuracy_rate"],
            "definition": (
                "Fraction of answerable questions (1-9) whose generated answer was "
                "judged both correct and grounded (no hallucination) in the retrieved "
                f"text. Judged automatically by a local LLM-as-judge ({JUDGE_MODEL_NAME}, "
                "a model separate from the generator to avoid self-judging bias, "
                "constrained to structured JSON output). Judge reliability was validated "
                "by cross-checking its verdicts against an independent cloud judge on a "
                "prior run: the cloud judge disagreed with a same-model self-judge's "
                "lenient verdicts on 6 of 9 questions (dev_logs.md Entry 14), which is "
                "why generator and judge are required to be different models here."
            ),
        },
        {
            "metric": "Refusal correctness rate",
            "k": "",
            "value": rates["refusal_correctness_rate"],
            "definition": (
                "Fraction of trap questions (10-12, no answer exists in the corpus) "
                "correctly refused rather than confidently (and wrongly) answered."
            ),
        },
        {
            "metric": "NDCG@5",
            "k": 5,
            "value": rates["ndcg_at_5"],
            "definition": (
                "Mean, over answerable questions, of 1/log2(rank+1) if the expected "
                "section appears at position `rank` in the top-5 post-rerank chunks, "
                "else 0. Unlike Recall@k, this rewards ranking the correct passage "
                "higher rather than treating every top-k position as equivalent -- "
                "included as an additional honest rate on ranking quality, not just hit/miss."
            ),
        },
    ]

    with RESULTS_CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "k", "value", "definition"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV report written to {RESULTS_CSV_PATH}")


def main():
    # force_rebuild=False: build_indexes() already self-heals if the index is
    # stale (collection.count() != len(chunk_ids) triggers a rebuild on its
    # own) or was never built. Forcing unconditionally here used to be needed
    # once (the embedding model moved from CPU/fp32 to GPU/fp16 after an
    # earlier index build), but is now just a wasted ~14min re-embed of a
    # perfectly valid index on every run.
    phase_a_retrieval(force_rebuild_index=False)
    rows = phase_b_generate_and_judge()
    rates = compute_rates(rows)
    write_report(rows, rates)
    write_report_csv(rates)
    print(json.dumps(rates, indent=2))


if __name__ == "__main__":
    main()
