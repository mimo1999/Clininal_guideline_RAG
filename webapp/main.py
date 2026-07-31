"""FastAPI chatbot backend for clinicians to query the Clinical RAG system.

Generation defaults to a LOCAL transformers model (generation/llm.py's
GENERATOR_MODEL_NAME, "google/gemma-4-E2B-it") -- no Ollama, no account,
no container needed at all for the default path (dev_logs.md Entry 18).
Retrieval + generation models all load in-process at startup and stay
resident for the server's lifetime.

An env var (CLINICAL_RAG_GENERATOR_MODEL) can override the model to an
Ollama tag instead -- e.g. a "-cloud" tag for faster/higher-quality
generation if you do have an Ollama account, or a local Ollama tag if you'd
rather use Ollama's model management than download weights directly via
transformers. Any value other than the transformers default is routed
through generation/ollama_llm.py (kept specifically for this, and for
evaluation/run_eval_cloud_diagnostic.py's cloud-only path -- see that
module's docstring for why it can't move to transformers). The
"cloud_generation" flag in each response and the UI footer reflect whichever
model is actually configured, not a hardcoded assumption.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from evaluation import tracing
from generation.llm import GENERATOR_MODEL_NAME
from generation.llm import generate as local_generate
from generation.prompt import (
    REFUSAL_STRINGS,
    build_context_block,
    build_messages,
    citation_source_pattern,
    detect_query_language,
)
from retrieval.hybrid_search import HybridSearcher

STATIC_DIR = Path(__file__).parent / "static"
LANGFUSE_URL = "http://localhost:3000"

# Overridable, not hardcoded -- e.g. `CLINICAL_RAG_GENERATOR_MODEL=gemma4:31b-cloud`
# to route through Ollama instead of the transformers default. Any value
# other than GENERATOR_MODEL_NAME is treated as an Ollama tag (see chat()
# below) -- there's no third generation backend to disambiguate against.
GENERATOR_MODEL = os.environ.get("CLINICAL_RAG_GENERATOR_MODEL", GENERATOR_MODEL_NAME)
_USE_OLLAMA = GENERATOR_MODEL != GENERATOR_MODEL_NAME

app = FastAPI(title="Clinical Guideline Assistant")

_searcher: HybridSearcher | None = None


def get_searcher() -> HybridSearcher:
    global _searcher
    if _searcher is None:
        _searcher = HybridSearcher()
    return _searcher


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    refused: bool
    refusal_reason: str | None = None
    routed_guideline_ids: list[str] = []
    citations: list[str] = []
    cloud_generation: bool = GENERATOR_MODEL.endswith("-cloud")
    generator_model: str = GENERATOR_MODEL
    # Per-stage wall-clock seconds -- "which step took how long" without
    # guessing. retrieval_timings comes straight from hybrid_search.py's own
    # instrumentation; generation_seconds/total_seconds measured here.
    retrieval_timings: dict[str, float] = {}
    generation_seconds: float | None = None
    total_seconds: float | None = None


@app.on_event("startup")
def _startup() -> None:
    # HybridSearcher() only loads the Chroma/BM25 index (cheap); bge-m3 and
    # the reranker are lazy (@lru_cache) and only actually load onto the GPU
    # on the first real search() call. Run one warm-up query now so that
    # cost lands during startup, not on the clinician's first real message.
    get_searcher().search("Zervixkarzinom Screening")

    if _USE_OLLAMA:
        # Only relevant when CLINICAL_RAG_GENERATOR_MODEL overrides the
        # transformers default to an Ollama tag. Non-blocking -- the server
        # still comes up either way; generation just fails per-request with
        # a clear cause instead of this start failing for an unclear one.
        from generation.ollama_llm import is_available

        if not is_available(GENERATOR_MODEL):
            print(f"WARNING: Ollama isn't reachable, or '{GENERATOR_MODEL}' isn't pulled. "
                  f"Chat will fail until this is fixed -- run `ollama pull {GENERATOR_MODEL}`.")
    else:
        # Loads (and caches) the transformers generator now rather than on
        # the clinician's first message -- same reasoning as the retrieval
        # warm-up query above.
        from generation.llm import load as load_local_generator

        load_local_generator(GENERATOR_MODEL)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/api/config")
def config() -> dict:
    return {"langfuse_url": LANGFUSE_URL, "generator_model": GENERATOR_MODEL}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    t_start = time.perf_counter()
    question = req.question.strip()
    if not question:
        return ChatResponse(answer="", refused=True, refusal_reason="empty_question")

    language = detect_query_language(question)
    refusal_string = REFUSAL_STRINGS[language]

    trace = tracing.create_live_trace(question)

    searcher = get_searcher()
    with tracing.trace_retrieval(trace, question) as span:
        result = searcher.search(question)
        if span is not None:
            span.update(
                output={
                    "refused": result.refused, "refusal_reason": result.refusal_reason,
                    "routed_guideline_ids": result.routed_guideline_ids,
                    "chunk_ids": [c.get("chunk_id") for c in result.chunks],
                },
                metadata={"timings": result.timings},
            )

    if result.refused:
        tracing.set_trace_output(trace, refusal_string)
        tracing.flush()
        return ChatResponse(
            answer=refusal_string, refused=True, refusal_reason=result.refusal_reason,
            routed_guideline_ids=result.routed_guideline_ids,
            retrieval_timings=result.timings, total_seconds=round(time.perf_counter() - t_start, 3),
        )

    messages = build_messages(question, result.chunks, language)
    t_gen = time.perf_counter()
    with tracing.trace_generation(trace, GENERATOR_MODEL, messages) as gen:
        if _USE_OLLAMA:
            from generation.ollama_llm import generate_with_usage as ollama_generate

            gen_resp = ollama_generate(messages, model=GENERATOR_MODEL, max_new_tokens=3000, think=False)
            raw_answer = gen_resp.content
            usage = {"input": gen_resp.input_tokens, "output": gen_resp.output_tokens, "unit": "TOKENS"}
        else:
            # No token-usage figures available from the transformers path
            # the way Ollama's response reports them -- left as None rather
            # than a rough estimate that could be mistaken for a real count.
            raw_answer = local_generate(messages, model_name=GENERATOR_MODEL, max_new_tokens=3000)
            usage = None
        refused_by_llm = refusal_string.lower() in raw_answer.strip().lower()
        if gen is not None:
            gen.update(output=raw_answer, metadata={"refused_by_llm": refused_by_llm}, usage=usage)
    generation_seconds = round(time.perf_counter() - t_gen, 3)

    if refused_by_llm:
        tracing.set_trace_output(trace, refusal_string)
        tracing.flush()
        return ChatResponse(
            answer=refusal_string, refused=True, refusal_reason="llm_grounding_refusal",
            routed_guideline_ids=result.routed_guideline_ids,
            retrieval_timings=result.timings, generation_seconds=generation_seconds,
            total_seconds=round(time.perf_counter() - t_start, 3),
        )

    _, tags = build_context_block(result.chunks, language)
    used_indices = sorted(set(int(m) for m in citation_source_pattern(language).findall(raw_answer)))
    used_tags = [tags[i - 1] for i in used_indices if 1 <= i <= len(tags)]
    if not used_tags:
        # Same fallback as generate.py / run_eval_cloud_diagnostic.py: cite
        # everything given rather than nothing, since those chunks were
        # already retrieval's deterministic output, not a guess.
        used_tags = tags

    tracing.set_trace_output(trace, raw_answer)
    tracing.flush()

    return ChatResponse(
        answer=raw_answer, refused=False, routed_guideline_ids=result.routed_guideline_ids,
        citations=used_tags, retrieval_timings=result.timings, generation_seconds=generation_seconds,
        total_seconds=round(time.perf_counter() - t_start, 3),
    )


@app.post("/api/shutdown")
def shutdown() -> dict:
    """Stops the Langfuse containers, then this server itself. Runs on a
    background thread with a short delay so the HTTP response actually
    reaches the browser before the process exits."""

    def _do_shutdown() -> None:
        import time

        time.sleep(0.5)
        try:
            subprocess.run(
                ["podman", "stop", "langfuse_v2_langfuse-server_1", "langfuse_v2_db_1"],
                timeout=30, capture_output=True,
            )
        except Exception:
            pass  # best-effort -- the server shutdown below still proceeds
        server = getattr(app.state, "server", None)
        if server is not None:
            server.should_exit = True

    threading.Thread(target=_do_shutdown, daemon=True).start()
    return {"status": "shutting_down"}
