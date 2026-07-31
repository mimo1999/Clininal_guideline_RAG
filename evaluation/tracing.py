"""Langfuse tracing for the eval harness -- wraps each question's
retrieval -> generation -> judge pipeline as one trace with nested spans,
viewable in the self-hosted Langfuse UI at http://localhost:3000 (see
../../langfuse_v2/ for the docker-compose setup: Langfuse **v2**, Postgres
only -- the current v3 architecture needs Postgres+ClickHouse+Redis and
recommends 8GB RAM, well past what this machine can spare alongside the local
GPU models, so v2's lighter 2-container footprint was the deliberate choice.
The Python SDK is pinned to the 2.x line to match -- the latest SDK (4.x) is
built for v3's API and fails auth_check() against a v2 server (confirmed: a
Pydantic validation error on missing `organization`/`metadata` fields the v2
API doesn't return).

Fully local: LANGFUSE_HOST defaults to the self-hosted instance, never
Langfuse's hosted cloud. If the server isn't reachable, tracing degrades to a
no-op (auth_check() failure caught once at import time) so the eval harness
itself never breaks because of an observability side-channel.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

_DEFAULTS = {
    "LANGFUSE_HOST": "http://localhost:3000",
}


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (no new dependency) -- only sets vars not already
    present in the environment, so an explicit shell export still wins."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


_load_dotenv(Path(__file__).parent.parent / ".env")
for _k, _v in _DEFAULTS.items():
    os.environ.setdefault(_k, _v)

_client = None
_enabled = False
_checked = False


def _get_client():
    global _client, _enabled, _checked
    if _client is None:
        from langfuse import Langfuse

        _client = Langfuse()
    if not _checked:
        _checked = True
        try:
            _enabled = _client.auth_check()
        except Exception as e:
            _enabled = False
            print(f"  [tracing] Langfuse not reachable/authenticated ({e}) -- continuing without tracing")
    return _client


def enabled() -> bool:
    _get_client()
    return _enabled


_TRACE_ID_PREFIX = "cgr3"  # bump this if the target Langfuse project ever changes again, OR if
# a trace's field set changes shape (e.g. this bump: removing `kind`/`run_label`/`question_id`
# from trace metadata) -- Langfuse merges metadata at the key level on upsert, not
# wholesale-replaces it, so omitting a field (or even passing metadata={}) on an
# already-existing trace id does NOT clear what a prior run already wrote there
# (confirmed directly: re-upserting cgr2-cloud-q1 with metadata={} left the old
# kind/run_label/question_id keys untouched). A fresh id sidesteps the merge
# entirely rather than fighting it.


def get_or_create_trace(question_id: int, question: str, kind: str, run_label: str = "local"):
    """Returns the trace for one evaluation question, or None if Langfuse
    isn't reachable. Uses a deterministic id (`{_TRACE_ID_PREFIX}-{run_label}-q{id}`)
    rather than a freshly generated one -- Phase A (retrieval) and Phase B
    (generation+judge) run as SEPARATE processes, often minutes to hours
    apart, so there's no live Python object to pass between them. Calling
    `.trace(id=<same id>, ...)` again from a fresh client in Phase B upserts
    onto the same trace record (confirmed empirically: two independent
    client instances each adding one observation to the same id both show up
    under one trace, not two) -- that's what lets a question's retrieval
    span and its later generation/judge spans end up nested together in the
    UI despite never sharing a process.

    Trace ids are claimed instance-wide, not scoped per project -- switching
    which Langfuse project's API key the SDK uses (e.g. moving to a fresh
    project) while reusing the SAME id the old project already created hits
    a `ForbiddenError: Access denied for trace creation <old-project>` on
    every write, since a different project can't take over an id another
    project already owns (confirmed directly via the server logs, not
    guessed). `_TRACE_ID_PREFIX` exists specifically so a future project
    switch is one constant, not a silent multi-minute string of failed
    writes to debug again.

    `run_label` distinguishes concurrent/repeated eval runs over the SAME
    question set (e.g. the local Qwen run vs. the Ollama cloud diagnostic,
    see run_eval.py vs run_eval_cloud_diagnostic.py) -- without it, both
    scripts would upsert onto the same trace id per question and their
    generations would get mixed together under one trace, which would make
    "what model actually produced this answer" ambiguous in the UI.

    `kind` (gold/self_labeled/trap) is accepted for the caller's convenience
    but deliberately NOT written to the trace -- for trap questions it's an
    answer key ("this one should be refused"), and even though it never
    reaches the LLM itself (build_messages() never sees it, confirmed by
    inspection), there's no reason for it to sit visible on the trace in the
    UI either. The trace is scoped to exactly question number + question +
    generated response (set via set_trace_output() once the answer is
    known) -- nothing else."""
    if not enabled():
        return None
    return _get_client().trace(
        id=f"{_TRACE_ID_PREFIX}-{run_label}-q{question_id}",
        name=f"eval-{run_label}-question-{question_id}",
        input={"question_number": question_id, "question": question},
    )


def create_live_trace(question: str):
    """Fresh trace for one live chat request (webapp/main.py) -- unlike
    get_or_create_trace()'s deterministic id (built specifically so the eval
    harness's separate Phase A/Phase B processes can resume the SAME trace),
    a live request never needs to be resumed from a different process: it's
    retrieval + generation in one call, in one process. A Langfuse-assigned
    random id is simpler and correct here, and sidesteps the "id already
    claimed by a different project" ForbiddenError entirely."""
    if not enabled():
        return None
    return _get_client().trace(name="webapp-chat", input={"question": question})


def set_trace_output(trace, output_text: str) -> None:
    """Sets the trace's own top-level output (the generated response) so
    it's visible directly on the trace in the UI, not only nested inside its
    generation span."""
    if trace is None:
        return
    trace.update(output=output_text)


@contextmanager
def trace_retrieval(trace, query: str):
    if trace is None:
        yield None
        return
    span = trace.span(name="retrieval", input={"query": query})
    try:
        yield span
    finally:
        span.end()


@contextmanager
def trace_generation(trace, model: str, messages: list[dict]):
    if trace is None:
        yield None
        return
    gen = trace.generation(name="generation", model=model, input=messages)
    try:
        yield gen
    finally:
        gen.end()


@contextmanager
def trace_judge(trace, model: str, question: str, reference_answer: str, generated_answer: str):
    if trace is None:
        yield None
        return
    gen = trace.generation(
        name="judge", model=model,
        input={"question": question, "reference_answer": reference_answer, "generated_answer": generated_answer},
    )
    try:
        yield gen
    finally:
        gen.end()


def flush():
    if _client is not None:
        _client.flush()
