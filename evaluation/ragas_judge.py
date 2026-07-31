"""Ragas-based alternative to judge.py's hand-rolled LLM-as-judge -- a
potential replacement, not a mandated swap (see run_eval.py: judge.py stays
the default; this is wired in behind a flag so both can be compared on the
same questions). Same JudgeVerdict-shaped return as judge.py so it's a
drop-in.

Metric mapping onto the existing "correct"/"grounded" interface:
- grounded  -> ragas Faithfulness: are the generated answer's claims
  supported by the retrieved chunks? Same intent as judge.py's "grounded"
  criterion, but decomposes the answer into individual claims and checks
  each one against context rather than a single holistic LLM judgment --
  more expensive (more LLM calls per question) but more auditable (you can
  see which specific claim failed).
- correct   -> ragas AnswerCorrectness: blends factual overlap and semantic
  similarity against the reference answer into a continuous 0-1 score.
  Thresholded at ANSWER_CORRECTNESS_THRESHOLD to produce the boolean
  `correct` the rest of the harness expects -- this is a real, lossy
  simplification (a 0.51 and a 0.95 both become True), noted rather than
  hidden, since judge.py's single-shot boolean has the opposite failure mode
  (no visibility into "how correct").

Backend: ragas' metrics are themselves LLM-based judgments, so they need a
LangChain-compatible LLM + embeddings model to run at all -- this isn't a
free/local-only metric library. Embeddings reuse this project's own dense
embedder (retrieval/embed.py, whatever its current MODEL_NAME is) via a
small shim below, so no new model or download is needed there. The judge LLM
defaults to nemotron-3-nano:30b-cloud
via Ollama Cloud, matching the same diagnostic-only cloud model already used
by judge.py's Ollama-backed comparison path (generation/ollama_llm.py,
run_eval_cloud_diagnostic.py) -- current project direction is cloud models
for all LLM-related tasks while other things (VRAM, chunking, routing) get
fixed up; NOT the submitted system's final choice, same caveat as everywhere
else cloud models appear in this codebase. A locally-pulled model (e.g.
qwen3:1.7b) also works via `judge_model`/`base_url`, but was measured
unworkably slow for this multi-call metric pipeline on this hardware -- a
single trivial "Say OK" call took ~42s because it's a reasoning/"thinking"
model that generates a lengthy internal trace before any real output, and
Faithfulness alone issues several LLM calls per question (claim
decomposition + one verification call per claim).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from functools import lru_cache

DEFAULT_JUDGE_MODEL = "nemotron-3-nano:30b-cloud"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
ANSWER_CORRECTNESS_THRESHOLD = 0.5


@dataclass
class RagasJudgeVerdict:
    correct: bool
    grounded: bool
    correctness_score: float  # raw 0-1 ragas AnswerCorrectness score, before thresholding
    faithfulness_score: float  # raw 0-1 ragas Faithfulness score
    reasoning: str


class _ProjectEmbeddings:
    """LangChain Embeddings-protocol shim over retrieval/embed.py's dense
    embedder -- lets ragas reuse the project's already-loaded local embedding
    model instead of requiring a separate Ollama embedding model pull. A thin
    pass-through, not tied to any specific model: whatever embed.py's
    MODEL_NAME currently is (bge-m3, then granite-embedding-278m-multilingual
    after the Entry 10 swap) is what this uses -- named generically so it
    doesn't go stale the next time that model changes, unlike its previous
    name (_Bgem3Embeddings) which silently kept referring to a model this
    class hadn't actually used since that swap."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        from retrieval.embed import embed_texts

        return embed_texts(list(texts)).tolist()

    def embed_query(self, text: str) -> list[float]:
        from retrieval.embed import embed_query

        return embed_query(text).tolist()

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)


@lru_cache(maxsize=4)
def _ragas_llm(judge_model: str, base_url: str):
    from langchain_ollama import ChatOllama
    from ragas.llms import LangchainLLMWrapper

    # reasoning=False: nemotron-3-nano (and other Ollama "thinking" models)
    # emit a chain-of-thought pass before the real answer. Beyond the
    # token-budget risk that alone poses (see dev_logs.md Entry 4), it
    # actively broke ragas here -- confirmed via a full 12-question run that
    # crashed with RagasOutputParserException: ragas' PydanticPrompt parser
    # expects clean structured output and has no tolerance for a thinking
    # preamble wrapped around it, and its own self-correction retries
    # (fix_output_format_prompt) still failed against the polluted output
    # until the retry budget ran out. Disabling reasoning mode outright (via
    # ChatOllama's `reasoning` field, confirmed present via model_fields)
    # is the real fix, not a bigger token budget.
    return LangchainLLMWrapper(
        ChatOllama(model=judge_model, base_url=base_url, temperature=0, num_predict=2000, reasoning=False)
    )


@lru_cache(maxsize=1)
def _ragas_embeddings():
    from ragas.embeddings import LangchainEmbeddingsWrapper

    return LangchainEmbeddingsWrapper(_ProjectEmbeddings())


def _metrics(judge_model: str, base_url: str):
    from ragas.metrics import AnswerCorrectness, AnswerSimilarity, Faithfulness

    llm = _ragas_llm(judge_model, base_url)
    embeddings = _ragas_embeddings()
    # AnswerCorrectness's `embeddings=` constructor kwarg alone doesn't wire
    # up its semantic-similarity component -- confirmed via a real
    # AssertionError ("AnswerSimilarity must be set") from inside _ascore().
    # It needs an explicit AnswerSimilarity instance.
    answer_similarity = AnswerSimilarity(embeddings=embeddings)
    return (
        Faithfulness(llm=llm),
        AnswerCorrectness(llm=llm, embeddings=embeddings, answer_similarity=answer_similarity),
    )


async def _score_async(sample, faithfulness, answer_correctness) -> tuple[float, float]:
    # Independent metrics -- run concurrently rather than sequentially, since
    # each is its own round-trip of LLM calls.
    faith_score, correct_score = await asyncio.gather(
        faithfulness.single_turn_ascore(sample),
        answer_correctness.single_turn_ascore(sample),
    )
    return faith_score, correct_score


def ragas_judge_answer(
    question: str,
    reference_answer: str,
    generated_answer: str,
    retrieved_contexts: list[str],
    judge_model: str = DEFAULT_JUDGE_MODEL,
    base_url: str = DEFAULT_OLLAMA_BASE_URL,
) -> RagasJudgeVerdict:
    from ragas.dataset_schema import SingleTurnSample

    sample = SingleTurnSample(
        user_input=question,
        response=generated_answer,
        reference=reference_answer,
        retrieved_contexts=retrieved_contexts or [generated_answer],  # Faithfulness requires a non-empty list
    )
    faithfulness, answer_correctness = _metrics(judge_model, base_url)
    faith_score, correct_score = asyncio.run(_score_async(sample, faithfulness, answer_correctness))

    return RagasJudgeVerdict(
        correct=correct_score >= ANSWER_CORRECTNESS_THRESHOLD,
        grounded=faith_score >= ANSWER_CORRECTNESS_THRESHOLD,
        correctness_score=correct_score,
        faithfulness_score=faith_score,
        reasoning=(
            f"ragas answer_correctness={correct_score:.3f} (threshold {ANSWER_CORRECTNESS_THRESHOLD}), "
            f"faithfulness={faith_score:.3f}"
        ),
    )
