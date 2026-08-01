"""transformers-based wrappers for the two local models used by this
system's local-only path: GENERATOR_MODEL_NAME for answer generation
(google/gemma-4-E2B-it, multimodal-capable but used text-only here),
JUDGE_MODEL_NAME for the evaluation LLM-as-judge (Qwen/Qwen3.5-2B).

Fifth design for this module in one session (dev_logs.md Entries 14, 16, 17,
18, 19) -- worth staying explicit about why, since each prior attempt was a
reasonable try that hit a real, specific wall:
1. HuggingFace transformers (Qwen2.5-1.5B, then Qwen3-1.7B/"Gemma4-4B"):
   repeated OOMs and severe thermal throttling on this project's 4GB dev GPU
   even at 1.7B params (Entry 13); the "Gemma4-4B" HF repo id used then
   ("google/gemma-4-4b-it") turned out not to exist at all -- wrong naming
   convention, not a real model (Entry 14/16).
2. llama-cpp-python + GGUF: solved the memory problem (4-bit quantization),
   but introduced real friction of its own -- a Windows source-build
   failure (long-path limits), a gated official Google GGUF repo requiring
   HF auth, and no first-class way to disable Qwen3's "thinking" mode.
3. Local (non-`-cloud`) Ollama model tags (gemma3:4b / qwen3:4b): worked
   well and was verified end-to-end (Entry 17) -- but the user then wanted
   to try the ACTUAL Gemma 4 family, `gemma4:e4b` (the "E4B" effective-
   parameter naming, not "4b" -- Entry 16's dead end was a naming guess,
   this is the real thing), which crashed Ollama's own llama-server backend
   outright on this GPU (`GGML_ASSERT` stack-buffer-overrun), reproduced
   twice and still present after upgrading Ollama 0.32.3 -> 0.32.5 -- ruling
   out "stale version" as the cause.
4. `google/gemma-4-E2B-it` (correct HF naming, confirmed to exist and be
   ungated via the Hub API before use) via transformers directly, alongside
   `Qwen/Qwen3.5-4B` as judge (Entry 18). Verified the generator actually
   loads and generates correctly on this machine (a real joke about RAM,
   cleanly parsed) once `device_map="auto"` was swapped for an explicit
   fallback -- "auto" misjudged available capacity and tried an unsupported
   whole-model disk-offload path (see _from_pretrained_with_device_fallback
   below). Judge reliability was left explicitly open at this point: this
   dev machine's CPU-only generation was too slow to wait out a real test of
   whether Qwen3.5-4B's `enable_thinking=False` actually suppresses
   reasoning-as-content the way Ollama's `think=False` failed to (Entry 17).
5. This version: the user independently verified a specific config on a
   Colab GPU that resolves point 4's open question -- `torch_dtype=
   torch.bfloat16` (not `dtype="auto"`), `apply_chat_template(...,
   tokenize=False, ...)` followed by a separate `tokenizer()` call (not the
   combined `tokenize=True, return_dict=True` call the generator uses), and
   `do_sample=True, temperature=0.7, top_p=0.8, top_k=20`. Ported here
   exactly for the judge path (Entry 19) -- the generator's own,
   separately-verified config from point 4 is unchanged.

generation/ollama_llm.py is NOT deleted -- kept intentionally as a working,
tested reference implementation. It solved a problem this module still has
no wired-in equivalent for: Ollama's `format=` JSON-schema parameter
reliably constrains judge output to valid JSON server-side (Entry 17). This
module's `generate()` accepts `format` for interface compatibility but
ignores it -- judge reliability here rests on the verified sampling config
above plus evaluation/judge.py's own regex+json.loads parsing (with its
existing "unparseable"/"invalid JSON" fallback), not a hard schema
guarantee. ollama_llm.py also stays actively used by
evaluation/run_eval_cloud_diagnostic.py regardless of this module's
choices -- the "-cloud" tags it uses (gemma4:31b-cloud,
nemotron-3-nano:30b-cloud) are Ollama-hosted proprietary relay models with
no local weights and therefore no transformers equivalent at all, not a
design choice to reconsider.
"""

from __future__ import annotations

GENERATOR_MODEL_NAME = "google/gemma-4-E2B-it"
JUDGE_MODEL_NAME = "Qwen/Qwen3.5-2B"

# Back-compat alias -- evaluation/run_eval.py imports this name purely for
# Langfuse trace labeling; kept pointing at the generator (the model that
# actually produces the traced answer), not the judge.
MODEL_NAME = GENERATOR_MODEL_NAME

_state: dict = {}


def _load_generator():
    if "generator" in _state:
        return _state["generator"]

    from transformers import AutoModelForMultimodalLM, AutoProcessor

    processor = AutoProcessor.from_pretrained(GENERATOR_MODEL_NAME)
    model = _from_pretrained_with_device_fallback(AutoModelForMultimodalLM, GENERATOR_MODEL_NAME)
    _state["generator"] = (processor, model)
    return _state["generator"]


def _load_judge():
    if "judge" in _state:
        return _state["judge"]

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL_NAME)
    # torch_dtype=torch.bfloat16 (not dtype="auto") -- matches the exact
    # config the user independently verified working on a Colab GPU
    # (dev_logs.md Entry 19), not a guess.
    model = _from_pretrained_with_device_fallback(AutoModelForCausalLM, JUDGE_MODEL_NAME, torch.bfloat16)
    _state["judge"] = (tokenizer, model)
    return _state["judge"]


def _from_pretrained_with_device_fallback(model_cls, model_id: str, torch_dtype="auto"):
    """device_map="auto" is the right default -- it uses available GPU
    memory properly on a machine that actually has room (e.g. Colab). But
    confirmed directly on this project's dev machine that "auto" can
    misjudge capacity and try to offload the WHOLE model to disk, which
    from_pretrained doesn't support directly and raises a ValueError
    instead of just working (or failing with a clearer OOM). Falling back
    to CPU-only in that specific case -- slower, but actually runs, rather
    than erroring out on a machine "auto" judged (rightly or wrongly) as too
    tight."""
    try:
        return model_cls.from_pretrained(model_id, torch_dtype=torch_dtype, device_map="auto")
    except ValueError as e:
        if "offload" not in str(e).lower():
            raise
        return model_cls.from_pretrained(model_id, torch_dtype=torch_dtype, device_map="cpu")


def load(model_name: str = GENERATOR_MODEL_NAME):
    if model_name == JUDGE_MODEL_NAME:
        return _load_judge()
    return _load_generator()


def unload(model_name: str | None = None):
    import gc

    import torch

    if model_name == JUDGE_MODEL_NAME:
        _state.pop("judge", None)
    elif model_name == GENERATOR_MODEL_NAME:
        _state.pop("generator", None)
    else:
        _state.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def generate(
    messages: list[dict],
    model_name: str = GENERATOR_MODEL_NAME,
    max_new_tokens: int = 400,
    do_sample: bool = False,
    format: object = None,
) -> str:
    import gc
    # `format` (Ollama's JSON-schema output constraint, see
    # generation/ollama_llm.py) has no direct transformers equivalent wired
    # in here -- accepted for interface compatibility with callers (e.g.
    # evaluation/judge.py) but not enforced. See this module's docstring for
    # what judge reliability actually rests on instead.
    del format

    if model_name == JUDGE_MODEL_NAME:
        tokenizer, model = _load_judge()
        # tokenize=False + a separate tokenizer() call, and the
        # do_sample=True/temperature/top_p/top_k below -- this exact
        # pattern (not the tokenize=True/return_dict=True combined call
        # used for the generator) is what the user independently verified
        # actually suppresses Qwen3.5's reasoning-as-content on a Colab GPU
        # (dev_logs.md Entry 19). The sampling values are fixed to that
        # verified config, not this function's do_sample=False project-wide
        # default, since that default was never what was tested here.
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=True, temperature=0.7, top_p=0.8, top_k=20,
        )
        result = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        # Explicitly release input/output tensors so their GPU memory is
        # returned to the allocator immediately, before the next question's
        # generate() call. Without this, fragmentation accumulates across
        # the 9 answerable questions and exhausts the available headroom
        # (confirmed OOM at Q7 on a 14.56GB card, 42MB free at failure point).
        del inputs, outputs
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        return result

    processor, model = _load_generator()
    inputs = processor.apply_chat_template(
        messages, tokenize=True, return_dict=True, return_tensors="pt",
        add_generation_prompt=True, enable_thinking=False,
    ).to(model.device)
    input_len = inputs["input_ids"].shape[-1]
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample)
    response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
    parsed = processor.parse_response(response, prefix=inputs["input_ids"])
    result = str(parsed.get("content", "")).strip() if isinstance(parsed, dict) else str(parsed).strip()
    # Same tensor cleanup as the judge path above -- prevents fragmented
    # allocations from accumulating across the Phase B question loop.
    del inputs, outputs
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    return result
