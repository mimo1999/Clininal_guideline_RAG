"""Ollama-backed generation/judge backend, via the official `ollama` Python
package (not raw requests -- see dev_logs.md Entry 17). Serves both the
cloud diagnostic path (gemma4:31b-cloud, nemotron-3-nano:30b-cloud --
requires an Ollama account, see dev_logs.md Entry 4) and the local-only path
(generation/llm.py's local model tags -- no account needed), since both are
just different model tags through the same Ollama server API; this module
doesn't care which kind of tag it's given.

Using the official client instead of hand-rolled requests calls gets two
real capabilities for free:
- `think=False` -- suppresses the <think>...</think> reasoning BLOCK, but
  NOT a reasoning-capable model's tendency to still write verbose
  step-by-step prose as regular content (confirmed directly: qwen3:4b with
  think=False produced no <think> tag, but still burned an entire
  2000-token budget on English narrative reasoning instead of the requested
  German JSON verdict -- 64s, no valid output). think=False alone does not
  solve the token-budget problem Entry 4/13 first ran into with reasoning
  models; see `format` below for what actually does.
- `format` -- a JSON-schema (or bare "json") chat() parameter that
  constrains the model's output server-side, forcing it straight to valid
  JSON instead of hoping free-form prose happens to end with parseable
  JSON. Confirmed directly: the same judge call above, with a schema passed
  via `format`, dropped from 64s/unparseable to 7.9s/valid JSON. This is
  the actual fix for judge reliability, not just a nice-to-have -- see
  evaluation/judge.py's use of it.
"""

from __future__ import annotations

from dataclasses import dataclass

import ollama

# 127.0.0.1, not "localhost" -- confirmed two separate ollama.exe processes
# can end up listening on port 11434 simultaneously on this machine (one
# bound to 0.0.0.0/[::] -- the tray "Ollama app" -- one to 127.0.0.1
# specifically), with diverged in-memory model lists even if they share the
# same on-disk model store. "localhost" resolved to the IPv6 loopback first
# here, silently hitting the stale process and 404ing on a just-pulled model
# (gemma3:270m) that WAS present on the 127.0.0.1-bound one. Pinning the IP
# sidesteps the resolution ambiguity entirely.
OLLAMA_HOST = "http://127.0.0.1:11434"

_client = ollama.Client(host=OLLAMA_HOST)


@dataclass
class OllamaResponse:
    content: str
    input_tokens: int | None
    output_tokens: int | None


def generate(
    messages: list[dict], model: str, max_new_tokens: int = 400, think: bool = False, format: object = None,
) -> str:
    """Back-compat wrapper -- returns just the text, same as before. Use
    generate_with_usage() where the token counts are needed (e.g. for
    Langfuse tracing -- see run_eval_cloud_diagnostic.py)."""
    return generate_with_usage(messages, model, max_new_tokens, think=think, format=format).content


def generate_with_usage(
    messages: list[dict], model: str, max_new_tokens: int = 400, think: bool = False, format: object = None,
) -> OllamaResponse:
    resp = _client.chat(
        model=model, messages=messages, think=think, format=format, options={"num_predict": max_new_tokens},
    )
    return OllamaResponse(
        content=resp.message.content.strip(),
        input_tokens=resp.get("prompt_eval_count"),
        output_tokens=resp.get("eval_count"),
    )


def is_available(model: str, timeout: float = 3.0) -> bool:
    """Whether Ollama is reachable at all AND the given model is already
    pulled. Used by webapp/main.py and setup.py to decide whether the
    cloud/local generation path is actually usable before relying on it.

    Client.list() itself takes no timeout argument -- a short-timeout client
    is constructed fresh per call instead (cheap; this is a lightweight
    health check, not the connection used for actual generation, which
    needs a much longer timeout and lives on the module-level `_client`)."""
    try:
        tags = [m.model for m in ollama.Client(host=OLLAMA_HOST, timeout=timeout).list().models]
        return model in tags
    except Exception:
        return False


def ensure_pulled(model: str) -> bool:
    """Pulls a model if it isn't already present. Returns True if the model
    is available afterward (already present, or successfully pulled) --
    False if Ollama itself isn't reachable or the pull failed (e.g. no
    network, unknown model tag). Never raises -- callers (setup.py) treat a
    failed pull as a warning, not a hard stop."""
    if is_available(model):
        return True
    try:
        _client.pull(model)
        return is_available(model)
    except Exception:
        return False
