"""Local LLM-as-judge for the "Answer accuracy rate" metric: given a question,
its reference answer, and the system's generated answer, judges whether the
generated answer is (a) correct relative to the reference and (b) grounded
(no unsupported additions).

Uses a SEPARATE model (generation/llm.py's JUDGE_MODEL_NAME, qwen3:4b via
Ollama) from the generator (GENERATOR_MODEL_NAME, gemma3:4b) -- previously
shared one model for both roles, which turned out to matter: an independent
cloud judge disagreed with that shared model's self-assessed "correct"
verdict on 6 of 9 golden questions (dev_logs.md Entry 14), including a
direct factual contradiction and an answer that literally just repeated the
question back, both self-rated as correct. A model judging its own
generation is a fundamentally weaker check than a different model doing it,
so this is no longer optional. Judge verdicts are still validated against
Gold 1-3 (known-correct answers) before being trusted on the self-labeled
Q4-9, per the brief's explicit ask to report how the judge was validated.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from generation.llm import JUDGE_MODEL_NAME, generate

# Passed as Ollama's `format` chat() parameter (dev_logs.md Entry 17) --
# constrains output server-side to valid JSON matching this shape, instead
# of hoping free-form prose happens to end with parseable JSON. Confirmed
# directly this is what actually fixes judge reliability for a
# reasoning-capable model: the same judge call, same prompt, dropped from
# 64s/unparseable-output to 7.9s/valid-JSON once this was added.
JUDGE_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "correct": {"type": "boolean"},
        "grounded": {"type": "boolean"},
        "begruendung": {"type": "string"},
    },
    "required": ["correct", "grounded", "begruendung"],
}

JUDGE_SYSTEM_PROMPT = """Du bist ein strenger Gutachter fuer medizinische Leitlinien-Antworten. Du bekommst eine Frage, eine Referenzantwort aus der Leitlinie, und eine vom System generierte Antwort. Beurteile:
1. correct: Beantwortet die generierte Antwort die Kernfrage inhaltlich richtig, im Einklang mit der Referenzantwort (auch wenn anders formuliert oder ausfuehrlicher)?
2. grounded: Enthaelt die generierte Antwort KEINE Behauptungen, die der Referenzantwort widersprechen oder offensichtlich erfunden wirken?

WICHTIG zu Quellenangaben: Die generierte Antwort enthaelt fast immer Klammerzusaetze wie "(Quelle 1)", "(Quelle 2, Quelle 5)" oder "(Source 1)". Das sind reine System-Zitiermarkierungen, die automatisch angehaengt werden, um zu zeigen welcher Retrieval-Treffer verwendet wurde -- sie sind KEINE inhaltliche Behauptung. Ignoriere diese Klammerzusaetze bei der grounded-Bewertung vollstaendig, auch wenn die Referenzantwort selbst keine Quellenangabe enthaelt. Eine Quellenangabe darf NIEMALS als Grund fuer grounded=false gewertet werden.

Beispiel: Referenzantwort "X betraegt 5mg." / Generierte Antwort "X betraegt 5mg (Quelle 2)." -> grounded: true (die Quellenangabe ist keine Behauptung, also kein Widerspruch).

Nur echte inhaltliche Widersprueche oder erfundene Fakten (z.B. falsche Zahlen, falsche Empfehlungen, zusaetzliche medizinische Behauptungen, die nicht in der Leitlinie stehen) zaehlen als nicht gegruendet. Zusaetzliche, plausible Detailinformationen aus der Leitlinie, die die Referenzantwort nicht widerlegen, zaehlen ebenfalls NICHT als ungegruendet.

Antworte NUR mit einem JSON-Objekt, ohne weitere Erklaerung, in genau diesem Format:
{"correct": true, "grounded": true, "begruendung": "kurze Begruendung"}"""


@dataclass
class JudgeVerdict:
    correct: bool
    grounded: bool
    reasoning: str
    raw: str


def judge_answer(question: str, reference_answer: str, generated_answer: str) -> JudgeVerdict:
    user_prompt = f"""Frage: {question}

Referenzantwort: {reference_answer}

Generierte Antwort: {generated_answer}

Bewertung (nur JSON):"""
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    # Bumped from 150 -> 500 -> 2000: the cloud diagnostic (dev_logs.md Entry
    # 4) found a reasoning-model judge (nemotron-3-nano) silently truncating
    # to empty content because its "thinking" pass alone exceeded the token
    # budget. Still a generous margin even with `format` (below) doing most
    # of the real work now.
    raw = generate(messages, model_name=JUDGE_MODEL_NAME, max_new_tokens=2000, format=JUDGE_RESPONSE_SCHEMA)

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return JudgeVerdict(correct=False, grounded=False, reasoning="judge output unparseable", raw=raw)
    try:
        data = json.loads(match.group(0))
        return JudgeVerdict(
            correct=bool(data.get("correct")),
            grounded=bool(data.get("grounded")),
            reasoning=str(data.get("begruendung", "")),
            raw=raw,
        )
    except json.JSONDecodeError:
        return JudgeVerdict(correct=False, grounded=False, reasoning="judge output invalid JSON", raw=raw)
