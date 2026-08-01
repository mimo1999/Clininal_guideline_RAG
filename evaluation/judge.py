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

JUDGE_SYSTEM_PROMPT = """Du bist ein strenger Gutachter fuer medizinische Leitlinien-Antworten. Du bekommst eine Frage, eine Referenzantwort aus der Leitlinie, den tatsaechlich abgerufenen Quelltext (die Chunks, auf deren Basis die Antwort generiert wurde), und eine vom System generierte Antwort. Beurteile:

Pruefe in dieser Reihenfolge:
1. Welche ZENTRALE(N) SACHAUSSAGE(N) bzw. EMPFEHLUNG(EN) enthaelt die Referenzantwort?
2. Enthaelt die generierte Antwort dieselbe medizinische Kernaussage, auch wenn sie anders formuliert, ausfuehrlicher oder kuerzer ist?
3. Enthalten zusaetzliche Aussagen der generierten Antwort medizinisch relevante neue Behauptungen?
4. Sind diese Aussagen durch den Quelltext UND/ODER die Referenzantwort gedeckt?
Erst danach entscheide ueber "correct" und "grounded".

1. correct: Enthaelt die generierte Antwort die ZENTRALE(N) SACHAUSSAGE(N)/EMPFEHLUNG(EN) der Referenzantwort -- nicht nur ein thematisch verwandtes, plausibel klingendes Statement? Die Kernaussage muss INHALTLICH (semantisch) enthalten sein; Wortlaut, Satzstruktur, Reihenfolge oder Detaillierungsgrad muessen nicht uebereinstimmen. Eine Antwort ist correct, wenn ein medizinischer Fachexperte sie als semantisch gleichwertige Wiedergabe der Referenzantwort bewerten wuerde.

Eine Antwort, die eine andere (auch fachlich richtige) Empfehlung zu einem angrenzenden Aspekt des Themas gibt, aber die in der Referenzantwort genannte SPEZIFISCHE Kernaussage nicht enthaelt, ist NICHT correct -- selbst wenn sie der Referenzantwort nicht widerspricht. "Im Einklang mit der Referenzantwort" heisst: die Kernaussage ist tatsaechlich enthalten (ggf. anders formuliert oder ausfuehrlicher), nicht nur "widerspricht ihr nicht".

Nicht erforderlich ist, dass alle Nebeninformationen oder jedes Detail der Referenzantwort genannt werden. Enthält die generierte Antwort die zentrale Empfehlung korrekt, fuehren fehlende ergaenzende Informationen allein NICHT zu correct=false.

Eine Antwort ist nur dann correct=false, wenn mindestens eine der folgenden Bedingungen erfuellt ist:
- die zentrale Empfehlung fehlt,
- die zentrale Empfehlung wird widersprochen,
- oder eine andere Empfehlung wird als eigentliche Antwort praesentiert.

2. grounded: Wird jede medizinisch relevante sachliche Behauptung der generierten Antwort durch den bereitgestellten Quelltext UND/ODER die Referenzantwort gedeckt? Eine Behauptung, die weder im Quelltext noch in der Referenzantwort auffindbar ist, oder die einer von beiden widerspricht, zaehlt als NICHT gegruendet -- auch wenn sie plausibel klingt oder allgemein medizinisch korrekt ist.

Sprachliche Umformulierungen, Zusammenfassungen, offensichtliche Schlussfolgerungen aus dem Quelltext sowie unterschiedlich formulierte, aber semantisch identische Aussagen gelten NICHT als zusaetzliche Behauptungen. Nur medizinisch relevante neue Fakten oder Empfehlungen muessen explizit gedeckt sein.

WICHTIG zu Quellenangaben: Die generierte Antwort enthaelt fast immer Klammerzusaetze wie "(Quelle 1)", "(Quelle 2, Quelle 5)" oder "(Source 1)". Das sind reine System-Zitiermarkierungen, die automatisch angehaengt werden, um zu zeigen welcher Retrieval-Treffer verwendet wurde -- sie sind KEINE inhaltliche Behauptung. Ignoriere diese Klammerzusaetze bei der grounded-Bewertung vollstaendig, auch wenn die Referenzantwort selbst keine Quellenangabe enthaelt. Eine Quellenangabe darf NIEMALS als Grund fuer grounded=false gewertet werden.

Beispiel 1: Referenzantwort "X betraegt 5mg." / Generierte Antwort "X betraegt 5mg (Quelle 2)." -> grounded: true (die Quellenangabe ist keine Behauptung, also kein Widerspruch).

Beispiel 2 (WICHTIG -- Abgrenzung correct vs. "widerspricht nicht"): Frage "Was ist das empfohlene Vorgehen bei Befund Y?" / Referenzantwort "Wiederholung von Test Z nach 12 Monaten statt sofortiger Behandlung W." / Generierte Antwort "Bei positivem Test soll ohne Verzoegerung eine andere Untersuchung erfolgen; eine direkte Ueberweisung zu W wird nicht empfohlen." -> correct: FALSE. Die generierte Antwort widerspricht der Referenz zwar nicht direkt, nennt aber die konkrete Empfehlung (Wiederholung von Z nach 12 Monaten) an keiner Stelle -- sie beantwortet eine verwandte, aber andere Frage. Nur "nicht widersprechen" reicht fuer correct NICHT aus.

Nur echte inhaltliche Widersprueche oder erfundene Fakten (z.B. falsche Zahlen, falsche Empfehlungen, zusaetzliche medizinische Behauptungen, die weder im Quelltext noch in der Leitlinie stehen) zaehlen als nicht gegruendet. Zusaetzliche, plausible Detailinformationen, die sich im bereitgestellten Quelltext wiederfinden und die Referenzantwort nicht widerlegen, zaehlen NICHT als ungegruendet.

Im Zweifelsfall entscheide zugunsten von correct=true und grounded=true, sofern die generierte Antwort dieselbe klinische Empfehlung vermittelt und keine zusaetzlichen widerspruechlichen oder ungedeckten medizinischen Aussagen enthaelt.

Antworte NUR mit einem JSON-Objekt, ohne weitere Erklaerung, in genau diesem Format:
{"correct": true, "grounded": true, "begruendung": "kurze Begruendung"}"""

@dataclass
class JudgeVerdict:
    correct: bool
    grounded: bool
    reasoning: str
    raw: str


def judge_answer(question: str, reference_answer: str, generated_answer: str, retrieved_context: str = "") -> JudgeVerdict:
    """retrieved_context: the actual retrieved chunk text the generator was
    given (joined), so `grounded` can be checked against real evidence
    instead of only against the fixed reference_answer string -- without
    this, the judge has no way to tell "plausible but off-topic" from
    "actually supported by what was retrieved" (confirmed a real gap this
    way: a generated answer citing an unrelated section's general principle
    was rated grounded=true purely because it didn't contradict the
    reference, never having been checked against the source text it claimed
    to draw from). Optional/defaults to "" for callers that don't have it
    handy -- degrades to the reference-only check, not a hard requirement."""
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
