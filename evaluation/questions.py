"""The brief's 12 evaluation questions. Gold 1-3 reference answers/evidence
are given directly in the brief. Q4-9 reference answers/evidence were
self-labeled by reading the actual guideline sections (see dev_logs.md Entry
3) -- section numbers cited below are as found in the gold document
(015-027OL Langfassung), verified by direct reading, not guessed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

QuestionKind = Literal["gold", "self_labeled", "trap"]


@dataclass
class EvalQuestion:
    id: int
    question: str
    kind: QuestionKind
    reference_answer: str | None = None  # None for traps -- correct behavior is refusal
    expected_section_number: str | None = None


QUESTIONS: list[EvalQuestion] = [
    EvalQuestion(
        id=1,
        question="Welche Untersuchung und in welchem Intervall wird für Frauen im Alter von 20 bis 34 Jahren im organisierten Screening empfohlen?",
        kind="gold",
        reference_answer="Zytologie (Pap-Test) auf jährlicher Basis; HPV-Testung ist in dieser Altersgruppe nicht das primäre Screeninginstrument.",
        expected_section_number="8.2.",
    ),
    EvalQuestion(
        id=2,
        question="Welches Testverfahren und welches Intervall gelten für Frauen ab 35 Jahren?",
        kind="gold",
        reference_answer="Ko-Testung (HPV-Test und Zytologie) im Intervall von 3 Jahren.",
        expected_section_number="8.2.",
    ),
    EvalQuestion(
        id=3,
        question="Wie ist das empfohlene Vorgehen bei einer Frau ab 35, die HPV-positiv, aber zytologisch unauffällig ist?",
        kind="gold",
        reference_answer="Wiederholung der Ko-Testung nach ca. 12 Monaten statt sofortiger Kolposkopie; bei erneuter Positivität bei der Kontrolle erfolgt dann eine Kolposkopie.",
        expected_section_number="10.6.",
    ),
    EvalQuestion(
        id=4,
        question="Ab welchem Alter beginnt das organisierte Zervixkarzinom-Screening in Deutschland?",
        kind="self_labeled",
        reference_answer=(
            "Das organisierte Screening kann ab 25 Jahren beginnen (Empfehlung 8.2, Konsensbasierte "
            "Empfehlung). In Deutschland haben Frauen ab 20 Jahren weiterhin einen gesetzlichen Anspruch "
            "auf eine Screeninguntersuchung gemäß Krebsfrüherkennungs-Richtlinie (KFE-RL)."
        ),
        expected_section_number="8.1.",
    ),
    EvalQuestion(
        id=5,
        question="Welche Rolle spielt die HPV-Selbstabnahme (Selbstentnahme) laut Leitlinie, und für welche Gruppe?",
        kind="self_labeled",
        reference_answer=(
            "Der HPV-Selbstabstrich soll den Frauen vorbehalten bleiben, die sich nicht an der regulären "
            "Krebsvorsorgeuntersuchung beteiligen (Non-Responder) -- Empfehlungsgrad A. Er verdoppelt "
            "annähernd die Teilnahmerate bei diesen Frauen, hat aber eine etwas niedrigere Sensitivität "
            "als ein professionell entnommener Abstrich."
        ),
        expected_section_number="13.3.",
    ),
    EvalQuestion(
        id=6,
        question="Welche HPV-Typen sind für die Mehrzahl der Zervixkarzinome verantwortlich?",
        kind="self_labeled",
        reference_answer=(
            "HPV-Typen 16 und 18 sind für 60 bis 70% aller Zervixkarzinome verantwortlich. Die acht "
            "häufigsten HPV-Typen (16, 18, 33, 45, 31, 58, 52, 35) wurden in bis zu 90% der untersuchten "
            "Tumormaterialien nachgewiesen."
        ),
        expected_section_number="4.5.",
    ),
    EvalQuestion(
        id=7,
        question="Was ist der Unterschied zwischen einer zytologiebasierten und einer HPV-basierten Screening-Strategie?",
        kind="self_labeled",
        reference_answer=(
            "Ein organisiertes HPV-basiertes Screening (ab 30 Jahren, Intervall 3-5 Jahre) führt zu einer "
            "niedrigeren Rate an Neuerkrankungen am Zervixkarzinom im Vergleich zu einem rein "
            "zytologiebasierten Screening mit 3-jährlichen Intervallen, da HPV-Tests eine höhere "
            "Sensitivität für die Detektion von CIN 3+ haben. Cytologie-basiertes Screening hat dagegen "
            "eine höhere Spezifität und weniger falsch-positive Ergebnisse, besonders bei jüngeren Frauen "
            "-- deshalb wird HPV-basiertes Screening unter 30 Jahren nicht empfohlen."
        ),
        expected_section_number="8.2.",
    ),
    EvalQuestion(
        id=8,
        question="Welche Empfehlung gibt die Leitlinie zur Kolposkopie bei auffälligem Befund?",
        kind="self_labeled",
        reference_answer=(
            "Die Indikation zur kolposkopischen Abklärung soll ab einer Post-Test-Wahrscheinlichkeit für "
            "ein durchschnittliches kumulatives CIN 3+ Risiko von 10% gestellt werden (Empfehlung 10.2, "
            "Konsensbasierte Empfehlung, EK)."
        ),
        expected_section_number="10.2.",
    ),
    EvalQuestion(
        id=9,
        question="Welche Bedeutung hat die HPV-Impfung im Kontext der Prävention laut Leitlinie?",
        kind="self_labeled",
        reference_answer=(
            "Die HPV-Impfung ist primäre Prävention (im Gegensatz zum Screening als sekundäre Prävention). "
            "Die STIKO empfiehlt die Impfung aller Mädchen im Alter von 9 bis 14 Jahren; sie verhindert die "
            "Infektion mit den Impf-HPV-Typen (u.a. 16, 18) sowie die Entstehung der daraus resultierenden "
            "Krebsvorstufen."
        ),
        expected_section_number="5.1.",
    ),
    EvalQuestion(
        id=10,
        question="Welche Thrombolyse-Therapie und welches Zeitfenster gelten beim akuten ischämischen Schlaganfall?",
        kind="trap",
    ),
    EvalQuestion(
        id=11,
        question="Welches Antibiotikum ist first-line bei einer ambulant erworbenen Pneumonie?",
        kind="trap",
    ),
    EvalQuestion(
        id=12,
        question="Welche medikamentöse Erstlinientherapie wird bei arterieller Hypertonie empfohlen?",
        kind="trap",
    ),
]
