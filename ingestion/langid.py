"""Offline language identification via lingua-language-detector.

Independent of the AWMF scrape/regex metadata path -- run against the actual
extracted body text, so it can't inherit their failure modes.

Generalized to lingua's full language set rather than a hardcoded list --
the corpus started German+English-only, but guidelines in any language
should be detected without a code change (see dev_logs.md Entry 5: this was
previously hardcoded to exactly [GERMAN, ENGLISH] with a hand-built ISO-code
dict, which broke the moment a third language showed up). `Language.
iso_code_639_1` is used directly instead of a hand-maintained mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from lingua import LanguageDetectorBuilder


@dataclass
class LanguageResult:
    language: str  # ISO 639-1 code, e.g. "de", "en"
    confidence: float


@lru_cache(maxsize=1)
def _detector():
    return LanguageDetectorBuilder.from_all_languages().build()


def detect_language(text: str) -> LanguageResult:
    sample = " ".join(text.split())[:5000]
    if not sample.strip():
        return LanguageResult(language="unknown", confidence=0.0)

    detector = _detector()
    confidence_values = detector.compute_language_confidence_values(sample)
    if not confidence_values:
        return LanguageResult(language="unknown", confidence=0.0)

    top = confidence_values[0]
    iso_code = top.language.iso_code_639_1.name.lower()
    return LanguageResult(language=iso_code, confidence=top.value)
