"""Regex fallback for English-language AWMF documents (filename suffix 'e').

English covers use "S3-Guideline <title>" rather than "S3-Leitlinie <title>",
and English month names rather than German ones -- both patterns are tried
here before falling back to the shared German-oriented patterns (a
document might mix German metadata with an English abstract, etc.).
"""

import re

from . import common

_EN_TITLE_RE = re.compile(r"^(S[123][ek]?[\s-]*Guideline\s+.+)$", re.MULTILINE | re.IGNORECASE)
_EN_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}
_EN_VERSION_DATE_RE = re.compile(
    r"version\s+([0-9]+(?:\.[0-9]+)?)\s*-\s*([A-Za-z]+)\s+(\d{4})", re.IGNORECASE
)


def try_extract(text: str) -> dict:
    result = common.try_extract(text)

    if "title" not in result:
        m = _EN_TITLE_RE.search(text)
        if m:
            raw = m.group(1).strip()
            result["title"] = re.sub(r"^S[123][ek]?[\s-]*Guideline\s+", "", raw, flags=re.IGNORECASE).strip()

    if "cover_year_month" not in result:
        m = _EN_VERSION_DATE_RE.search(text)
        if m:
            version, month_name, year = m.groups()
            month = _EN_MONTHS.get(month_name.strip().lower())
            if month:
                result.setdefault("version", version)
                result["cover_year_month"] = f"{year}-{month:02d}"

    if "language" not in result:
        result["language"] = "en"

    return result
