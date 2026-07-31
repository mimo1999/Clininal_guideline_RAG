"""Shared regex patterns for AWMF cover pages.

Observed AWMF cover-page template (confirmed against the mandatory gold PDF):

    S3-Leitlinie Prävention des Zervixkarzinoms
    Langversion 1.1 -März 2020 AWMF-Registernummer 015/027OL
    Leitlinie (Langversion)

i.e. title line, then "<Fassung> <version> -<German month> <year> AWMF-Registernummer <num>".
This module extracts what it can from that shared shape; per-doc-type modules
override/extend individual fields where a layout diverges.
"""

from __future__ import annotations

import re
from typing import Optional

_GERMAN_MONTHS = {
    "januar": 1, "februar": 2, "märz": 3, "maerz": 3, "april": 4, "mai": 5, "juni": 6,
    "juli": 7, "august": 8, "september": 9, "oktober": 10, "november": 11, "dezember": 12,
}

_TITLE_RE = re.compile(r"^(S[123][ek]?-Leitlinie\s+.+)$", re.MULTILINE)
_VERSION_DATE_RE = re.compile(
    r"(?:Lang|Kurz|Patienten)?version\s+([0-9]+(?:\.[0-9]+)?)\s*-\s*([A-Za-zäöüÄÖÜ]+)\s+(\d{4})",
    re.IGNORECASE,
)
_REGNR_RE = re.compile(r"AWMF-Registernummer\s+([0-9]{3}\s*[/-]\s*[0-9]{3}[A-Za-z]*)")


def try_extract(text: str) -> dict:
    """Best-effort extraction of title/version/date/register-number from cover-page text.
    Returns only keys it found -- callers merge this over/under other sources."""
    result: dict = {}

    title_match = _TITLE_RE.search(text)
    if title_match:
        raw = title_match.group(1).strip()
        result["title"] = re.sub(r"^S[123][ek]?-Leitlinie\s+", "", raw).strip()

    vd_match = _VERSION_DATE_RE.search(text)
    if vd_match:
        version, month_name, year = vd_match.groups()
        result["version"] = version
        month = _GERMAN_MONTHS.get(month_name.strip().lower())
        if month:
            # No day printed on the cover -- record year-month only (day=1 as a placeholder
            # is misleading, so we keep this as a separate "cover_date" hint, not a full date).
            result["cover_year_month"] = f"{year}-{month:02d}"

    regnr_match = _REGNR_RE.search(text)
    if regnr_match:
        result["awmf_register_number"] = re.sub(r"\s*[/-]\s*", "-", regnr_match.group(1))

    return result
