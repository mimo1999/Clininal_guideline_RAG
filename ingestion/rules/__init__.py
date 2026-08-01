"""Dispatches cover-page regex extraction by AWMF document type.

Primary signal is now the guideline's sidecar document-manifest label (e.g.
"Langfassung der Leitlinie ...", "Kurzinformation Gebärmutterhalskrebs...")
-- the sidecar states what a file *is* in its own words, which is more
reliable than guessing from the filename. Filename-suffix guessing stays only
as a fallback for a PDF the sidecar doesn't cover. Confirmed necessary: the
naive "suffix starts with X" approach used before this would have
misclassified 032-033OL's "ki" (Kurzinformation) suffix as "kurzfassung"
(both start with 'k'), and had no case at all for "p" (Patientenversion).

New layouts get a new module here plus a line in DOC_TYPE_MODULES; the
orchestrator never needs to change.
"""

from __future__ import annotations

import re
from typing import Optional

from . import common, english, kurzfassung, langfassung, methodenreport

DOC_TYPE_MODULES = {
    "langfassung": langfassung,
    "kurzfassung": kurzfassung,
    "methodenreport": methodenreport,
    "english": english,
}

# "OL" is optional -- confirmed a real gap: 015-059 and 032-042 (S2k-level
# guidelines, not part of the OL-Programm) have no "OL" in their register
# number at all (filenames like "015-059l_...", not "015-059OLl_..."), so
# the earlier OL-required pattern returned no suffix at all for them. Mirrors
# build_document.py's own _FILENAME_REGNR_RE, which already treats "OL" as
# optional for the register number itself.
_SUFFIX_RE = re.compile(r"\d{3}-\d{3}(?:OL)?([a-zA-Z]+\d*)_", re.IGNORECASE)

# Exact filename-suffix -> doc_type, checked case-insensitively before any
# prefix heuristic (order matters: "ki" must be checked as its own entry, not
# fall through to a "starts with k" match against "kurzfassung").
_KNOWN_SUFFIXES = {
    "l": "langfassung",
    "k": "kurzfassung",
    "ki": "kurzinformation",
    "m": "methodenreport",
    "p": "patientenversion",
    "d": "other",  # Dia-Version -- presentation slides, not prose content
    "eng": "english",
    "e1": "english",
    "e2": "english",
}

# Ordered, most-specific-first: matched as a case-insensitive substring
# against a sidecar document-manifest label. First match wins.
_LABEL_KEYWORDS = [
    ("langfassung", "langfassung"),
    ("kurzinformation", "kurzinformation"),
    ("kurzfassung", "kurzfassung"),
    ("patientenleitlinie", "patientenversion"),
    ("patientenversion", "patientenversion"),
    ("leitlinienreport", "methodenreport"),
    ("methodenreport", "methodenreport"),
    ("evidenzbericht", "methodenreport"),
    ("english", "english"),
    ("evidence", "english"),
    ("dia-version", "other"),
]


def detect_doc_type_from_label(label: str) -> str:
    """Primary path: match a sidecar document-manifest label to a normalized
    doc_type. Falls back to "other" (not "unknown") since a labeled-but-
    unrecognized entry is still a known, real document -- just not one of
    our named categories."""
    lowered = label.lower()
    for keyword, doc_type in _LABEL_KEYWORDS:
        if keyword in lowered:
            return doc_type
    return "other"


def extract_filename_suffix(filename: str) -> Optional[str]:
    """The AWMF short suffix token (l/k/m/p/d/ki/e1/e2/...) between the
    register number and the descriptive filename part -- distinct per PDF
    within a guideline by AWMF's own naming convention, even when two PDFs
    share a normalized doc_type category (e.g. both "methodenreport"). Used
    to disambiguate doc_id on such a collision."""
    m = _SUFFIX_RE.search(filename)
    return m.group(1).lower() if m else None


def detect_doc_type(filename: str) -> str:
    """Fallback path: guess from the filename suffix, for a PDF the sidecar's
    manifest doesn't cover at all (or when there's no sidecar)."""
    suffix = extract_filename_suffix(filename)
    if not suffix:
        return "unknown"
    if suffix in _KNOWN_SUFFIXES:
        return _KNOWN_SUFFIXES[suffix]
    # last-resort prefix heuristic for a genuinely novel suffix
    if suffix.startswith("l"):
        return "langfassung"
    if suffix.startswith("m"):
        return "methodenreport"
    if suffix.startswith("e"):
        return "english"
    # AWMF uses "p1"/"p2"/... for a multi-part Patientenleitlinie (not just
    # bare "p", already in _KNOWN_SUFFIXES) -- confirmed a real, current gap:
    # 032-055OL's own sidecar has a malformed/truncated filename for its
    # Patientenleitlinie entry (a line-wrap artifact from the AWMF page
    # scrape), so the sidecar-label match fails for that PDF and this
    # filename-suffix fallback is what actually has to resolve it correctly.
    if suffix.startswith("p"):
        return "patientenversion"
    return "unknown"


def try_extract(filename: str, first_page_text: str, doc_type: Optional[str] = None) -> dict:
    """doc_type, if provided (e.g. resolved from the sidecar manifest by the
    caller), takes precedence over filename-suffix guessing."""
    resolved_type = doc_type or detect_doc_type(filename)
    module = DOC_TYPE_MODULES.get(resolved_type, common)
    result = module.try_extract(first_page_text)
    result["doc_type"] = resolved_type
    return result
