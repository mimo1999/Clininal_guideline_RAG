"""Parses the per-guideline `.txt` sidecar file (e.g. `015-027OL/015-027OL.txt`)
-- a flattened dump of the AWMF register page (same label/value shape as
scrape_awmf.py's live scrape), plus a "Verfügbare Dokumente" block mapping
document-type labels to PDF filenames.

This is now the *primary* metadata source (ahead of the live AWMF scrape) --
see build_document.py. Two real parsing wrinkles confirmed by reading actual
sidecar files: the document-list isn't valid JSON (inconsistent quoting, e.g.
`{Dia-Version" : "..."}`), and filenames sometimes include `.pdf` and
sometimes don't -- both handled here, not assumed away.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# Ordered as they appear in a sidecar file; used to slice the text into
# per-label spans (values can be multi-line, unlike scrape_awmf.py's
# single-line fields, so a simple "next line" grab isn't enough here).
LABELS = [
    "Art der Anmeldung:",
    "Angemeldete Klasse:",
    "Anmeldedatum:",
    "Geplante Fertigstellung:",
    "Gründe für die Themenwahl:",
    "Zielorientierung der Leitlinie:",
    "Verbindung zu themenverwandten Leitlinien:",
    "Anmelder bei der AWMF (Person):",
    "Anmeldende Fachgesellschaft(en):",
    "Version:",
    "Stand:",
    "Gültig bis:",
    "Aktueller Hinweis:",
    "Verfügbare Dokumente:",
]

# Tolerant of the sidecar's malformed pseudo-JSON: optional braces, optional
# leading quote on the label (some entries are missing it, e.g. `{Dia-Version"
# : "..."}`), and either straight or curly quotes as delimiters (one real
# guideline used a curly closing quote before the colon: `guideline
# "Prevention..."` : "..."} -- confirmed via raw byte inspection).
_Q = '"“”'
_DOC_ENTRY_RE = re.compile(rf'\{{?\s*[{_Q}]?([^{{}}{_Q}:]+)[{_Q}]\s*:\s*[{_Q}]([^{_Q}{{}}]+)[{_Q}]')

_DATE_RE = re.compile(r"(\d{2})\.(\d{2})\.(\d{4})")


@dataclass
class SidecarResult:
    title: str | None
    version: str | None
    stand_date: str | None  # ISO
    valid_until_date: str | None  # ISO
    last_change_note: str | None
    publishing_organization: str | None
    purpose_text: str | None
    related_guidelines: list[str] = field(default_factory=list)
    document_manifest: dict[str, str] = field(default_factory=dict)  # label -> filename (with .pdf)
    unmatched_labels: list[str] = field(default_factory=list)  # manifest entries with no file on disk
    unlisted_files: list[str] = field(default_factory=list)  # files on disk not mentioned in the manifest


def _parse_de_date(text: str | None) -> str | None:
    if not text:
        return None
    m = _DATE_RE.search(text)
    if not m:
        return None
    day, month, year = m.groups()
    from datetime import date
    try:
        return date(int(year), int(month), int(day)).isoformat()
    except ValueError:
        return None


def _slice_by_labels(text: str) -> dict[str, str]:
    positions = []
    for label in LABELS:
        idx = text.find(label)
        if idx != -1:
            positions.append((idx, label))
    positions.sort()

    spans: dict[str, str] = {}
    for i, (idx, label) in enumerate(positions):
        start = idx + len(label)
        end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        spans[label] = text[start:end].strip()
    return spans


def _extract_title(text: str) -> str | None:
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    for line in lines:
        if line == "Angemeldet":
            continue
        # strip a trailing parenthetical like "(Konsultationsfassung: https://...)"
        return re.sub(r"\s*\([^)]*\)\s*$", "", line).strip()
    return None


def _parse_document_manifest(block: str) -> dict[str, str]:
    manifest = {}
    # Cut off known trailing boilerplate that isn't part of the document list.
    for marker in ("auch verfügbar in der OL-App", "und im Leitlinien-Hub", "Konsultationsfasssung"):
        idx = block.find(marker)
        if idx != -1:
            block = block[:idx]

    for m in _DOC_ENTRY_RE.finditer(block):
        label, filename = m.group(1).strip(), m.group(2).strip()
        if not filename.lower().endswith(".pdf"):
            filename = f"{filename}.pdf"
        manifest[label] = filename
    return manifest


def parse_sidecar(sidecar_path: str | Path, guideline_dir: str | Path) -> SidecarResult:
    text = Path(sidecar_path).read_text(encoding="utf-8")
    spans = _slice_by_labels(text)

    related = [
        line.strip() for line in spans.get("Verbindung zu themenverwandten Leitlinien:", "").split("\n")
        if line.strip()
    ]
    manifest = _parse_document_manifest(spans.get("Verfügbare Dokumente:", ""))

    actual_files = {p.name for p in Path(guideline_dir).glob("*.pdf")}
    actual_files_lower = {f.lower(): f for f in actual_files}
    unmatched_labels = [label for label, fname in manifest.items() if fname.lower() not in actual_files_lower]
    mentioned_lower = {fname.lower() for fname in manifest.values()}
    unlisted_files = sorted(f for f in actual_files if f.lower() not in mentioned_lower)

    return SidecarResult(
        title=_extract_title(text),
        version=spans.get("Version:") or None,
        stand_date=_parse_de_date(spans.get("Stand:")),
        valid_until_date=_parse_de_date(spans.get("Gültig bis:")),
        last_change_note=spans.get("Aktueller Hinweis:") or None,
        publishing_organization=spans.get("Anmeldende Fachgesellschaft(en):", "").split("\n")[0].strip() or None,
        purpose_text=spans.get("Zielorientierung der Leitlinie:") or None,
        related_guidelines=related,
        document_manifest=manifest,
        unmatched_labels=unmatched_labels,
        unlisted_files=unlisted_files,
    )
