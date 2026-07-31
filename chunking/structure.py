"""Structure extraction: builds a Document -> Chapter -> Section hierarchy from
the cleaned Markdown's numbered heading prefixes, and tags each section with
best-effort recommendation/evidence-grade and topic metadata.

This is explicitly *not* a knowledge graph -- no entities, no edges, no graph
database. It produces plain metadata fields attached to sections (and, via
chunker.py, to chunks): Document -> Chapter -> Section -> Recommendation as
hierarchy metadata, and a small topic vocabulary as tags. See the plan's
reasoning for why a real ontology/KG isn't justified for this question set.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

HEADING_RE = re.compile(r"^## (.+)$", re.MULTILINE)

# AWMF's recommendation-box format, confirmed against the gold document:
# "| Empfehlungsgrad A | <recommendation text> |" (one row per recommendation,
# grade in {A, B, 0}), or a bare "EK" (Expertenkonsens) marker for
# consensus-based recommendations with no formal evidence grading.
RECOMMENDATION_RE = re.compile(r"^\|\s*Empfehlungsgrad\s+(\S+)\s*\|(.+)\|\s*$", re.MULTILINE)

# Small, hand-seeded topic vocabulary for this corpus (cervical cancer
# prevention/screening) -- plain keyword tagging, not NLP entity extraction.
TOPIC_VOCABULARY = {
    "hpv": "HPV",
    "papillomavir": "HPV",
    "screening": "Screening",
    "früherkennung": "Screening",
    "zytologie": "Zytologie",
    "zytologisch": "Zytologie",
    "pap-": "Zytologie",
    "kolposkopie": "Kolposkopie",
    "kolposkopisch": "Kolposkopie",
    "impfung": "Impfung",
    "vakzin": "Impfung",
    "ko-testung": "Ko-Testung",
    "cotest": "Ko-Testung",
    "selbstabnahme": "Selbstabnahme",
    "selbstabstrich": "Selbstabnahme",
    "selbstentnahme": "Selbstabnahme",
    "biomarker": "Biomarker",
    "konisation": "Therapie",
    "exzision": "Therapie",
    "zervixkarzinom": "Zervixkarzinom",
    "cin": "CIN",
}


@dataclass
class Recommendation:
    evidence_grade: str
    text: str
    start: int
    end: int


@dataclass
class Section:
    section_number: str  # e.g. "3.1" ("" for unnumbered/front-matter)
    section_title: str
    heading_path: list[str]
    depth: int
    start: int
    end: int
    body: str = ""
    topics: list[str] = field(default_factory=list)
    recommendations: list[Recommendation] = field(default_factory=list)

    @property
    def section_id(self) -> str:
        return self.section_number or f"unnumbered:{abs(hash(self.section_title)) % 10_000}"


def _heading_number(heading_text: str) -> str:
    m = re.match(r"^([\d.]+)\.?\s", heading_text + " ")
    return m.group(1) if m else ""


def _depth(section_number: str) -> int:
    return section_number.count(".") + 1 if section_number else 0


def _tag_topics(text: str) -> list[str]:
    lowered = text.lower()
    found = []
    for keyword, topic in TOPIC_VOCABULARY.items():
        if keyword in lowered and topic not in found:
            found.append(topic)
    return found


def _find_recommendations(text: str, base_offset: int) -> list[Recommendation]:
    recs = []
    for m in RECOMMENDATION_RE.finditer(text):
        grade, body = m.group(1), m.group(2).strip()
        recs.append(Recommendation(
            evidence_grade=grade,
            text=body,
            start=base_offset + m.start(),
            end=base_offset + m.end(),
        ))
    return recs


def build_sections(markdown: str) -> list[Section]:
    """Flat list of sections (leaf and non-leaf) in document order, each with
    its own body span, ancestor heading_path, topics, and recommendations."""
    matches = list(HEADING_RE.finditer(markdown))
    if not matches:
        return [Section("", "", [], 0, 0, len(markdown), body=markdown,
                         topics=_tag_topics(markdown),
                         recommendations=_find_recommendations(markdown, 0))]

    sections: list[Section] = []
    if matches[0].start() > 0:
        preamble = markdown[: matches[0].start()]
        sections.append(Section("", "Front matter", [], 0, 0, matches[0].start(),
                                 body=preamble, topics=_tag_topics(preamble),
                                 recommendations=_find_recommendations(preamble, 0)))

    ancestor_stack: list[tuple[int, str]] = []  # (depth, title)
    for idx, m in enumerate(matches):
        heading_text = m.group(1).strip()
        number = _heading_number(heading_text)
        depth = _depth(number) or 1
        title = heading_text

        while ancestor_stack and ancestor_stack[-1][0] >= depth:
            ancestor_stack.pop()
        heading_path = [t for _, t in ancestor_stack] + [title]
        ancestor_stack.append((depth, title))

        start = m.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(markdown)
        body = markdown[start:end]

        sections.append(Section(
            section_number=number,
            section_title=title,
            heading_path=heading_path,
            depth=depth,
            start=start,
            end=end,
            body=body,
            topics=_tag_topics(body),
            recommendations=_find_recommendations(body, start),
        ))

    return sections


def leaf_sections(sections: list[Section]) -> list[Section]:
    """Sections that have no deeper child section immediately following them
    -- these are the units chunker.py packs into chunks."""
    leaves = []
    for i, s in enumerate(sections):
        next_depth = sections[i + 1].depth if i + 1 < len(sections) else 0
        if next_depth <= s.depth:
            leaves.append(s)
    return leaves
