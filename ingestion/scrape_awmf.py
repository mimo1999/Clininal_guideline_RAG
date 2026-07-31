"""Scrapes a single AWMF register detail page for the fields the listing page
doesn't carry: version, publishing organization, and the free-text change log
(which is where things like a "Langfassung ausgetauscht" file-swap note live --
see 015-027OL, where the nominal Stand date and the date the PDF itself was
last replaced are different).
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from playwright.sync_api import sync_playwright

AWMF_BASE = "https://register.awmf.org"
CACHE_DIR = Path(__file__).parent.parent / "data_corpus" / "cache" / "awmf"
DETAIL_CACHE_TTL_SECONDS = 30 * 24 * 3600


@dataclass
class ScrapedDetail:
    register_number: str
    title: Optional[str]
    version: Optional[str]
    stand_date: Optional[str]  # ISO date string
    valid_until_date: Optional[str]  # ISO date string
    last_change_note: Optional[str]
    publishing_organization: Optional[str]
    source_url: str


_DATE_RE = re.compile(r"(\d{2})\.(\d{2})\.(\d{4})")


def _parse_de_date(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    m = _DATE_RE.search(text)
    if not m:
        return None
    day, month, year = m.groups()
    try:
        return date(int(year), int(month), int(day)).isoformat()
    except ValueError:
        return None


def _cache_path(register_number: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", register_number)
    return CACHE_DIR / f"detail_{safe}.json"


def scrape_detail(register_number: str, detail_url: Optional[str] = None, force_refresh: bool = False) -> ScrapedDetail:
    cache_path = _cache_path(register_number)
    if not force_refresh and cache_path.exists():
        age = time.time() - cache_path.stat().st_mtime
        if age < DETAIL_CACHE_TTL_SECONDS:
            return ScrapedDetail(**json.loads(cache_path.read_text(encoding="utf-8")))

    url = detail_url or f"{AWMF_BASE}/de/leitlinien/detail/{register_number}"
    result = _scrape_live(register_number, url)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _scrape_live(register_number: str, url: str) -> ScrapedDetail:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            page.goto(url, wait_until="networkidle", timeout=60000)
            page.wait_for_selector("text=Version:", timeout=20000, state="attached")
            page.wait_for_timeout(300)
            overview_text = page.inner_text("body")

            title = _extract_title(overview_text)
            version = _extract_labeled(overview_text, "Version:")
            stand_raw = _extract_labeled(overview_text, "Stand:")
            valid_until_raw = _extract_labeled(overview_text, "Gültig bis:")
            last_change_note = _extract_labeled(overview_text, "Aktueller Hinweis:")

            publishing_organization = None
            try:
                page.get_by_text("Herausgeber & Autoren", exact=True).first.click(timeout=5000)
                page.wait_for_timeout(500)
                authors_text = page.inner_text("body")
                publishing_organization = _extract_labeled(
                    authors_text, "Federführende Fachgesellschaft(en):", stop_at="(Visitenkarte)"
                )
            except Exception:
                pass  # tab may already be open, or layout differs -- leave as None, regex fallback may fill it
        finally:
            browser.close()

    return ScrapedDetail(
        register_number=register_number,
        title=title,
        version=version,
        stand_date=_parse_de_date(stand_raw),
        valid_until_date=_parse_de_date(valid_until_raw),
        last_change_note=last_change_note,
        publishing_organization=publishing_organization,
        source_url=url,
    )


def _extract_labeled(text: str, label: str, stop_at: Optional[str] = None) -> Optional[str]:
    idx = text.find(label)
    if idx == -1:
        return None
    rest = text[idx + len(label):]
    lines = [l for l in rest.split("\n") if l.strip() != ""]
    if not lines:
        return None
    if stop_at:
        collected = []
        for line in lines:
            if stop_at in line:
                break
            collected.append(line.strip())
        return " ".join(collected).strip() or None
    return lines[0].strip()


def _extract_title(text: str) -> Optional[str]:
    # The guideline title line follows the pattern "<class>-Leitlinie <title>",
    # e.g. "S3-Leitlinie Prävention des Zervixkarzinoms".
    m = re.search(r"^(S[123][ek]?-Leitlinie\s+.+)$", text, re.MULTILINE)
    if m:
        title = m.group(1).strip()
        title = re.sub(r"^S[123][ek]?-Leitlinie\s+", "", title)
        return title
    return None
