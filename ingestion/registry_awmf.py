"""Builds and caches a {register_number -> listing metadata} map for a given
AWMF Fachgesellschaft (medical society), by crawling the society's guideline
listing page once. This avoids hand-maintained register-number/URL mappings
and scales to however many guidelines a society has (DGGG currently lists ~46).

The listing page (https://register.awmf.org/de/leitlinien/aktuelle-leitlinien)
groups guidelines under each society; each guideline row already carries
register number, title, class (S1/S2e/S2k/S3), Stand (as-of date) and
Gueltig-bis (valid-until date) -- so this crawl doubles as a first pass of
metadata, not just a URL index.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from playwright.sync_api import sync_playwright

AWMF_BASE = "https://register.awmf.org"
LISTING_URL = f"{AWMF_BASE}/de/leitlinien/aktuelle-leitlinien"

CACHE_DIR = Path(__file__).parent.parent / "data_corpus" / "cache" / "awmf"
REGISTRY_CACHE_TTL_SECONDS = 30 * 24 * 3600  # 30 days -- society guideline lists change slowly


@dataclass
class RegistryEntry:
    register_number: str
    title: str
    klasse: Optional[str]
    stand: Optional[str]
    gueltig_bis: Optional[str]
    detail_url: str


def _registry_cache_path(society_slug: str) -> Path:
    return CACHE_DIR / f"registry_{society_slug}.json"


def _slugify(society_name: str) -> str:
    return "".join(c.lower() if c.isalnum() else "_" for c in society_name).strip("_")


def build_registry(society_name_substring: str, force_refresh: bool = False) -> dict[str, RegistryEntry]:
    """Crawl the AWMF listing page for the society matching `society_name_substring`
    (e.g. 'Gynäkologie und Geburtshilfe') and return {register_number: RegistryEntry}.

    Cached to disk; re-crawls only when the cache is missing, stale, or force_refresh=True.
    """
    slug = _slugify(society_name_substring)
    cache_path = _registry_cache_path(slug)

    if not force_refresh and cache_path.exists():
        age = time.time() - cache_path.stat().st_mtime
        if age < REGISTRY_CACHE_TTL_SECONDS:
            raw = json.loads(cache_path.read_text(encoding="utf-8"))
            return {k: RegistryEntry(**v) for k, v in raw.items()}

    entries = _crawl_society_listing(society_name_substring)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps({k: asdict(v) for k, v in entries.items()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return entries


def _crawl_society_listing(society_name_substring: str) -> dict[str, RegistryEntry]:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            page.goto(LISTING_URL, wait_until="networkidle", timeout=60000)
            page.wait_for_selector(f"text={society_name_substring}", timeout=20000, state="attached")

            society_href = page.eval_on_selector_all(
                "a",
                """(els, needle) => {
                    const hit = els.find(e => e.getAttribute('href')
                        && e.getAttribute('href').includes('fachgesellschaft')
                        && e.innerText && e.innerText.includes(needle));
                    return hit ? hit.getAttribute('href') : null;
                }""",
                society_name_substring,
            )
            if not society_href:
                raise RuntimeError(f"Could not find a Fachgesellschaft matching {society_name_substring!r}")

            page.goto(f"{AWMF_BASE}{society_href}", wait_until="networkidle", timeout=60000)
            page.wait_for_selector("ion-row.guideline-listing-row", timeout=20000, state="attached")
            page.wait_for_timeout(300)

            rows = page.eval_on_selector_all(
                "ion-row.guideline-listing-row",
                """els => els.map(e => ({
                    text: e.innerText,
                    href: e.querySelector('a') ? e.querySelector('a').getAttribute('href') : null
                }))""",
            )
        finally:
            browser.close()

    entries: dict[str, RegistryEntry] = {}
    for row in rows:
        href = row["href"]
        parts = [p for p in (row["text"] or "").split("\n") if p != ""]
        if not href or len(parts) < 4:
            continue
        register_number, title, klasse, stand = parts[0], parts[1], parts[2], parts[3]
        gueltig_bis = parts[4] if len(parts) > 4 else None
        detail_url = href if href.startswith("http") else f"{AWMF_BASE}{href}"
        entries[register_number] = RegistryEntry(
            register_number=register_number,
            title=title,
            klasse=klasse,
            stand=stand,
            gueltig_bis=gueltig_bis,
            detail_url=detail_url,
        )
    return entries


def lookup(register_number: str, society_name_substring: str) -> Optional[RegistryEntry]:
    registry = build_registry(society_name_substring)
    return registry.get(register_number)
