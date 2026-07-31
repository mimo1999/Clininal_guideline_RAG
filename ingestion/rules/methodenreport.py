"""Regex fallback for AWMF 'Methodenreport'/'Leitlinienreport' cover pages.

These often don't carry the "Sx-Leitlinie <title>" header the main guideline
document uses -- title is more often just "Leitlinienreport" -- so this module
falls back to the guideline's own title only when the shared pattern misses.
"""

import re

from . import common

_REPORT_TITLE_RE = re.compile(r"^(Leitlinienreport.*)$", re.MULTILINE)


def try_extract(text: str) -> dict:
    result = common.try_extract(text)
    if "title" not in result:
        m = _REPORT_TITLE_RE.search(text)
        if m:
            result["title"] = m.group(1).strip()
    return result
