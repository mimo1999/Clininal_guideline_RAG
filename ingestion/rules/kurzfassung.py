"""Regex fallback for AWMF 'Kurzfassung' (short-version) cover pages."""

from . import common


def try_extract(text: str) -> dict:
    return common.try_extract(text)
