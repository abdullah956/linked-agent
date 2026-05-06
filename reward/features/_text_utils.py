"""Shared text-processing utilities for reward features.

HARD RULE: every pattern-matching feature must call `normalize_text` on its
input before matching. Curly quotes/apostrophes/dashes appear in real posts
(autocorrected by iOS, copied from Word/Notion) and would silently bypass
ASCII-only regex lists otherwise.

CAVEAT: the em-dash *density* signal in `style.ai_tells` must count em-dashes
from the ORIGINAL un-normalized text. Em-dash spam is itself a real
AI-generated tell — we shouldn't normalize away the very thing we're
trying to measure.
"""

from __future__ import annotations


_TYPOGRAPHIC_MAP = {
    "’": "'",  # right single quote / curly apostrophe
    "‘": "'",  # left single quote
    "“": '"',  # left double quote
    "”": '"',  # right double quote
    "—": "--", # em-dash
    "–": "-",  # en-dash
}


def normalize_text(s: str) -> str:
    """Map typographic variants to ASCII for pattern-matching purposes."""
    if not s:
        return s
    for src, dst in _TYPOGRAPHIC_MAP.items():
        s = s.replace(src, dst)
    return s
