"""Safety feature: things that would embarrass (or end the career of) a real
human poster.

Different shape from the other three features:
  - Returns a `multiplier`, not a weighted total.
  - The composed reward in score.py is:
        (W_HOOK * hook + W_STRUCTURE * structure + W_STYLE * style)
            * safety["multiplier"]
  - A clean post returns multiplier=1.0 (no effect).
  - Penalty curve is a CLIFF, not a gradient:
        0 hits → 1.0
        1 hit  → 0.3
        2+     → 0.2
    The cliff IS the semantic — safety failures are categorical, not
    gradient. Soften the cliff and you teach the model that violations
    are expensive-but-survivable, which is the wrong lesson.

Sub-signals (each returns an int hit count, not a [0, 1] score):
  - banned_phrases:   slurs, harassment, threats, explicit sexual content.
                      Loaded from data/safety_banned.yaml (real, gitignored)
                      or data/safety_banned.example.yaml (public fallback).
  - naming_and_shaming: proper-name + negative-attribution-verb within a
                      ~150-char proximity window. Positive name mentions
                      do NOT fire (no negative verb nearby).
  - private_info:     literal email addresses, phone numbers, street
                      addresses, doxxing-marker phrases.

Input validation: this module does NOT short-circuit on empty input.
Empty post returns multiplier=1.0 (absence of content is not unsafe content).
The composition layer in score.py is responsible for handling empty input
explicitly via its own short-circuit.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import yaml

from reward.features._text_utils import normalize_text


_LOG = logging.getLogger(__name__)


# --- Banned-phrase list loading ------------------------------------------

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_BANNED_REAL = _DATA_DIR / "safety_banned.yaml"
_BANNED_EXAMPLE = _DATA_DIR / "safety_banned.example.yaml"


def _load_banned_patterns() -> list[re.Pattern[str]]:
    """Prefer the real (gitignored) list; fall back to .example with warning."""
    source: Path
    if _BANNED_REAL.exists():
        source = _BANNED_REAL
    elif _BANNED_EXAMPLE.exists():
        _LOG.warning(
            "Using example banned-phrase list at %s — production reward "
            "function should use full list at %s",
            _BANNED_EXAMPLE.name,
            _BANNED_REAL.name,
        )
        source = _BANNED_EXAMPLE
    else:
        return []

    try:
        data = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        _LOG.exception("failed to parse %s", source.name)
        return []

    patterns: list[re.Pattern[str]] = []
    for category, items in data.items():
        if not isinstance(items, list):
            continue
        for raw in items:
            if not isinstance(raw, str) or not raw.strip():
                continue
            try:
                # Word-boundary anchors so "retarded" matches but "retardant"
                # would also match — accept that for v1; tighten the YAML
                # patterns directly if needed.
                patterns.append(re.compile(rf"\b{raw}\b", re.IGNORECASE))
            except re.error:
                _LOG.warning("skipping malformed pattern in %s: %r", category, raw)
                continue
    return patterns


_BANNED_PATTERNS = _load_banned_patterns()


# --- Naming-and-shaming detector -----------------------------------------
#
# Two-part check: a plausible person-name token within PROXIMITY_CHARS of a
# negative-attribution verb. Both required for a hit. Positive mentions
# (e.g. "Sarah at Anthropic taught me X") have no negative verb nearby
# and are correctly NOT flagged.

# TODO: revalidate PROXIMITY_CHARS against bad_posts.yaml during the
# human-correlation calibration step.
PROXIMITY_CHARS = 150

# A person-name token: capitalized first word optionally followed by a
# capitalized last word, NOT at the start of a sentence. We also accept the
# "[Name] at [Company]" frame since that's the dominant naming-and-shaming
# shape on LinkedIn.
_NAME_RE = re.compile(
    r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)"
    r"(?:\s+(?:at|from|of)\s+[A-Z][\w&]+)?\b"
)

_NEGATIVE_VERBS = (
    r"ghosted",
    r"stole(?:\s+(?:my|our))?",
    r"lied(?:\s+to\s+(?:me|us))?",
    r"scammed",
    r"ripped\s+(?:me|us)\s+off",
    r"betrayed",
    r"backstabbed",
    r"is\s+a\s+(?:fraud|liar|scammer|crook|fake|narcissist|thief)",
    r"is\s+(?:incompetent|toxic|abusive)",
    r"fired\s+me\s+(?:unfairly|without\s+cause|over\s+nothing)",
    r"harassed",
    r"sexually\s+harassed",
    r"discriminated\s+against\s+me",
    r"plagiarized",
    r"took\s+credit\s+for\s+my",
)
_NEGATIVE_VERB_RE = re.compile(
    r"\b(?:" + "|".join(_NEGATIVE_VERBS) + r")\b",
    re.IGNORECASE,
)

# First-word-of-sentence filter: don't treat "After seven years..." as a
# proper name. Lightweight heuristic — match capitalized tokens that aren't
# preceded by sentence-terminating punctuation.
_SENTENCE_STARTERS = {
    "After", "Before", "When", "While", "Then", "First", "Next", "Finally",
    "But", "And", "Or", "So", "Because", "If", "Although", "Though", "However",
    "Today", "Yesterday", "Tomorrow", "Last", "This", "That", "These", "Those",
    "I", "We", "You", "He", "She", "They", "It",
    "My", "Our", "Your", "His", "Her", "Their",
    "The", "A", "An",
}


def _find_naming_shaming(normalized: str) -> list[tuple[str, str]]:
    """Return list of (name, verb) pairs that fire the proximity check."""
    pairs: list[tuple[str, str]] = []
    seen_keys: set[tuple[int, int]] = set()
    for verb_match in _NEGATIVE_VERB_RE.finditer(normalized):
        v_start = verb_match.start()
        v_end = verb_match.end()
        window_start = max(0, v_start - PROXIMITY_CHARS)
        window_end = min(len(normalized), v_end + PROXIMITY_CHARS)
        window = normalized[window_start:window_end]
        for name_match in _NAME_RE.finditer(window):
            name_text = name_match.group(1)
            first_word = name_text.split()[0]
            if first_word in _SENTENCE_STARTERS:
                continue
            # Single capitalized word with no last name and no "at Company"
            # frame is too noisy — skip unless the full match included an
            # affiliation phrase ("at Stripe", "from Google").
            if " " not in name_text:
                full_span = name_match.group(0)
                if not re.search(r"\b(?:at|from|of)\s+[A-Z]", full_span):
                    continue
            key = (window_start + name_match.start(), v_start)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            pairs.append((name_match.group(0).strip(), verb_match.group(0).strip()))
    return pairs


# --- Private-info leakage detectors --------------------------------------

_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PHONE_INTL_RE = re.compile(r"\+\d{1,3}[\s.-]?\d{2,4}[\s.-]?\d{3,4}[\s.-]?\d{3,4}")
_PHONE_US_RE = re.compile(r"\b\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b")
_STREET_ADDR_RE = re.compile(
    r"\b\d{1,5}\s+[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?\s+"
    r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Way|Court|Ct)\b"
)
_DOXXING_RE = re.compile(
    r"\b(?:my|their|his|her)\s+"
    r"(?:home\s+address|phone\s+number|email\s+address|apartment(?:\s+number)?|SSN|social\s+security)"
    r"\s+is\b",
    re.IGNORECASE,
)


def _find_private_info(text: str) -> list[tuple[str, str]]:
    """Return (kind, match) pairs. Use ORIGINAL text — phone formats and
    addresses can include unicode dashes that normalize_text would change."""
    hits: list[tuple[str, str]] = []
    for kind, pattern in (
        ("email", _EMAIL_RE),
        ("phone_intl", _PHONE_INTL_RE),
        ("phone_us", _PHONE_US_RE),
        ("street_address", _STREET_ADDR_RE),
        ("doxxing_marker", _DOXXING_RE),
    ):
        for m in pattern.finditer(text):
            hits.append((kind, m.group(0)))
    return hits


# --- Banned-phrase detector ----------------------------------------------

def _find_banned(normalized: str) -> list[str]:
    """Return list of matched banned-phrase strings."""
    hits: list[str] = []
    for p in _BANNED_PATTERNS:
        for m in p.finditer(normalized):
            hits.append(m.group(0))
    return hits


# --- Penalty curve -------------------------------------------------------

def _multiplier_from_hits(total_hits: int) -> float:
    if total_hits == 0:
        return 1.0
    if total_hits == 1:
        return 0.3
    return 0.2


# --- Public API ----------------------------------------------------------

def score_safety(text: str) -> dict[str, Any]:
    """Score a post on safety.

    Returns a dict with raw hit counts, a debug section logging WHICH patterns
    fired (essential for reward-hack postmortems), and a `multiplier` in
    [0.2, 1.0] that gets multiplied against the additive sum of the other
    three features in score.py.

    NOTE: this module is NOT responsible for input validation. An empty post
    returns multiplier=1.0 because absence of content is not unsafe content.
    The composition layer (score.py) handles the empty-input case via its
    own short-circuit before reaching here.
    """
    if not text:
        return {
            "banned_phrases": 0,
            "naming_and_shaming": 0,
            "private_info": 0,
            "total_hits": 0,
            "multiplier": 1.0,
            "debug": {"banned": [], "name_verb_pairs": [], "private": []},
        }

    normalized = normalize_text(text)

    banned_hits = _find_banned(normalized)
    name_verb_pairs = _find_naming_shaming(normalized)
    private_hits = _find_private_info(text)

    banned_n = len(banned_hits)
    naming_n = len(name_verb_pairs)
    private_n = len(private_hits)
    total_hits = banned_n + naming_n + private_n

    return {
        "banned_phrases": banned_n,
        "naming_and_shaming": naming_n,
        "private_info": private_n,
        "total_hits": total_hits,
        "multiplier": _multiplier_from_hits(total_hits),
        "debug": {
            "banned": banned_hits,
            "name_verb_pairs": name_verb_pairs,
            "private": private_hits,
        },
    }
