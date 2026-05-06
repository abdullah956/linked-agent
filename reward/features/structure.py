"""Structure feature: scannability of a LinkedIn post.

Sub-signals (each in [0, 1]):
  - paragraph_rhythm:    paragraphs-per-char density sits in the readable band.
                         The dominant scannability signal.
  - max_paragraph_length: longest paragraph is below the wall-of-text threshold.
                         Catches the "buried 700-char paragraph among short ones"
                         failure that paragraph_rhythm would miss.
  - payoff_shape:        last paragraph is short and standalone — the post lands
                         a closer rather than trailing off mid-thought. Content
                         quality of the closer (cliched "thoughts?" endings) is
                         intentionally NOT scored here — that belongs in style.py.

Plus a multiplicative GATE (not a weighted term):
  - total_length_band:   ramp in [200, 300], full credit in [300, 1400], ramp
                         out in [1400, 2200], zero outside. Multiplied against
                         the weighted sum of the three sub-signals above. The
                         gate is soft on the edges deliberately — hard cliffs
                         are reward-hack bait.

Composition:
    structure = total_length_band * (
        W_RHYTHM * paragraph_rhythm
        + W_MAX   * max_paragraph_length
        + W_PAYOFF* payoff_shape
    )
"""

from __future__ import annotations

import re

# Weights for the additive portion. Sum to 1.0.
W_RHYTHM = 0.45   # dominant scannability signal
W_MAX = 0.25      # wall-of-text guard
W_PAYOFF = 0.30   # finished-post shape

# Soft multiplicative gate on total post length (characters).
LEN_HARD_MIN, LEN_SOFT_MIN = 200, 300
LEN_SOFT_MAX, LEN_HARD_MAX = 1400, 2200

# Paragraph rhythm: comfortable density spans the full LinkedIn range, from
# punchy 1-2 sentence paragraphs (the platform's signature look) up through
# chunkier essay-style writing. Real well-broken posts often sit at 60-100
# chars/para — calibrating the lower bound for essay-style only would punish
# the platform-native style.
#
# Calibration note: CHARS_PER_PARA_LOW lowered from 150 to 70 during initial
# calibration after the well-broken-post test fixture (5 paragraphs, ~80
# chars/para) scored ~0.02 on rhythm. LinkedIn-native rhythm runs shorter
# than essay rhythm. Revisit if good_posts.yaml leans toward longer-form
# writing — at that point the comfortable band may need to widen rather than
# shift.
CHARS_PER_PARA_SOFT_MIN = 30    # below this, chopped into engagement-bait fragments
CHARS_PER_PARA_LOW = 70         # comfortable lower bound (punchy short paragraphs)
CHARS_PER_PARA_HIGH = 300       # comfortable upper bound
CHARS_PER_PARA_SOFT_MAX = 600   # past this, wall of text

# Wall-of-text threshold (per-paragraph). 500 = deliberate long beat is fine,
# 700+ = the reader's eye gives up.
MAX_PARA_SOFT = 500
MAX_PARA_HARD = 700

# Payoff: last paragraph reads as a closer if it's standalone-short.
PAYOFF_LOW = 10        # below this, not a real closer (e.g. just punctuation)
PAYOFF_HIGH = 150      # past this, the "payoff" is a fresh paragraph of body
PAYOFF_HARD = 250      # past this, no payoff credit at all


_PARA_SPLIT_RE = re.compile(r"\n\s*\n+")


def _trapezoid(x: float, hard_lo: float, soft_lo: float, soft_hi: float, hard_hi: float) -> float:
    """Trapezoidal membership: 0 outside [hard_lo, hard_hi], 1 in [soft_lo, soft_hi],
    linear ramp on the edges. Used for length-band style scores."""
    if x <= hard_lo or x >= hard_hi:
        return 0.0
    if soft_lo <= x <= soft_hi:
        return 1.0
    if x < soft_lo:
        return (x - hard_lo) / (soft_lo - hard_lo)
    return (hard_hi - x) / (hard_hi - soft_hi)


def _paragraphs(text: str) -> list[str]:
    """Split on blank lines and strip empties. Single newlines do NOT create
    a new paragraph — only blank-line breaks count, which matches how
    LinkedIn renders the feed."""
    parts = _PARA_SPLIT_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def _length_gate(text: str) -> float:
    """Soft multiplicative gate on total post length."""
    n = len(text)
    return _trapezoid(n, LEN_HARD_MIN, LEN_SOFT_MIN, LEN_SOFT_MAX, LEN_HARD_MAX)


def _paragraph_rhythm(paragraphs: list[str], total_len: int) -> float:
    """Score the chars-per-paragraph density. Undefined on zero paragraphs."""
    if not paragraphs or total_len == 0:
        return 0.0
    if len(paragraphs) == 1:
        # One-paragraph post: rhythm is structurally absent. Don't reward it.
        return 0.0
    chars_per_para = total_len / len(paragraphs)
    return _trapezoid(
        chars_per_para,
        CHARS_PER_PARA_SOFT_MIN,
        CHARS_PER_PARA_LOW,
        CHARS_PER_PARA_HIGH,
        CHARS_PER_PARA_SOFT_MAX,
    )


def _max_paragraph_length(paragraphs: list[str]) -> float:
    """1.0 if no paragraph exceeds the wall threshold, ramps to 0 past the
    hard cap. Only meaningful in multi-paragraph posts: a single paragraph
    cannot "avoid" walls — it IS the wall by definition. Returning 0.0 in
    the single-paragraph case (rather than a free 1.0) closes a reward-hack
    hole where the model could earn structural credit without doing
    anything structural."""
    if len(paragraphs) < 2:
        return 0.0
    longest = max(len(p) for p in paragraphs)
    if longest <= MAX_PARA_SOFT:
        return 1.0
    if longest >= MAX_PARA_HARD:
        return 0.0
    return (MAX_PARA_HARD - longest) / (MAX_PARA_HARD - MAX_PARA_SOFT)


def _payoff_shape(paragraphs: list[str]) -> float:
    """Score the last paragraph's shape. Requires it to be standalone (the
    `_paragraphs` split already guarantees blank-line separation), and to sit
    in the short-closer length band."""
    if len(paragraphs) < 2:
        # No closer if there's no body to close, or just a single paragraph.
        return 0.0
    closer = paragraphs[-1]
    n = len(closer)
    if n < PAYOFF_LOW or n >= PAYOFF_HARD:
        return 0.0
    if n <= PAYOFF_HIGH:
        # Full credit anywhere in [PAYOFF_LOW, PAYOFF_HIGH].
        return 1.0
    # Ramp from PAYOFF_HIGH (1.0) down to PAYOFF_HARD (0.0).
    return (PAYOFF_HARD - n) / (PAYOFF_HARD - PAYOFF_HIGH)


def score_structure(text: str) -> dict[str, float]:
    """Score a LinkedIn post on structural scannability.

    Returns a dict with sub-signal scores in [0, 1] and a 'total' that is the
    composed structure score, also in [0, 1]. The total_length_band acts as a
    soft multiplicative gate, NOT as a weighted term — too-short or too-long
    posts have undefined paragraph rhythm and payoff shape, so we zero them
    out rather than giving partial credit.
    """
    if not text:
        return {
            "total_length_band": 0.0,
            "paragraph_rhythm": 0.0,
            "max_paragraph_length": 0.0,
            "payoff_shape": 0.0,
            "total": 0.0,
        }

    paragraphs = _paragraphs(text)
    total_len = len(text)

    length_gate = _length_gate(text)
    rhythm = _paragraph_rhythm(paragraphs, total_len)
    max_para = _max_paragraph_length(paragraphs)
    payoff = _payoff_shape(paragraphs)

    additive = W_RHYTHM * rhythm + W_MAX * max_para + W_PAYOFF * payoff
    total = length_gate * additive

    return {
        "total_length_band": length_gate,
        "paragraph_rhythm": rhythm,
        "max_paragraph_length": max_para,
        "payoff_shape": payoff,
        "total": total,
    }
