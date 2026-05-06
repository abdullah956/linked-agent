"""Hook feature: does the first line grab attention?

Scores the opening line on four sub-signals, returning a dict so callers can
inspect each component separately (essential for debugging reward hacking).

Sub-signals:
  - filler_gate:    boolean-ish gate. 0 if the opener matches a known dead
                    LinkedIn cliche, 1 otherwise. Multiplicative — a filler
                    open zeros out the rest of the hook score.
  - specificity:    concrete content in the first line — digits, percentages,
                    money, time units, proper-noun-shaped tokens. Saturating.
  - length_band:    trapezoidal score over first-line length. Sweet spot is
                    [40, 140] chars; ramps to 0 outside [25, 180].
  - personal:       multiplier (not weighted term). 0.7 if the opener reads as
                    abstract / third-person ("Companies that...", "Leaders
                    should..."), 1.0 otherwise.

Composition:
    hook = filler_gate * personal * (W_SPEC * specificity + W_LEN * length_band)
"""

from __future__ import annotations

import re

from reward.features._text_utils import normalize_text

# Weights for the additive portion of the hook score. They sum to 1.0 so the
# additive part stays in [0, 1] before the gate / personal multipliers apply.
W_SPECIFICITY = 0.65  # concrete content is the dominant positive signal
W_LENGTH = 0.35       # length matters but only as "are you in the readable band"

# Multiplier applied when the opener is abstract/third-person. Soft penalty —
# abstract openers aren't automatically dead, just usually weak.
ABSTRACT_PENALTY = 0.7

# First-line length band (characters). Outside [HARD_MIN, HARD_MAX] -> 0.
SOFT_MIN, SOFT_MAX = 40, 140
HARD_MIN, HARD_MAX = 25, 180

# Saturating curve for specificity: this many "specific" tokens -> ~0.9.
SPECIFICITY_SATURATION = 3.0


# Cliche LinkedIn openers. Each pattern is anchored to start-of-line and
# matched case-insensitively. Grow this list as the policy finds new dead
# phrases — it is data, not code.
FILLER_OPENERS: tuple[str, ...] = (
    r"excited to (announce|share)",
    r"i'?m (thrilled|excited|humbled|honored) to",
    r"thrilled to (announce|share)",
    r"humbled to (announce|share)",
    r"honored to (announce|share)",
    r"happy to (announce|share)",
    r"proud to (announce|share)",
    r"delighted to (announce|share)",
    r"in today'?s (fast[- ]paced|ever[- ]changing|digital) world",
    r"in this day and age",
    r"are you ready to",
    r"let me tell you (a story )?about",
    r"have you ever (wondered|thought about|asked yourself)",
    r"did you know that",
    r"picture this",
    r"imagine (a world|this)",
    r"here'?s the thing[:.]",
    r"big (news|announcement)",
    r"(🚀|🎉|🔥)\s*(big|huge|exciting)",  # rocket/party/fire + hype word
    r"(it'?s| i am )?(with great|with deep) (pleasure|excitement|pride)",
    r"i wanted to take a moment",
    r"just wanted to share",
    r"quick (update|post|note)[:.]",
    r"buckle up",
    r"hot take[:.]",
    r"unpopular opinion[:.]",  # so popular it's now a cliche
    r"plot twist[:.]",
    r"life hack[:.]",
    r"game[- ]changer",
    r"breaking[:.]",
)

_FILLER_RE = re.compile(
    "|".join(f"(?:{p})" for p in FILLER_OPENERS),
    re.IGNORECASE,
)

# Abstract / third-person openers — generic plural subject + verb, or
# universal-modal patterns. Matched on the first few tokens only.
ABSTRACT_SUBJECTS = (
    "companies", "businesses", "organizations", "teams", "leaders",
    "managers", "engineers", "developers", "founders", "startups",
    "people", "everyone", "nobody", "society", "ai",
)
_ABSTRACT_RE = re.compile(
    r"^\s*(?:"
    + "|".join(ABSTRACT_SUBJECTS)
    + r")\b",
    re.IGNORECASE,
)
_MODAL_ABSTRACT_RE = re.compile(
    r"^\s*(?:you|everyone|nobody|we all)\s+(?:should|must|need to|have to)\b",
    re.IGNORECASE,
)

# Specificity token patterns.
_NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b")
_PERCENT_RE = re.compile(r"\d+\s*%")
_MONEY_RE = re.compile(r"[$€£]\s*\d|\b\d+\s*(?:k|m|bn|b)\b", re.IGNORECASE)
_TIME_UNIT_RE = re.compile(
    r"\b\d+\s*(?:second|minute|hour|day|week|month|year|quarter|qtr)s?\b",
    re.IGNORECASE,
)
# Proper-noun-shaped token: capitalized word that is NOT the very first word
# of the line and not "I". Imperfect but cheap.
_PROPER_NOUN_RE = re.compile(r"(?<=\s)[A-Z][a-zA-Z]{2,}")


def _first_line(text: str) -> str:
    """Extract what we treat as the 'hook' — text up to the first paragraph
    break or sentence terminator, whichever comes first. Stable definition so
    scores don't jitter on minor formatting changes."""
    if not text:
        return ""
    stripped = text.lstrip()
    # Cut at first newline.
    nl = stripped.find("\n")
    candidate = stripped if nl == -1 else stripped[:nl]
    # Cut at first sentence terminator followed by space (".", "!", "?").
    m = re.search(r"[.!?]\s", candidate)
    if m:
        candidate = candidate[: m.end() - 1]
    return candidate.strip()


def _filler_gate(line: str) -> float:
    if not line:
        return 0.0
    return 0.0 if _FILLER_RE.match(line) else 1.0


def _personal_multiplier(line: str) -> float:
    if not line:
        return 1.0
    if _ABSTRACT_RE.match(line) or _MODAL_ABSTRACT_RE.match(line):
        return ABSTRACT_PENALTY
    return 1.0


def _specificity(line: str) -> float:
    if not line:
        return 0.0
    hits = 0
    hits += len(_NUMBER_RE.findall(line))
    hits += len(_PERCENT_RE.findall(line))
    hits += len(_MONEY_RE.findall(line))
    hits += len(_TIME_UNIT_RE.findall(line))
    hits += len(_PROPER_NOUN_RE.findall(line))
    # Saturating curve: 1 - exp(-hits / SAT). 3 hits -> ~0.63, 5 -> ~0.81.
    # Adjust the scale so 3 hits lands near 0.9 (matches the docstring intent).
    import math
    return 1.0 - math.exp(-hits / (SPECIFICITY_SATURATION / 2.3))


def _length_band(line: str) -> float:
    n = len(line)
    if n <= HARD_MIN or n >= HARD_MAX:
        return 0.0
    if SOFT_MIN <= n <= SOFT_MAX:
        return 1.0
    if n < SOFT_MIN:
        return (n - HARD_MIN) / (SOFT_MIN - HARD_MIN)
    return (HARD_MAX - n) / (HARD_MAX - SOFT_MAX)


def score_hook(text: str) -> dict[str, float]:
    """Score the opening line of a LinkedIn post.

    Returns a dict with sub-signal scores in [0, 1] and a 'total' that is the
    composed hook score, also in [0, 1].
    """
    # Normalize typographic variants (curly apostrophes/quotes, em/en dashes)
    # so patterns like "i'?m thrilled" match iOS-autocorrected input.
    line = _first_line(normalize_text(text))

    filler_gate = _filler_gate(line)
    personal = _personal_multiplier(line)
    specificity = _specificity(line)
    length_band = _length_band(line)

    additive = W_SPECIFICITY * specificity + W_LENGTH * length_band
    total = filler_gate * personal * additive

    return {
        "filler_gate": filler_gate,
        "specificity": specificity,
        "length_band": length_band,
        "personal": personal,
        "total": total,
    }
