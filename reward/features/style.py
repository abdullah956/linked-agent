"""Style feature: LinkedIn voice without crossing into cringe.

Sub-signals (each in [0, 1]):
  - cliched_closer:  R1 mitigator. 0 if the closer matches a curated
                     pattern from notes/cliched_closers.md, 1 otherwise.
                     Multi-paragraph posts match the last paragraph;
                     single-paragraph posts match the last 200 chars.
  - ai_tells:        graded penalty for known AI-generated lexical/phrasal
                     tells, plus em-dash density measured from the ORIGINAL
                     (un-normalized) text.
  - personal_voice:  density of first-person pronouns. Floor 0.3 when zero
                     "I" — sharp observational posts are a valid genre.
  - noise_density:   composite of emoji / hashtag / all-caps density. One
                     sub-signal because the failure modes are correlated.

Composition:
    style = W_CLOSER * cliched_closer
          + W_AI     * ai_tells
          + W_VOICE  * personal_voice
          + W_NOISE  * noise_density

INVARIANT (R1): the global product W_CLOSER * style_global_weight must
exceed W_PAYOFF * structure_global_weight, so a cliched closer always costs
more reward than the free credit structure.payoff_shape pays for any short
closer. Re-verify after weights.yaml is retuned during human-correlation
calibration.
"""

from __future__ import annotations

import re
from pathlib import Path

from reward.features._text_utils import normalize_text


# Weights for the additive composition. Sum to 1.0.
W_CLOSER = 0.35   # R1 dominant. See INVARIANT note above.
W_AI = 0.30       # second-strongest; catches embarrassing default LLM voice
W_VOICE = 0.15    # presence-of-voice; partially captured by hook
W_NOISE = 0.20    # emoji/hashtag/caps density


# --- Cliched closer patterns ---------------------------------------------
#
# Loaded at import time from notes/cliched_closers.md. The markdown file is
# the source of truth — edit it, restart, patterns reload.

_CLOSERS_FILE = Path(__file__).resolve().parents[2] / "notes" / "cliched_closers.md"
_CODE_BLOCK_RE = re.compile(r"```\s*\n(.*?)```", re.DOTALL)


def _load_closer_patterns() -> list[re.Pattern[str]]:
    """Parse regex patterns out of fenced code blocks in cliched_closers.md.
    Each non-blank, non-comment line inside a code block is one pattern."""
    if not _CLOSERS_FILE.exists():
        return []
    md = _CLOSERS_FILE.read_text(encoding="utf-8")
    patterns: list[re.Pattern[str]] = []
    for block in _CODE_BLOCK_RE.findall(md):
        for raw in block.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                patterns.append(re.compile(line, re.IGNORECASE | re.MULTILINE))
            except re.error:
                # Skip malformed patterns rather than crashing import. Real
                # validation happens via tests; a typo in the markdown
                # shouldn't take down training.
                continue
    return patterns


# Pattern count at runtime may exceed the count declared in cliched_closers.md
# due to alternation expansion in regex compilation. Not a correctness issue.
_CLOSER_PATTERNS = _load_closer_patterns()


# --- AI tells ------------------------------------------------------------
#
# Curated. Biased toward phrases over single words to limit false positives
# (single English words like "crucial" are real words; the failure mode is
# the model overusing them, which the graded count handles).

_AI_TELL_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE) for p in [
        # Lexical tells (single words the model overuses)
        r"\bdelve\b", r"\btapestry\b", r"\bcornerstone\b", r"\btestament to\b",
        r"\bseamlessly\b", r"\bintricate\b", r"\bmultifaceted\b", r"\bnuanced\b",
        r"\bparamount\b", r"\bhone\b", r"\brealm of\b", r"\bnavigate the (?:landscape|complexities|intricacies)\b",
        r"\bharness the power of\b", r"\bboasts\b",  # "X boasts Y" is the tell
        # Phrasal tells
        r"\bin today'?s (?:[a-z-]+ )?world\b",
        r"\bit'?s important to note\b",
        r"\bit'?s worth (?:mentioning|noting)\b",
        r"\bin the realm of\b",
        r"\bat the heart of\b",
        r"\bthe key (?:is to|takeaway)\b",
        # Structural tell: "It's not just X — it's Y" / "It's not X. It's Y."
        r"\bit'?s not just\b.+\bit'?s\b",
        r"\bnot just\b[^.!?\n]*\bbut\b",
        # Buzzword soup (folded in here per design notes)
        r"\bsynergy\b", r"\bparadigm shift\b", r"\bvalue[- ]add\b",
        r"\bstakeholder alignment\b", r"\bnorth star\b",
        r"\b10x(?:ing)?\b", r"\bdisruptive\b",
    ]
)


def _ai_tells_score(normalized: str, original: str) -> float:
    """Graded count. Each lexical/phrasal hit subtracts a fixed penalty.
    Em-dash density is measured from the ORIGINAL text per the normalize_text
    docstring caveat — the em-dash itself is the signal."""
    hits = sum(1 for p in _AI_TELL_PATTERNS if p.search(normalized))

    # Em-dash density on original text: > 1 per 200 chars is "spam".
    # TODO: revalidate threshold against good_posts.yaml during human-
    # correlation calibration step (Prompt 7).
    em_dashes = original.count("—")
    if len(original) > 0:
        density = em_dashes / max(len(original), 1) * 200
        if density > 1.0:
            hits += 1

    # Each tell costs 0.15. Floor at 0.
    return max(0.0, 1.0 - 0.15 * hits)


# --- Personal voice ------------------------------------------------------

_FIRST_PERSON_RE = re.compile(
    r"\b(?:I|I'?m|I'?ve|I'?d|I'?ll|me|my|mine|myself)\b",
    re.IGNORECASE,
)
_WORD_RE = re.compile(r"\b\w+\b")

# Density bands (first-person tokens / total words)
VOICE_FLOOR = 0.30        # zero first-person → 0.30 (not catastrophic)
VOICE_RAMP_TOP = 0.03     # 3% density → full credit
VOICE_SOFT_HIGH = 0.20    # past 20%, mild ramp toward 0.7 floor
VOICE_HIGH_FLOOR = 0.70   # above SOFT_HIGH, settle here (self-absorbed but not zero)


def _personal_voice_score(normalized: str) -> float:
    words = _WORD_RE.findall(normalized)
    if not words:
        return 0.0
    fp_hits = len(_FIRST_PERSON_RE.findall(normalized))
    if fp_hits == 0:
        return VOICE_FLOOR
    density = fp_hits / len(words)
    if density >= VOICE_SOFT_HIGH:
        return VOICE_HIGH_FLOOR
    if density >= VOICE_RAMP_TOP:
        return 1.0
    # Linear ramp from VOICE_FLOOR (at density just above 0) up to 1.0 at VOICE_RAMP_TOP.
    return VOICE_FLOOR + (1.0 - VOICE_FLOOR) * (density / VOICE_RAMP_TOP)


# --- Noise density -------------------------------------------------------
#
# Emoji / hashtag / all-caps. Composed into one signal because the failure
# modes are correlated — engagement-bait posts tend to do all three.

# Emoji range covers the bulk of pictographs/symbols on LinkedIn. Imperfect
# but cheap; misses some skin-tone modifiers and ZWJ sequences. That's fine —
# we're penalizing density, not counting precisely.
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F700-\U0001F77F"  # alchemical
    "\U0001F900-\U0001F9FF"  # supplemental symbols
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "☀-➿"          # misc symbols + dingbats
    "]"
)
_HASHTAG_RE = re.compile(r"(?<!\w)#\w+")
# All-caps words: 3+ letters, all uppercase, not a known-fine acronym.
_ALL_CAPS_RE = re.compile(r"\b[A-Z]{3,}\b")
_ALLOWED_ACRONYMS = {"AI", "API", "CEO", "CTO", "CFO", "COO", "VP", "ML", "LLM",
                     "SQL", "AWS", "GCP", "PR", "QA", "UI", "UX", "OS", "IT",
                     "HR", "VC", "B2B", "B2C", "SaaS", "IPO", "USA", "UK"}


def _noise_density_score(original: str) -> float:
    """Original text used because emojis and caps are stylistic raw signals
    that normalization would scramble (e.g. emoji-mapped chars)."""
    if not original:
        return 0.0

    emoji_count = len(_EMOJI_RE.findall(original))
    hashtag_count = len(_HASHTAG_RE.findall(original))
    caps = [w for w in _ALL_CAPS_RE.findall(original) if w not in _ALLOWED_ACRONYMS]
    caps_count = len(caps)

    # Excess past tolerated thresholds. Each excess unit costs 0.15.
    emoji_excess = max(0, emoji_count - 4)        # 0-4 fine
    hashtag_excess = max(0, hashtag_count - 5)    # 0-5 fine
    caps_excess = max(0, caps_count - 1)          # 0-1 fine

    excess_total = emoji_excess + hashtag_excess + caps_excess
    return max(0.0, 1.0 - 0.15 * excess_total)


# --- Closer extraction ---------------------------------------------------

_PARA_SPLIT_RE = re.compile(r"\n\s*\n+")


def _closer_scope(normalized: str) -> str:
    """Return the substring we match cliched-closer patterns against.
    Multi-paragraph: last paragraph after final blank-line break.
    Single-paragraph: last 200 chars of the normalized text."""
    if not normalized:
        return ""
    paras = [p.strip() for p in _PARA_SPLIT_RE.split(normalized.strip()) if p.strip()]
    if len(paras) >= 2:
        return paras[-1]
    return normalized[-200:]


def _cliched_closer_score(normalized: str) -> float:
    if not _CLOSER_PATTERNS:
        return 1.0  # no patterns loaded → can't penalize; fail open
    scope = _closer_scope(normalized)
    if not scope:
        return 1.0
    for p in _CLOSER_PATTERNS:
        if p.search(scope):
            return 0.0
    return 1.0


# --- Public API ----------------------------------------------------------

def score_style(text: str) -> dict[str, float]:
    """Score a LinkedIn post on style/voice.

    Returns a dict with sub-signal scores in [0, 1] and a 'total' that is the
    composed style score, also in [0, 1].
    """
    if not text:
        return {
            "cliched_closer": 0.0,
            "ai_tells": 0.0,
            "personal_voice": 0.0,
            "noise_density": 0.0,
            "total": 0.0,
        }

    normalized = normalize_text(text)

    cliched_closer = _cliched_closer_score(normalized)
    ai_tells = _ai_tells_score(normalized, text)
    personal_voice = _personal_voice_score(normalized)
    noise_density = _noise_density_score(text)

    total = (
        W_CLOSER * cliched_closer
        + W_AI * ai_tells
        + W_VOICE * personal_voice
        + W_NOISE * noise_density
    )

    return {
        "cliched_closer": cliched_closer,
        "ai_tells": ai_tells,
        "personal_voice": personal_voice,
        "noise_density": noise_density,
        "total": total,
    }
