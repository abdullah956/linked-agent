"""Tests for reward.features.structure.score_structure.

Asserts ranges, not exact values. The total_length_band is a multiplicative
gate by deliberate design (see structure.py docstring), so when the gate is
0 the total must be 0 regardless of the additive sub-signals.
"""

from __future__ import annotations

from reward.features.structure import score_structure


# Shared sentence chunk used to build paragraphs of predictable length.
# 100 chars: "After seven years at a large company I quit on a Tuesday with no plan and saved cash kept me ok."
P100 = "After seven years at a large company I quit on a Tuesday with no plan and saved cash kept me ok."


def test_well_broken_post_in_band() -> None:
    """5 short paragraphs, ~600 chars, short closer. Should score high."""
    text = "\n\n".join([
        "After 7 years at Google, I quit on a Tuesday with no plan.",
        "The first month was the hardest. I rewrote my resume four times and threw out every draft.",
        "Then I built a small thing. It worked. I built another. That worked too.",
        "Six months in, I had a portfolio of three real projects and two interviews booked.",
        "Quitting without a plan is not advice. It just turned out to be the only way I would have started.",
    ])
    assert 400 < len(text) < 1400
    result = score_structure(text)
    assert result["total_length_band"] == 1.0
    assert result["paragraph_rhythm"] == 1.0
    assert result["max_paragraph_length"] == 1.0
    assert result["payoff_shape"] == 1.0
    assert result["total"] >= 0.95


def test_wall_of_text_post() -> None:
    """One ~1000-char paragraph. Length gate passes, but rhythm collapses
    because chars-per-para is way past the wall, max_para_length zeroes,
    and there's no closer."""
    text = (P100 + " ") * 10
    text = text.strip()
    assert 800 < len(text) < 1200
    result = score_structure(text)
    assert result["total_length_band"] == 1.0   # length is fine
    assert result["paragraph_rhythm"] == 0.0    # one paragraph -> no rhythm
    assert result["max_paragraph_length"] == 0.0  # past MAX_PARA_HARD
    assert result["payoff_shape"] == 0.0        # no closer
    assert result["total"] == 0.0


def test_too_short_post_gate_zeroes_total() -> None:
    """80-char post — below LEN_HARD_MIN. Gate is 0; total must be 0."""
    text = "Quit my job today. Scared. Excited. Both. We will see."
    assert len(text) < 200
    result = score_structure(text)
    assert result["total_length_band"] == 0.0
    assert result["total"] == 0.0


def test_thousand_char_comfortable_band() -> None:
    """~1000 chars, broken into 5 paragraphs with a short closer."""
    body = "\n\n".join([P100 * 2, P100 * 2, P100 * 2, P100 * 2, P100 * 2])  # 5 paragraphs of ~192 chars
    closer = "And that is how I learned to ship before I felt ready."
    text = body + "\n\n" + closer
    assert 900 < len(text) < 1200
    result = score_structure(text)
    assert result["total_length_band"] == 1.0
    assert result["paragraph_rhythm"] >= 0.9
    assert result["max_paragraph_length"] == 1.0
    assert result["payoff_shape"] == 1.0
    assert result["total"] >= 0.9


def test_eighteen_hundred_char_partial_gate() -> None:
    """1800-char post: inside [LEN_SOFT_MAX=1400, LEN_HARD_MAX=2200].
    The length gate should be a partial multiplier (not 0, not 1)."""
    para = P100 * 3  # ~300 chars
    text = "\n\n".join([para] * 6) + "\n\n" + "Short closer that lands the post."
    assert 1400 < len(text) < 2200
    result = score_structure(text)
    # Gate is in the ramp-down zone — strictly between 0 and 1.
    assert 0.0 < result["total_length_band"] < 1.0
    # Sub-signals can still be strong; total is gate * additive.
    assert result["payoff_shape"] == 1.0
    assert result["total"] < 1.0
    assert result["total"] > 0.0


def test_single_paragraph_500_char_post() -> None:
    """One paragraph, ~500 chars. Single-paragraph posts cannot have
    structure: rhythm is undefined, payoff is undefined, AND max_paragraph
    is undefined (the post IS the wall). All three sub-signals must be 0.
    Closes the reward-hack where one-para posts earn a free structural floor."""
    text = P100 * 5  # ~500 chars after stripping trailing space
    text = text.strip()
    assert 480 <= len(text) <= 520
    result = score_structure(text)
    assert result["total_length_band"] == 1.0
    assert result["paragraph_rhythm"] == 0.0    # single paragraph -> no rhythm
    assert result["max_paragraph_length"] == 0.0  # single paragraph -> not a meaningful signal
    assert result["payoff_shape"] == 0.0        # no closer
    assert result["total"] == 0.0


def test_buried_long_paragraph_among_short_ones() -> None:
    """5 short paragraphs but one of them is a 750-char wall in the middle.
    paragraph_rhythm looks okay-ish, but max_paragraph_length should catch it."""
    short = P100[:80]  # 80 chars
    wall = P100 * 8    # ~800 chars, past MAX_PARA_HARD
    text = "\n\n".join([short, short, wall, short, short])
    result = score_structure(text)
    assert result["total_length_band"] == 1.0
    assert result["max_paragraph_length"] == 0.0  # wall buried inside is caught
    # rhythm could still register because chars-per-para math averages it out;
    # the point is that max_paragraph_length is the guard.
    assert result["total"] < result["paragraph_rhythm"]  # max_para drags total down


def test_empty_post() -> None:
    result = score_structure("")
    assert all(v == 0.0 for v in result.values())


def test_two_thousand_two_hundred_char_post_at_hard_edge() -> None:
    """At/past LEN_HARD_MAX=2200, gate should be 0."""
    text = (P100 + "\n\n") * 25  # ~2450 chars, comfortably past the hard cap
    assert len(text) >= 2200
    result = score_structure(text)
    assert result["total_length_band"] == 0.0
    assert result["total"] == 0.0
