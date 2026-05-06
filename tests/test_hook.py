"""Tests for reward.features.hook.score_hook.

Each case asserts a *range*, not an exact value — exact thresholds will shift
as we tune weights, but the qualitative ordering (strong > weak > dead) must
hold. If a future change breaks one of these ranges, that's a signal to
re-validate against human ratings, not to slacken the test.
"""

from __future__ import annotations

from reward.features.hook import score_hook


# --- Strong hooks ---------------------------------------------------------

def test_strong_hook_specific_number_and_personal_anchor() -> None:
    text = "After 7 years at Google, I quit on a Tuesday with no plan and $43k saved.\n\nHere's what I learned."
    result = score_hook(text)
    assert result["filler_gate"] == 1.0
    assert result["personal"] == 1.0
    assert result["specificity"] >= 0.7  # multiple specific tokens (7, Google, Tuesday, $43k)
    assert result["length_band"] == 1.0
    assert result["total"] >= 0.75


def test_strong_hook_concrete_claim_in_band() -> None:
    text = "I shipped 12 features in 90 days and only one of them mattered."
    result = score_hook(text)
    assert result["filler_gate"] == 1.0
    assert result["personal"] == 1.0
    assert result["specificity"] >= 0.5
    assert result["length_band"] == 1.0
    assert result["total"] >= 0.6


# --- Weak hooks -----------------------------------------------------------

def test_weak_hook_filler_opener_zeroes_total() -> None:
    text = "Excited to announce that I'm joining Acme as Head of Engineering!"
    result = score_hook(text)
    assert result["filler_gate"] == 0.0
    assert result["total"] == 0.0  # gate kills the score regardless of specificity


def test_weak_hook_abstract_third_person() -> None:
    # No filler, no specificity, abstract subject — should be low but not zero.
    text = "Companies that adopt AI will outperform those that don't."
    result = score_hook(text)
    assert result["filler_gate"] == 1.0
    assert result["personal"] == 0.7  # abstract penalty applied
    assert result["total"] < 0.5


# --- Edge cases -----------------------------------------------------------

def test_edge_empty_string() -> None:
    result = score_hook("")
    assert result["filler_gate"] == 0.0
    assert result["specificity"] == 0.0
    assert result["length_band"] == 0.0
    assert result["total"] == 0.0


def test_edge_single_character() -> None:
    result = score_hook("X")
    # Below HARD_MIN length, so length_band must be 0; specificity ~0.
    assert result["length_band"] == 0.0
    assert result["specificity"] <= 0.1
    assert result["total"] <= 0.05


def test_edge_very_long_single_line() -> None:
    # 250+ char single line — past HARD_MAX, length_band collapses to 0.
    text = (
        "I have spent the last several years thinking very carefully about a "
        "wide range of topics related to engineering leadership and the way "
        "that teams collaborate across functions in modern technology orgs."
    )
    assert len(text) > 180
    result = score_hook(text)
    assert result["length_band"] == 0.0
    # Even with some proper nouns, total is dragged down by zero length_band
    # and the absence of strong specificity tokens.
    assert result["total"] <= 0.45


# --- Sub-signal sanity checks --------------------------------------------

def test_filler_gate_is_case_insensitive() -> None:
    for opener in [
        "EXCITED TO ANNOUNCE my new role at Stripe.",
        "I'm thrilled to share some news from the team this week.",
        "In today's fast-paced world, focus is the rarest skill.",
        "Unpopular opinion: most meetings should be docs.",
    ]:
        assert score_hook(opener)["filler_gate"] == 0.0, f"missed: {opener}"


def test_clean_concrete_opener_passes_gate() -> None:
    text = "I quit my job at Stripe last month after 6 years."
    result = score_hook(text)
    assert result["filler_gate"] == 1.0
    assert result["personal"] == 1.0
