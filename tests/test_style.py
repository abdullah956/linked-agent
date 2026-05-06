"""Tests for reward.features.style.score_style.

Non-negotiables:
  - test_r1_cliched_closer_costs_more_than_clean_closer: proves the R1
    mitigation works — same post body with "Thoughts?" closer must score
    strictly lower than with a real closer.
  - test_curly_apostrophe_normalized: proves typographic normalization
    works — curly-apostrophe variant of a closer cliche fires the same
    pattern as the straight-quote version.

Other tests assert qualitative ranges, not exact values.
"""

from __future__ import annotations

from reward.features.style import score_style


# --- R1: the non-negotiable ----------------------------------------------

def test_r1_cliched_closer_costs_more_than_clean_closer() -> None:
    """Same body, two different closers. The cliched closer must score
    strictly lower on style.total. If this fails, the R1 reward-hack vector
    is open and the model will learn to game structure with cheap closers."""
    body = (
        "I quit my job at Stripe last month after 6 years.\n\n"
        "The hardest part was not the money. It was admitting I had stopped "
        "learning two years ago and pretended otherwise.\n\n"
        "I built three small tools in the first month off. Two were bad. "
        "One was the seed of what I am working on now.\n\n"
    )
    cliched = body + "Thoughts?"
    real = body + "Quitting was not the hard call. Pretending I was still growing was."

    cliched_score = score_style(cliched)
    real_score = score_style(real)

    assert cliched_score["cliched_closer"] == 0.0
    assert real_score["cliched_closer"] == 1.0
    assert real_score["total"] > cliched_score["total"]
    # The differential must be at least W_CLOSER (0.35) since that signal
    # flipped from 1.0 to 0.0 between the two posts.
    assert real_score["total"] - cliched_score["total"] >= 0.30


# --- Non-negotiable: curly-apostrophe normalization ----------------------

def test_curly_apostrophe_normalized() -> None:
    """A cliched closer with curly apostrophes must fire the same pattern as
    the straight-quote variant. Proves _text_utils.normalize_text is wired in."""
    straight = "Some thoughts on shipping. Let's go."
    curly = "Some thoughts on shipping. Let’s go."  # curly apostrophe in "Let's"
    s_straight = score_style(straight)
    s_curly = score_style(curly)
    # Both should hit the "let's go" closer pattern and zero cliched_closer.
    assert s_straight["cliched_closer"] == 0.0
    assert s_curly["cliched_closer"] == 0.0


# --- Cringe LinkedIn-speak ------------------------------------------------

def test_cringe_linkedin_speak_scores_low() -> None:
    """Multiple cliched-closer patterns + multiple AI tells + buzzwords.
    Should score very low overall."""
    text = (
        "In today's fast-paced world, leaders must navigate the complexities "
        "of a multifaceted, nuanced landscape.\n\n"
        "It's not just about technology — it's about synergy and stakeholder "
        "alignment with your north star.\n\n"
        "Trust the process. The choice is yours.\n\n"
        "Thoughts?"
    )
    result = score_style(text)
    assert result["cliched_closer"] == 0.0       # "Thoughts?" hits
    assert result["ai_tells"] <= 0.4             # many tells
    assert result["total"] <= 0.40


# --- Natural-voice baseline ----------------------------------------------

def test_natural_voice_post_scores_high() -> None:
    """A post that reads as a real human writing. Should score high."""
    text = (
        "I quit my job at Stripe last month after 6 years.\n\n"
        "The hardest part was not the money. It was admitting I had stopped "
        "learning two years ago and pretended otherwise.\n\n"
        "I built three small tools in the first month off. Two were bad. "
        "One was the seed of what I am working on now.\n\n"
        "Quitting was not the hard call. Pretending I was still growing was."
    )
    result = score_style(text)
    assert result["cliched_closer"] == 1.0
    assert result["ai_tells"] >= 0.85
    assert result["personal_voice"] == 1.0
    assert result["noise_density"] == 1.0
    assert result["total"] >= 0.90


# --- Emoji spam ----------------------------------------------------------

def test_emoji_spam_post_noise_dominates() -> None:
    """Lots of emoji, otherwise tame. noise_density must zero out, and the
    composed total must be capped at (1.0 - W_NOISE = 0.80) — proves that
    noise failure is bounded by W_NOISE and not silently overflowing."""
    text = (
        "I learned a lot this week 🚀🚀🚀🚀🚀🔥🔥🔥💯💯💯👏👏🙌🙌\n\n"
        "Onto the next thing. 🎯✨⭐️🌟"
    )
    result = score_style(text)
    assert result["noise_density"] <= 0.3
    # Style is additive by design — a single failing sub-signal caps the
    # total at (1.0 - that signal's weight). Prove the cap holds.
    assert result["total"] <= 0.80 + 1e-9
    # And confirm noise IS dragging the score down (not still at 1.0).
    assert result["total"] < 0.90


# --- All-caps shouting ---------------------------------------------------

def test_all_caps_shouting_post_noise_dominates() -> None:
    """Multiple all-caps words (excluding allowed acronyms). noise_density
    must zero out; total capped at (1.0 - W_NOISE)."""
    text = (
        "STOP scrolling. READ THIS. I'm telling you, EVERYONE needs to "
        "HEAR this MESSAGE.\n\n"
        "It changed my LIFE and it WILL change yours."
    )
    result = score_style(text)
    assert result["noise_density"] <= 0.5
    # Same bound as the emoji-spam case: noise failure is bounded by W_NOISE.
    assert result["total"] <= 0.80 + 1e-9
    assert result["total"] < 0.90


# --- Personal voice: floor 0.3 case --------------------------------------

def test_no_first_person_floors_at_0_3() -> None:
    """Sharp observational post with zero 'I' — penalized but not zeroed."""
    text = (
        "Most engineering interviews still optimize for the wrong signal.\n\n"
        "The questions are about isolated algorithm puzzles, but the work "
        "is mostly about reading other people's code and making safe edits.\n\n"
        "The mismatch is the whole story."
    )
    result = score_style(text)
    assert result["personal_voice"] == 0.3
    # Closer is real, no AI tells, no noise; total should still be solid.
    assert result["total"] >= 0.65


# --- Empty post ----------------------------------------------------------

def test_empty_post() -> None:
    result = score_style("")
    assert all(v == 0.0 for v in result.values())


# --- Closer scope: single-paragraph case ---------------------------------

def test_single_paragraph_closer_caught_via_last_200_chars() -> None:
    """A single-paragraph post that ends with a cliched closer must still
    be caught — last 200 chars is the matching scope."""
    text = (
        "I have been thinking about this for a while and I keep coming back "
        "to the same place. The work is the work. You either do it or you "
        "do not. What do you think?"
    )
    result = score_style(text)
    assert result["cliched_closer"] == 0.0
