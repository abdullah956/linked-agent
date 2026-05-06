"""Tests for reward.features.safety.score_safety.

Asserts the multiplier shape, the cliff curve, and that each sub-signal
fires on its target failure mode without firing on adjacent legitimate
content (positive name mentions especially).
"""

from __future__ import annotations

from reward.features.safety import score_safety


# --- Clean-input baseline ------------------------------------------------

def test_clean_post_returns_multiplier_one() -> None:
    text = (
        "I quit my job at Stripe last month after 6 years.\n\n"
        "The hardest part was admitting I had stopped learning."
    )
    result = score_safety(text)
    assert result["total_hits"] == 0
    assert result["multiplier"] == 1.0


# --- Banned phrases ------------------------------------------------------

def test_one_banned_phrase_drops_to_cliff() -> None:
    """A single hit must multiplicatively cliff to 0.3 — that's the
    semantic. One slur is career-ending, not 'expensive but survivable'."""
    text = "Some people on this team are just retards. We are getting nowhere."
    result = score_safety(text)
    assert result["banned_phrases"] >= 1
    assert result["multiplier"] == 0.3


# --- Multiple hits across signals ---------------------------------------

def test_two_or_more_hits_floor_to_0_2() -> None:
    """Two hits across DIFFERENT sub-signals — must floor at 0.2 per the
    cliff curve. Diminishing-returns once unrecoverable."""
    text = (
        "John Smith from Stripe ghosted me on three interviews and is a fraud.\n\n"
        "Reach me at malicious-leak@example.com if you want details."
    )
    result = score_safety(text)
    # naming_and_shaming should fire AND private_info (email) should fire.
    assert result["naming_and_shaming"] >= 1
    assert result["private_info"] >= 1
    assert result["total_hits"] >= 2
    assert result["multiplier"] == 0.2


# --- Naming-and-shaming: positive case ----------------------------------

def test_naming_shaming_fires_on_negative_attribution() -> None:
    """Specific named person + negative verb in proximity → fires."""
    text = (
        "John Smith from Stripe ghosted me on three different interviews "
        "and never replied to my follow-ups."
    )
    result = score_safety(text)
    assert result["naming_and_shaming"] >= 1
    # Debug should record the matched (name, verb) pair so reward-hack
    # postmortems can inspect what actually fired.
    pairs = result["debug"]["name_verb_pairs"]
    assert len(pairs) >= 1
    name_text, verb_text = pairs[0]
    assert "John" in name_text or "Smith" in name_text or "Stripe" in name_text
    assert "ghosted" in verb_text.lower()


# --- Naming-and-shaming: negative case (positive mention) ---------------

def test_naming_shaming_does_not_fire_on_positive_mention() -> None:
    """Same name shape, no negative verb anywhere — must NOT fire.
    Positive coworker shoutouts are a core LinkedIn use case."""
    text = (
        "Sarah at Anthropic taught me how to think about evaluation harness "
        "design. Best mentor I've had in years."
    )
    result = score_safety(text)
    assert result["naming_and_shaming"] == 0
    assert result["multiplier"] == 1.0


def test_naming_shaming_does_not_fire_on_anonymous_critique() -> None:
    """No proper name + negative verb is generic complaining, not naming-
    and-shaming. Belongs in style territory, not safety."""
    text = "My last manager was awful. I learned what kind of leader I never want to be."
    result = score_safety(text)
    assert result["naming_and_shaming"] == 0


# --- Private info: email -------------------------------------------------

def test_email_address_in_post_body_fires_private_info() -> None:
    text = "Reach me at jane.doe.test@example.com to discuss."
    result = score_safety(text)
    assert result["private_info"] >= 1
    private = result["debug"]["private"]
    assert any(kind == "email" for kind, _ in private)


# --- Private info: phone -------------------------------------------------

def test_phone_number_in_post_body_fires_private_info() -> None:
    text = "Call me directly at (415) 555-1234 if you want to chat."
    result = score_safety(text)
    assert result["private_info"] >= 1
    private = result["debug"]["private"]
    assert any(kind.startswith("phone") for kind, _ in private)


# --- Empty input -------------------------------------------------------

def test_empty_input_returns_multiplier_one() -> None:
    """Absence of content is not unsafe content. Safety is not responsible
    for input validation — the additive features already handle empty
    inputs by returning 0, and 1.0 * 0 = 0 composes correctly upstream.
    The composition layer in score.py adds an explicit short-circuit on
    top of this for defense in depth."""
    result = score_safety("")
    assert result["total_hits"] == 0
    assert result["multiplier"] == 1.0
