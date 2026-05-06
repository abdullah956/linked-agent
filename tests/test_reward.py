"""Composition tests for reward.score.score_post.

The single most important test in this file is
`test_safety_multiplier_dominates_at_composition` — it proves the
multiplicative-vs-additive design choice for safety actually works at the
composition layer. If a high-additive post with one banned phrase doesn't
end up dominated by the safety multiplier, the cliff is decorative.

Future additions (once data is populated):
  - load good_posts.yaml + bad_posts.yaml
  - assert mean(score(good)) > mean(score(bad)) by a clear margin
  - assert correlation between score and my_rating > some threshold
"""

from __future__ import annotations

from reward.score import score_post


def _high_additive_post() -> str:
    """A post that scores well across hook, structure, and style. Used as
    the baseline for the safety-multiplier test — we need to prove the
    multiplier dominates a high additive score, not a borderline one."""
    return (
        "I quit my job at Stripe last month after 6 years.\n\n"
        "The hardest part was not the money. It was admitting I had stopped "
        "learning two years ago and pretended otherwise.\n\n"
        "I built three small tools in the first month off. Two were bad. "
        "One was the seed of what I am working on now.\n\n"
        "Quitting was not the hard call. Pretending I was still growing was."
    )


def test_high_additive_post_scores_high() -> None:
    """Sanity check: the baseline post should clear ~0.85 on additive."""
    result = score_post(_high_additive_post())
    assert result["safety"]["multiplier"] == 1.0
    assert result["additive"] >= 0.85
    assert result["total"] >= 0.85


def test_safety_multiplier_dominates_at_composition() -> None:
    """The non-negotiable composition test.

    A post that scores high on additive features but contains 1 banned
    phrase must compose to ≤ 0.30 of full reward. This proves:
      (a) safety.multiplier is actually applied, not silently dropped
      (b) the cliff at 1 hit (multiplier=0.3) dominates ANY additive
          score, so the model cannot trade off "good content" against
          "small safety violation"
      (c) the composed total = additive * 0.3 — math holds end-to-end

    Bound: total ≤ additive * 0.3 + 1e-9 (floating-point slack). This is
    the cleanest formulation — the bound scales with whatever additive
    score the post actually earns, rather than a hardcoded threshold
    based on a guess at the baseline.
    """
    base = _high_additive_post()
    # Insert a banned phrase mid-post without breaking structure too much.
    contaminated = base.replace(
        "Two were bad.",
        "Two were bad. Some critics on Twitter called me a retard for shipping them.",
    )

    base_result = score_post(base)
    contam_result = score_post(contaminated)

    # The contaminated post should still score high on additive features —
    # we only added one extra sentence; hook/structure/style are mostly
    # untouched. Enforce that additive stays >= 0.7 so this test is
    # actually exercising the multiplier path, not coincidentally passing
    # because the additive collapsed.
    assert contam_result["additive"] >= 0.7, (
        f"additive collapsed to {contam_result['additive']:.3f}; "
        "this test no longer proves multiplier dominance"
    )
    assert contam_result["safety"]["total_hits"] >= 1
    assert contam_result["safety"]["multiplier"] == 0.3
    # The composed total must equal additive * multiplier exactly (within FP).
    assert abs(contam_result["total"] - contam_result["additive"] * 0.3) < 1e-9
    # And the differential must be substantial — safety should drop the
    # composed total to no more than 1/3 of the clean version.
    assert contam_result["total"] <= 0.35 * base_result["total"]


def test_empty_input_short_circuits_in_composition_layer() -> None:
    """Composition layer must handle empty input explicitly, NOT rely on
    multiplicative behavior. Defense in depth — even if a feature module
    ever returned something nonzero on empty input, the composition layer
    catches it first."""
    result = score_post("")
    assert result["total"] == 0.0
    assert result["additive"] == 0.0


def test_whitespace_only_input_short_circuits() -> None:
    result = score_post("   \n\n  \t  ")
    assert result["total"] == 0.0
    assert result["additive"] == 0.0
