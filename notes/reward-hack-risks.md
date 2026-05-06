# Reward Hack Risks — Anticipated

A running log of reward-hack vectors we've identified during reward design,
*before* training. Each entry: the hack, where the seam lives, what catches
it, and how confident we are that it's caught. Update with actual hacks
discovered during training in `LEARNINGS.md` (this file is for predictions;
that file is for evidence).

---

## R1 — Cliched generic closer ("thoughts?", "agree?")  [STATUS: closed]

- **Hack**: Model produces a structurally-perfect post (good rhythm, short
  closer paragraph) but the closer is a generic engagement-bait question.
  Earns full `payoff_shape` credit in `structure.py` for free.
- **Seam**: between `structure.py` (shape of closer) and `style.py`
  (content of closer). Structure deliberately does NOT score closer
  *content* — that's a style concern.
- **Catches**: `style.py` cliched-ending detector at `W_CLOSER = 0.35`,
  the highest weight in the style module. The patterns live in
  `notes/cliched_closers.md` (44+ patterns across engagement-bait questions,
  CTA stock phrases, cliched aphorisms, fake-vulnerability lead-ins).
- **Invariant**: the global product `W_CLOSER * style_global_weight` must
  exceed `W_PAYOFF * structure_global_weight`, so a cliched closer always
  costs more than the free credit `structure.payoff_shape` pays for any
  short closer. Re-verify after `weights.yaml` is retuned during
  human-correlation calibration.
- **Validation**: `tests/test_style.py::test_r1_cliched_closer_costs_more_than_clean_closer`
  asserts the closed-form: same body, two closers, the cliched variant
  scores strictly lower. Empirical differential matched the predicted
  W_CLOSER (0.350) exactly.
- **Confidence**: high. Test passes with the predicted differential.
  Re-check after global weights are retuned.

---

## R2 — Fabricated-but-plausible specifics  [STATUS: acknowledged, open by design]

- **Hack**: Model writes confident-sounding posts with fabricated specifics
  ("I closed a $4.2M deal in 36 hours by...", "we cut p99 latency 87% in
  two weeks"). The numbers and shape look like a real LinkedIn post; the
  underlying claim is invented.
- **Seam**: nominally between `style.py` (does it sound like a real post)
  and `safety.py` (would a PR person delete this). Neither catches it.
- **Why we are not catching this**: there is no cheap regex signal that
  distinguishes "I closed a $4.2M deal in 36 hours" from "we shipped to 12
  customers in Q3". Both are first-person + specific number + short time
  window. The implausibility lives in *plausibility-checking against
  context* — domain-knowledge or LLM-judge territory. An LLM judge against
  the same model family we're training would also be circular.
- **Decision**: ship without a fabricated-specifics detector. A noisy
  detector that catches "$4.2M in 36 hours" but also flags legitimate
  "we shipped to 12 customers in Q3" would punish the model for being
  specific — exactly the wrong direction for hook quality.
- **Mitigation we DO ship**: `safety.naming_and_shaming` catches the
  subset of fabrication that includes attacking a named real person, and
  `style.ai_tells` catches the buzzword-soup variants. Pure numerical
  fabrication without those tells passes through.
- **Watch for during training**: posts that score high on hook (specificity
  saturated) and low/middling on style. Sample those manually during
  validation. If the model converges on a "fabricate impressive numbers"
  policy, document the discovery in LEARNINGS.md and consider an
  LLM-judge overlay (separate from the deterministic reward) as a
  post-training filter rather than retrofitting it into the reward.
- **Confidence**: this is an acknowledged limitation, not a closed risk.

---
