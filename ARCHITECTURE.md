# Architecture

Snapshot of how this repo is organized, why each piece exists, and where the
boundaries live. Intended audience: someone reading the codebase for the
first time who needs to make a non-trivial change.

This is a description of the **current** state, not the final state. Pieces
marked _stub_ are deliberately unbuilt — the project plan is to fully
calibrate the reward function against human ratings before writing the env
or training code, on the principle that _bad reward = bad model no matter
how fancy the RL_.

---

## 1. High-level shape

The project is a custom RL pipeline that fine-tunes a small open-weight LLM
(Qwen3-1.7B) to write LinkedIn posts that match the author's taste, scored
by a deterministic reward function calibrated against the author's own
human ratings.

```
                 ┌────────────────────────────────────────┐
                 │            REWARD FUNCTION             │
                 │   reward/score.py + reward/features/   │
                 │                                        │
                 │   Pure functions. No ML. No I/O at     │
                 │   scoring time. Composable, testable,  │
                 │   inspectable per-sub-signal.          │
                 └─────────────────┬──────────────────────┘
                                   │ score_post(text) -> dict
                                   ▼
        ┌──────────────────────────┴───────────────────────────┐
        │                                                      │
        ▼                                                      ▼
┌───────────────────────┐                          ┌─────────────────────────┐
│  CALIBRATION          │                          │  RL TRAINING (stub)     │
│  scripts/             │                          │  env/  agent/  training/│
│                       │                          │                         │
│  Human-in-the-loop    │   gates                  │  Gymnasium env wraps    │
│  rating + Spearman    │ ─────────►               │  Qwen3 + score_post.    │
│  correlation against  │   ρ > 0.6                │  GRPO via TRL fine-     │
│  reward_total.        │   required               │  tunes the policy.      │
└───────────────────────┘                          └─────────────────────────┘
```

The arrow in the middle is load-bearing. Nothing on the right is built yet.
The rule is: until the reward function ranks the author's labeled posts
with Spearman ρ > 0.6, no env, no training, no compute spent on RL.
See `notes/reward-hack-risks.md` and the project plan for the rationale.

---

## 2. Directory layout

```
linkedin-agent/
├── reward/                    REWARD FUNCTION  (built, 40 tests passing)
│   ├── score.py               Composition: weighted sum × safety multiplier
│   ├── weights.yaml           Per-feature global weights (tunable)
│   └── features/
│       ├── _text_utils.py     Typographic normalization (curly→ASCII)
│       ├── hook.py            First-line attention-grab
│       ├── structure.py       Scannability / paragraph rhythm
│       ├── style.py           LinkedIn voice / AI-tells / cliched closer
│       └── safety.py          Multiplicative penalty for unsafe content
│
├── scripts/                   CALIBRATION + DATA  (calibration built; data gen stub)
│   ├── collect_human_labels.py   Interactive CLI: rate posts 1-10
│   ├── validate_reward.py        Spearman/Pearson + per-feature diagnostics
│   └── generate_baseline.py      [stub] generate posts from base model
│
├── data/                      INPUTS  (configs + curated examples)
│   ├── topics.yaml                ~20 topic prompts the agent rolls out from
│   ├── good_posts.yaml            curated "good" reference posts
│   ├── bad_posts.yaml             curated "bad" reference posts
│   ├── safety_banned.example.yaml  public-safe minimal banned-phrase list
│   └── safety_banned.yaml          [gitignored] real production list
│
├── notes/                     DESIGN DOCS
│   ├── cliched_closers.md         pattern source for style.cliched_closer
│   └── reward-hack-risks.md       anticipated reward-hack vectors (R1, R2, …)
│
├── tests/                     TEST SUITE  (40 passing, 0 failing)
│   ├── test_hook.py
│   ├── test_structure.py
│   ├── test_style.py
│   ├── test_safety.py
│   └── test_reward.py             composition-level tests
│
├── env/                       RL ENV  (stub)
│   └── linkedin_env.py            Gymnasium env wrapping reward + model
│
├── agent/                     POLICY  (stub)
│   ├── baseline.py                untrained Qwen3 generator
│   └── train_grpo.py              GRPO entrypoint via TRL
│
├── training/                  TRAINING ARTIFACTS  (empty placeholder)
├── results/                   RUN OUTPUTS  (empty placeholder)
│
├── README.md                  Public face of the portfolio piece
├── LEARNINGS.md               Running log of "what I thought → what I know"
├── ARCHITECTURE.md            This file
└── pyproject.toml             Deps: numpy, scipy, pyyaml, rich, transformers,
                                     peft, trl, gymnasium, wandb, pytest
```

---

## 3. The reward function in detail

This is the most important component in the project. It is deliberately
**pure**: deterministic, side-effect-free at scoring time, no ML, no
network. Every sub-signal is inspectable in the returned dict so that
reward-hack postmortems can pinpoint exactly which sub-signal is being
gamed. Module-level state (compiled regexes, loaded YAML) is built once
at import time.

### 3.1 Composition (`reward/score.py`)

```
additive  =   W_HOOK      * hook["total"]
            + W_STRUCTURE * structure["total"]
            + W_STYLE     * style["total"]

total     =   additive * safety["multiplier"]
```

Why this shape:

- **Additive** for hook/structure/style because all three are non-trivial
  on every post; they're aspects of the same artifact and trade off
  against each other.
- **Multiplicative** for safety because safety failures are categorical,
  not gradient. A clean post sees `multiplier=1.0` (no-op); a single hit
  drops to `0.3`; two-or-more drops to `0.2`. The cliff is the semantic.

Weights live in `reward/weights.yaml` — currently `hook=0.30`,
`structure=0.20`, `style=0.50`. They sum to 1.0 across the additive
features. They are explicitly NOT to be tuned by intuition; they're
tuned only after `validate_reward.py` reports per-feature Spearman
correlations on real human ratings.

Empty-input guard: `score.py` short-circuits on whitespace-only input
to `total=0.0`. Defense in depth — the additive features already handle
empty input correctly, but relying on multiplicative behavior to handle a
case the additive features should have caught is brittle.

### 3.2 Sub-features

Each feature returns a dict — `{sub_signal_1, sub_signal_2, ..., total}`
all in `[0, 1]` — so the composition layer and downstream tooling can
inspect any sub-signal directly.

#### Hook (`reward/features/hook.py`)

Scores the **first line** (text up to first newline or sentence terminator).

| Sub-signal     | Type           | What it measures |
|---             |---             |--- |
| `filler_gate`  | gate (0 or 1)  | Opener doesn't match a known LinkedIn cliche ("excited to announce", "in today's fast-paced world", …). 0 zeros out the rest. |
| `specificity`  | additive       | Saturating count of digits, %, money, time units, proper nouns. |
| `length_band`  | additive       | Trapezoidal: full credit in [40, 140] chars, ramps to 0 outside [25, 180]. |
| `personal`     | multiplier     | 0.7 if opener is abstract third-person ("Companies that…"), 1.0 otherwise. |

Composition: `filler_gate * personal * (W_SPEC * specificity + W_LEN * length_band)`.

#### Structure (`reward/features/structure.py`)

Scannability — how the post _looks_ in the LinkedIn feed.

| Sub-signal             | Type        | What it measures |
|---                     |---          |--- |
| `total_length_band`    | gate (0..1) | Trapezoidal on total post length. Full credit in [300, 1400] chars. |
| `paragraph_rhythm`     | additive    | Chars-per-paragraph density. Single-paragraph posts → 0 (no rhythm). |
| `max_paragraph_length` | additive    | Wall-of-text guard. Full credit if longest para ≤ 500 chars. |
| `payoff_shape`         | additive    | Last paragraph is short and standalone (real closer, not trailing off). |

Composition: `total_length_band * (W_RHYTHM * rhythm + W_MAX * max_para + W_PAYOFF * payoff)`.

The `payoff_shape` signal scores _shape_ only, not _content_ — a
generic "thoughts?" closer earns the same `payoff_shape` as a real
landed thought. Content quality of the closer is `style.cliched_closer`'s
job. **This seam is intentional** and is exactly the seam reward-hack
risk R1 lives in.

#### Style (`reward/features/style.py`)

Voice. The largest weight in the global composition (0.50) — the style
of the post is what most distinguishes a real human from generated slop.

| Sub-signal       | Type     | What it measures |
|---               |---       |--- |
| `cliched_closer` | additive | 0 if closer matches a curated pattern from `notes/cliched_closers.md`, else 1. R1 mitigator. |
| `ai_tells`       | additive | Graded penalty for known AI-generated lexical/phrasal tells, plus em-dash density (measured on **original** text, not normalized). |
| `personal_voice` | additive | Density of first-person pronouns. Floor 0.3 when zero "I" — observational posts are a valid genre. |
| `noise_density`  | additive | Composite of emoji + hashtag + all-caps density. One signal because the failure modes are correlated. |

Composition is a straight weighted sum:
`W_CLOSER*0.35 + W_AI*0.30 + W_VOICE*0.15 + W_NOISE*0.20`.

**INVARIANT (R1)**: the global product `W_CLOSER * style_global_weight`
must exceed `W_PAYOFF * structure_global_weight`, so a cliched closer
always costs more reward than the free credit `structure.payoff_shape`
pays for any short closer. Documented at the top of `style.py`.
Re-verify after `weights.yaml` is retuned during calibration.

#### Safety (`reward/features/safety.py`)

Different shape from the other three: returns a `multiplier` in [0.2, 1.0],
not a weighted total. Hits are reported as raw integer counts plus a
`debug` block listing exactly which patterns fired (essential for
postmortem when a real post gets unfairly penalized).

| Sub-signal           | What it catches |
|---                   |--- |
| `banned_phrases`     | Slurs, harassment patterns, threats, explicit sexual content. Loaded from YAML. |
| `naming_and_shaming` | Proper-name token within ~150 chars of a negative-attribution verb ("ghosted", "stole my…", "is a fraud"). |
| `private_info`       | Email addresses, phone numbers, street addresses, doxxing markers ("their home address is…"). |

**Curve**:

```
0 hits  → multiplier = 1.0
1 hit   → multiplier = 0.3
2+ hits → multiplier = 0.2
```

This is a cliff, not a ramp, on purpose. Soften the cliff and you teach
the model that violations are expensive-but-survivable, which is the
wrong lesson.

**Banned-phrase list policy**: shipping a comprehensive slur inventory
in a public repo creates the exact harm vectors the inventory is meant
to prevent. The loader prefers `data/safety_banned.yaml` (gitignored,
real production list) and falls back to `data/safety_banned.example.yaml`
(public-safe minimal list, ~5-10 patterns) with a logged warning when
the real list isn't present. See `LEARNINGS.md` for the full reasoning.

#### Text normalization (`reward/features/_text_utils.py`)

`normalize_text(s)` maps typographic variants (curly quotes, em/en dashes)
to ASCII so that pattern-matching features don't silently miss
iOS-autocorrected or Word-pasted input.

**HARD RULE** (per the file header): every pattern-matching feature must
call `normalize_text` on its input before matching.

**CAVEAT**: the em-dash _density_ signal in `style.ai_tells` deliberately
counts em-dashes from the **original** un-normalized text. Em-dash spam
is itself the AI tell — we shouldn't normalize away the very thing we're
trying to measure.

### 3.3 Output schema

```python
score_post(text) -> {
    "hook": {
        "filler_gate": float,
        "specificity": float,
        "length_band": float,
        "personal": float,
        "total": float,
    },
    "structure": {
        "total_length_band": float,
        "paragraph_rhythm": float,
        "max_paragraph_length": float,
        "payoff_shape": float,
        "total": float,
    },
    "style": {
        "cliched_closer": float,
        "ai_tells": float,
        "personal_voice": float,
        "noise_density": float,
        "total": float,
    },
    "safety": {
        "banned_phrases": int,        # raw hit counts
        "naming_and_shaming": int,
        "private_info": int,
        "total_hits": int,
        "multiplier": float,          # in [0.2, 1.0]
        "debug": {
            "banned": [str],          # which patterns fired
            "name_verb_pairs": [(name, verb)],
            "private": [(kind, match)],
        },
    },
    "additive": float,                # weighted sum of feature totals
    "total": float,                   # additive * safety.multiplier
}
```

The reason every sub-signal is exposed: when calibration disagrees with
human ratings, `validate_reward.py` needs to point at the specific
sub-signal that explains the gap. Hiding internals behind a flat score
would make reward-hack debugging impossible.

---

## 4. Calibration pipeline

The bridge between the reward function and any future RL work. The whole
project rests on one number: the Spearman correlation between the author's
1-10 ratings of real LinkedIn posts and `score_post(...)["total"]`. The
gate is **Spearman ρ > 0.6**.

### 4.1 `scripts/collect_human_labels.py`

Interactive CLI. Reads a text file of posts separated by lines containing
only `---`, prompts for a 1-10 rating + one-line reason for each, writes
results to `data/labeled_posts.yaml` (keyed by the first 8 chars of a
SHA-256 hash of the post text).

Design points worth knowing:

- **Score is computed _after_ the rating is recorded.** The human
  judgment must not be biased by seeing the model's score first. The
  prompt layout enforces this: rating → reason → _then_ `score_post(post)`.
- **Save after every rating**, not just at session end. Crash mid-session
  loses no work.
- **Resume by hash**. Re-running the script skips posts already in the
  output file. Whitespace edits to the input file change the hash, so
  small edits to a post will ask you to re-rate it.
- **Stats line under the panel** (chars/words/paragraphs) is shown as
  context for rating, not as a score hint.

Run:
```
python scripts/collect_human_labels.py \
    --input data/posts_to_label.txt \
    --output data/labeled_posts.yaml
```

### 4.2 `scripts/validate_reward.py`

Reads `data/labeled_posts.yaml`, prints:

1. **Headline correlations**: Spearman ρ (rank agreement — _the_ gate)
   and Pearson r (linear agreement — informational).
2. **ASCII scatter plot** of `my_rating` vs `reward_total`. Hand-drawn
   grid; no extra dep beyond `rich`.
3. **Per-feature Spearman table**: ρ of each of `hook`/`structure`/`style`
   _individually_ vs `my_rating`. Tells you _which_ feature is doing
   real work and which is noise (or anti-aligned).
4. **Top-5 disagreements**: posts with the largest
   `|my_rating_normalized − reward_total|`. For each, full text + my
   reason + breakdown + a "likely culprit" hint that names the feature
   responsible and the specific sub-signal explaining the direction
   of the gap (suppressing or inflating).

The diagnosis hint checks `safety.multiplier` first: if it's < 1.0 it
attributes the gap to safety, because a non-1.0 multiplier dominates
everything else and attributing to a feature would be misleading.

**Exit code 0** if `ρ > 0.6`, else 1, so this script is usable as a
gate in CI or before kicking off training.

Run:
```
python scripts/validate_reward.py
echo "Exit: $?"
```

### 4.3 What happens if the gate fails

Documented in conversation but worth restating here. Iterate in this
order:

1. **Read the top-5 disagreements**. They cluster around one failure
   mode 90% of the time. Fix that sub-signal.
2. **Check the per-feature ρ table** for an anti-aligned feature
   (ρ < -0.2). That's a feature whose definition is fighting the
   author's taste. Investigate before retuning weights.
3. **Retune `weights.yaml`** to shift global weight toward the
   most-aligned feature.
4. **Only then** consider adding a new feature for a missing dimension.

The validator's "likely culprit" hint is a starting point, not a
verdict — read the full breakdown.

---

## 5. RL pipeline (stubs)

Everything in this section is **not yet built**. The interfaces are
sketched so the scaffolding is recognizable, but every entrypoint
currently raises `NotImplementedError`. They will be filled in only
after calibration passes.

### 5.1 `env/linkedin_env.py` — Gymnasium env

Planned shape:
- **Observation**: a topic prompt sampled from `data/topics.yaml`.
- **Action**: a generated post (string) — sampled from the policy.
- **Reward**: `score_post(action)["total"]`, a scalar in [0, 1].
- **Episode**: single-step (one prompt → one post → one reward).

The env is a thin shim. The interesting machinery (generation, gradient
updates) lives in `agent/train_grpo.py`. The env exists to give TRL's
GRPO trainer a uniform interface and to make the reward function
swappable for downstream experiments.

### 5.2 `agent/baseline.py` — untrained reference generator

Generates posts from the base Qwen3-1.7B with no fine-tuning. The
"before" half of the eventual win-rate evaluation. Will be invoked by
`scripts/generate_baseline.py` to produce ~200 baseline posts written
to `results/baseline/<run_id>.jsonl` for human labeling and reward
validation.

### 5.3 `agent/train_grpo.py` — GRPO trainer

Wires Qwen3-1.7B + LoRA (via `peft`) + `LinkedInEnv` + `score_post` into
TRL's GRPO trainer. Outputs trained adapter weights into `training/`.

### 5.4 `results/` — evaluation artifacts

Empty for now. Will hold:
- `results/baseline/<run_id>.jsonl` — pre-training generations
- `results/trained/<run_id>.jsonl` — post-training generations
- Win-rate evaluation outputs for the README "Results" section

---

## 6. Configuration and data

| File | Purpose |
|---   |--- |
| `reward/weights.yaml`              | Global per-feature weights. Tunable. |
| `data/topics.yaml`                 | ~20 topic prompts seeding the agent. |
| `data/good_posts.yaml`             | Hand-curated "good" reference posts. |
| `data/bad_posts.yaml`              | Hand-curated "bad" reference posts. |
| `data/safety_banned.example.yaml`  | Public-safe minimal banned-phrase list. |
| `data/safety_banned.yaml`          | [gitignored] real production list. |
| `notes/cliched_closers.md`         | Source-of-truth for `style.cliched_closer` patterns; loaded at import time. |

Three of these are also _data, not code_:

- `weights.yaml`: tuned by calibration evidence, not by editing source.
- `cliched_closers.md`: patterns added in markdown, reload on restart.
- `safety_banned.yaml`: production config kept out of source control.

---

## 7. Tests

40 tests, all passing. One file per feature plus a composition test:

```
tests/test_hook.py        — first-line scoring, including filler/abstract/specificity edge cases
tests/test_structure.py   — paragraph rhythm, wall-of-text, payoff shape, length gate
tests/test_style.py       — cliched closer (including R1 invariant test), AI tells, voice, noise
tests/test_safety.py      — banned/naming-and-shaming/private-info, multiplier curve
tests/test_reward.py      — composition: empty input, safety multiplier interactions, full schema
```

Notable: `tests/test_style.py::test_r1_cliched_closer_costs_more_than_clean_closer`
asserts the closed-form invariant from `notes/reward-hack-risks.md` R1
— same body, two closers, the cliched variant must score strictly lower.
The invariant assertion replaced an earlier test that hard-coded a
threshold of 0.26 — the kind of numerical-threshold test that rots the
moment weights are retuned. Tests that encode the design contract are
durable; tests that encode arbitrary numerical thresholds are not.

Run:
```
pytest -q
```

---

## 8. Documented reward-hack risks

The project tracks anticipated reward-hack vectors in
`notes/reward-hack-risks.md` _before_ training, and discovered hacks in
`LEARNINGS.md` _after_ training (with evidence). Currently:

| ID | Hack                                     | Status                       | Catches |
|--- |---                                       |---                           |--- |
| R1 | Generic "thoughts?" closer earning structure credit | **closed**            | `style.cliched_closer` at W_CLOSER=0.35; tested. |
| R2 | Fabricated-but-plausible specifics       | **acknowledged, open by design** | `safety.naming_and_shaming` catches the named-person subset; pure numerical fabrication passes through. Watching for it during training. |

A third bucket — _discovered-during-training hacks_ — will land in
`LEARNINGS.md` with evidence as the policy surfaces them. That bucket is
what makes the portfolio writeup interesting.

---

## 9. Conventions and invariants

A few load-bearing rules that aren't obvious from any single file:

1. **Reward function purity.** No I/O at scoring time. `score_post(text)`
   reads the same files (`weights.yaml`, `cliched_closers.md`,
   `safety_banned*.yaml`) at module import and never again. Hot-reloading
   patterns means restarting the process.

2. **Sub-signal exposure.** Every feature returns a dict with all its
   sub-signals, not just a total. Reward-hack debugging requires
   inspectable internals.

3. **Normalization discipline.** Pattern-matching features call
   `normalize_text` on input; the only intentional exception is em-dash
   density on the original text in `style.ai_tells`.

4. **Safety is a multiplier, not a term.** Don't fold safety into the
   additive sum. Don't soften the cliff in `_multiplier_from_hits`.

5. **Calibration before training.** No env, no agent, no GRPO until
   `validate_reward.py` exits 0. The reward function is the most
   important file in the project — bad reward = bad model no matter
   how fancy the RL.

6. **R1 invariant in `style.py`**: re-verify after any retune of
   `weights.yaml`. The invariant is asserted in tests, but the ratio
   is set by global weights, so the test will catch a violation only
   if the test fixture covers the post shape that triggers it. Reading
   the invariant comment in `style.py` after a weight change is
   cheaper than debugging a regression.

---

## 10. Status snapshot

| Area | Status |
|---   |--- |
| Reward function (4 features + composition)        | **Complete**, 40 tests passing |
| Calibration scripts (collect + validate)          | **Complete** |
| Human labeling (30 posts)                         | _In progress_ (manual) |
| Calibration gate (Spearman ρ > 0.6 on real data)  | _Pending_ |
| Baseline generation (`scripts/generate_baseline.py`) | _Stub_ |
| Gymnasium env (`env/linkedin_env.py`)             | _Stub_ |
| GRPO training (`agent/train_grpo.py`)             | _Stub_ |
| Win-rate evaluation                               | _Not started_ |

The project plan is sequential through the gate and parallelizable after.
The next blocking step is the calibration gate; everything downstream of
it is gated on its outcome.
