# LEARNINGS

A running log of things this project taught me. Format for every entry:

> **what I thought → what I know now → evidence**

Keep entries dated and concrete. "Evidence" should point at a commit, a run id, a chart, or a
specific reward-hacking example — not a vibe. Update entries when later evidence revises them
rather than silently rewriting history.

---

### Banned-phrase list as production config, not source code

**What I thought**: a comprehensive slur/harassment list for `safety.py`
should live in the repo like every other reward configuration — checked in,
versioned, transparent.

**What I know now**: shipping a slur inventory in a public-portfolio repo
creates the exact harm vectors the inventory is meant to prevent. Curated
slur lists are a moderation primitive, not a portfolio artifact, and they
get extracted and reused for the opposite of the original purpose. The
right pattern is the one production moderation pipelines use: a public
fallback (`data/safety_banned.example.yaml`) with a deliberately minimal
public-safe set, plus a gitignored real list (`data/safety_banned.yaml`)
that the loader prefers when present and warns about when absent.

**Evidence**: `data/safety_banned.example.yaml` (5-10 universally-published
patterns); `data/safety_banned.yaml` in `.gitignore`;
`reward/features/safety.py::_load_banned_patterns` logs a warning when
falling back to the example. Decision rationale documented in the file
header so a future reviewer doesn't think the example IS the production
list.

---
