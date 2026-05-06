# Cliched Closer Patterns

The R1 mitigator. Each line below is a regex pattern (case-insensitive,
anchored or not as noted) that, when matched against the closer of a post,
zeroes out `cliched_closer_penalty` in `style.py`.

This file is **data, not code** — edit freely, then `style.py` will load it
on import. Patterns live here so we can iterate without touching the module.

## Match scope

- Multi-paragraph posts: match against the last paragraph (after final blank-line break).
- Single-paragraph posts: match against `text[-200:]` (last 200 chars).
- All matches case-insensitive.
- A pattern hits if the regex finds a match *anywhere in the closer scope*
  (we don't require full-string match, because cliches often appear at the
  end of a real sentence: "...so let me know — thoughts?").
- Input is normalized via `reward/features/_text_utils.normalize_text` before
  matching: curly quotes/apostrophes and en/em-dashes are mapped to ASCII so
  patterns don't silently bypass on iOS-autocorrected or Word-pasted text.
  Em-dash *density* is measured separately in `ai_tells` from the original,
  un-normalized text — that's a real signal, not a normalization target.

## Three categories

### A. Bare engagement-bait questions

These are short standalone closer paragraphs that are *only* a question
designed to farm comments. The post leads up to nothing and ends with a
prompt for engagement.

```
\bthoughts\??\s*$
\bagree\??\s*$
\bdisagree\??\s*$
\bwhat do you think\??\s*$
\bwhat are your thoughts\??\s*$
\bwhat'?s your take\??\s*$
\bam i (the only one|wrong|missing something)\??\s*$
\bwho else\b.*\?\s*$
\banyone else\b.*\?\s*$
\bdid i miss anything\??\s*$
\bright\??\s*$
\bany (other )?thoughts\??\s*$
\bwhat would you (do|add|change)\??\s*$
\bwhat'?s been your experience\b.*\?\s*$
```

### B. CTA stock phrases

Direct calls-to-action that are pure engagement-farming behavior, the
"like-and-subscribe" of LinkedIn.

```
\bcomment below\b
\bdrop a (?:🔥|🙌|👏|💯|comment|line|note|\w+) (?:if|below)\b
\btag someone who\b
\bshare (?:this )?if you (?:agree|relate)\b
\brepost if (?:you agree|this resonates)\b
\blike and share\b
\blike if you\b
\bfollow (?:me )?for more\b
\bdouble[- ]tap if\b
\bhit (?:the )?(?:like|follow) (?:button )?if\b
\bsave this (?:post )?for later\b
\bsmash that (?:like|follow)\b
```

### C. Cliched aphorism endings

The motivational-poster closing line. Reads as profound, says nothing.

```
\bthe choice is yours\b
\byou'?ve got this\b
\band the rest is history\b
\bthat'?s the (?:secret|key|truth|takeaway)\b
\bthe rest will follow\b
\bthe sky'?s the limit\b
\btrust the process\b
\bonwards and upwards\b
\bonward and upward\b
\bnever stop (?:learning|growing|believing|dreaming)\b
\bbe the change\b
(?:^|[.!?]\s|\n\s*)(?:just )?do the work\b\.?\s*$
\bkeep going\b\.?\s*$
\bstay (?:hungry|humble|focused|curious)\b\.?\s*$
\blet'?s (?:go|do this|build|change the world)\b\.?\s*$
\bthat'?s (?:all|it) for today\b\.?\s*$
\bthat'?s a wrap\b\.?\s*$
```

### D. Fake-vulnerability closer-introducers

These don't *end* a closer — they *open* it. Pattern shape: a stock
"I'll be honest/vulnerable/real" lead-in that signals the closer paragraph
is about to dispense Wisdom. Functionally part of the closer.

```
\bi'?ll be (?:vulnerable|honest|real|raw|transparent) (?:here|with you|for a (?:moment|second))\b
\b(?:can i be|let me be) (?:vulnerable|honest|real) (?:here|with you|for a (?:moment|second))\b
\bi'?m going to be (?:vulnerable|honest|real) (?:here|with you)\b
\b(?:real talk|honest moment|truth bomb)\s*[:.]\s*$
```

## Notes on regex choices

- `\s*$` at the end of bare-question patterns ties them to the actual end of
  the closer scope — we want "thoughts?" as a closer, not "any thoughts?
  Here's mine..." in the middle of a paragraph.
- `\b` word boundaries on phrase patterns prevent partial-word false positives.
- Apostrophes use `'?` to handle both straight and missing apostrophes.
  Curly apostrophes (`'`) are NOT handled here — that's a known issue
  flagged on the hook module too. Fix at one seam later.

## Pattern count

Currently 44 patterns across A (14), B (12), C (14), D (4). User decision
during edit: keep all — extra pattern is cheap, missing pattern is a
reward hack. Don't trim for budget.
