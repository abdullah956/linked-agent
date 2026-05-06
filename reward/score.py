"""Top-level reward composition.

`score_post(text)` returns a dict with per-feature scores and a weighted total.
This is the most important file in the project — keep it simple, transparent,
and validated against human ratings before any RL training is run.

Composition shape:
    additive = (W_HOOK * hook["total"]
              + W_STRUCTURE * structure["total"]
              + W_STYLE * style["total"])
    total = additive * safety["multiplier"]

Empty-input guard: this layer short-circuits on empty/whitespace-only input.
Defense in depth — the additive features already handle empty inputs, but
relying on multiplicative behavior to handle a case the additive features
should have caught is brittle. Catch it explicitly here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from reward.features.hook import score_hook
from reward.features.safety import score_safety
from reward.features.structure import score_structure
from reward.features.style import score_style


_WEIGHTS_FILE = Path(__file__).resolve().parent / "weights.yaml"


def _load_weights() -> dict[str, float]:
    data = yaml.safe_load(_WEIGHTS_FILE.read_text(encoding="utf-8")) or {}
    features = data.get("features", {})
    return {
        "hook": float(features.get("hook", 0.30)),
        "structure": float(features.get("structure", 0.20)),
        "style": float(features.get("style", 0.50)),
    }


_WEIGHTS = _load_weights()


def score_post(text: str) -> dict[str, Any]:
    """Score a LinkedIn post.

    Returns a dict with the four sub-feature dicts plus the composed total.
    Schema:
        {
            "hook":      <hook score dict>,
            "structure": <structure score dict>,
            "style":     <style score dict>,
            "safety":    <safety score dict (with multiplier)>,
            "additive":  float in [0, 1],
            "total":     float in [0, 1],
        }
    """
    if not text or not text.strip():
        return {
            "hook": {"total": 0.0},
            "structure": {"total": 0.0},
            "style": {"total": 0.0},
            "safety": {"multiplier": 1.0, "total_hits": 0},
            "additive": 0.0,
            "total": 0.0,
        }

    hook = score_hook(text)
    structure = score_structure(text)
    style = score_style(text)
    safety = score_safety(text)

    additive = (
        _WEIGHTS["hook"] * hook["total"]
        + _WEIGHTS["structure"] * structure["total"]
        + _WEIGHTS["style"] * style["total"]
    )
    total = additive * safety["multiplier"]

    return {
        "hook": hook,
        "structure": structure,
        "style": style,
        "safety": safety,
        "additive": additive,
        "total": total,
    }
