"""Baseline agent: untrained Qwen3-1.7B used to generate reference posts.

This is the "before" side of the comparison. No RL, no fine-tuning — just the
base model conditioned on a topic prompt. Stub for now.
"""

from __future__ import annotations


def generate_baseline_post(topic_prompt: str) -> str:
    """Generate a single post from the base model. Stub."""
    raise NotImplementedError
