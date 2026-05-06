# linkedin-rl-agent

A custom RL environment + GRPO fine-tune of **Qwen3-1.7B** that learns to write good LinkedIn posts.

## Overview

Why this project exists, what it does, and what makes it interesting.

> **Core principle.** The reward function is the most important file. Bad reward = bad model
> no matter how fancy the RL. We build and validate the reward against human judgment FIRST,
> then layer RL on top.

## Reward Design

How `score_post(text) -> dict` is composed: hook, structure, style, safety. Weights live in
`reward/weights.yaml`. Document each feature, the signal it tries to capture, and how it was
calibrated against human ratings.

## Environment

Gymnasium env wrapping the model + reward fn. Action = generated post; observation = topic
prompt; reward = `score_post(text)["total"]`.

## Training

GRPO setup (TRL), base model, hyperparams, hardware, run command.

## Results

Baseline vs trained, eval methodology, win rate, sample posts.

## Reward Hacking Discovered

Document every degenerate behavior the policy found and how the reward was patched in
response. This section is a feature, not a footnote.

## Limitations

What this model is not, scope of training data, known failure modes.

## Reproduce

Setup, env vars, commands to regenerate baseline + run training + eval.
