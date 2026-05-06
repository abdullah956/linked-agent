"""Generate ~200 baseline posts from the untrained model.

Samples topics from data/topics.yaml (with replacement), calls
agent.baseline.generate_baseline_post, and writes results to
results/baseline/<run_id>.jsonl for later human labeling and reward
validation.

Stub — implement after agent.baseline is functional.
"""

from __future__ import annotations


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
