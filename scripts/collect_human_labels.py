"""CLI to collect human ratings on a corpus of LinkedIn posts.

Reads a text file of posts separated by lines containing only "---", prompts
the user for a 1-10 rating + one-line reason for each, and writes a YAML file
mapping each post (keyed by sha256 hash of its text) to:

    - the post text
    - my_rating, my_reason
    - reward_total + full reward breakdown (per-feature dict from score_post)

The reward is computed AFTER the rating is recorded, so the human judgement is
not biased by seeing the score. Output is flushed to disk after every rating
so a crash mid-session loses nothing.

Used as the input to validate_reward.py, which checks Spearman correlation
between my_rating and reward_total. Spearman > 0.6 is the gate for moving on
to RL training.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.panel import Panel

from reward.score import score_post


def _post_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


def _split_posts(raw: str) -> list[str]:
    """Split on lines that are exactly '---' (after stripping)."""
    chunks: list[list[str]] = [[]]
    for line in raw.splitlines():
        if line.strip() == "---":
            chunks.append([])
        else:
            chunks[-1].append(line)
    posts = ["\n".join(c).strip() for c in chunks]
    return [p for p in posts if p]


def _post_stats(text: str) -> tuple[int, int, int]:
    chars = len(text)
    words = len(text.split())
    paragraphs = sum(1 for p in text.split("\n\n") if p.strip())
    return chars, words, paragraphs


def _load_existing(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        return {}
    return data


def _save(path: Path, data: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=100),
        encoding="utf-8",
    )


def _prompt_rating(console: Console) -> str:
    while True:
        raw = console.input("[bold]Rate 1-10 (or 's' to skip, 'q' to quit and save): [/bold]").strip().lower()
        if raw in {"s", "q"}:
            return raw
        try:
            n = int(raw)
        except ValueError:
            console.print("[red]Enter an integer 1-10, 's', or 'q'.[/red]")
            continue
        if 1 <= n <= 10:
            return str(n)
        console.print("[red]Out of range. 1-10 only.[/red]")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="Text file of posts separated by '---' lines.")
    parser.add_argument("--output", type=Path,
                        default=Path("data/labeled_posts.yaml"),
                        help="YAML file to write/append labels to.")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Skip posts whose hash already appears in --output.")
    args = parser.parse_args()

    console = Console()

    if not args.input.exists():
        console.print(f"[red]Input file not found: {args.input}[/red]")
        return 2

    raw = args.input.read_text(encoding="utf-8")
    posts = _split_posts(raw)
    if not posts:
        console.print("[red]No posts found. Separate posts with a line containing only '---'.[/red]")
        return 2

    existing = _load_existing(args.output) if args.resume else {}
    if args.resume and existing:
        already = sum(1 for p in posts if _post_id(p) in existing)
        if already:
            console.print(f"[dim]Skipping {already} already-labeled posts.[/dim]")

    labels: dict[str, dict[str, Any]] = dict(existing)
    rated = 0
    skipped = 0
    quit_early = False

    todo = [p for p in posts if not (args.resume and _post_id(p) in existing)]
    total = len(todo)

    for i, post in enumerate(todo, start=1):
        pid = _post_id(post)
        chars, words, paragraphs = _post_stats(post)

        console.rule(f"[bold cyan]Post {i} of {total}[/bold cyan]  id={pid}")
        console.print(Panel(post, border_style="cyan"))
        console.print(
            f"[dim]{chars} chars · {words} words · {paragraphs} paragraph(s)[/dim]"
        )

        choice = _prompt_rating(console)
        if choice == "q":
            quit_early = True
            break
        if choice == "s":
            skipped += 1
            console.print("[yellow]Skipped.[/yellow]")
            continue

        rating = int(choice)
        reason = console.input("[bold]Why? (one line): [/bold]").strip()

        # Score AFTER rating is recorded — never show the model's score before
        # the human commits to a number.
        breakdown = score_post(post)
        reward_total = float(breakdown["total"])

        labels[pid] = {
            "id": pid,
            "text": post,
            "my_rating": rating,
            "my_reason": reason,
            "reward_total": reward_total,
            "reward_breakdown": breakdown,
        }
        _save(args.output, labels)
        rated += 1
        console.print(f"[green]Saved.[/green] [dim]({len(labels)} total in {args.output})[/dim]")

    # Summary.
    console.rule("[bold]Session summary[/bold]")
    console.print(f"Rated this session: {rated}")
    console.print(f"Skipped this session: {skipped}")
    if quit_early:
        console.print("[yellow]Quit early — progress saved.[/yellow]")
    if labels:
        ratings = [v["my_rating"] for v in labels.values() if "my_rating" in v]
        rewards = [v["reward_total"] for v in labels.values() if "reward_total" in v]
        if ratings:
            console.print(f"Mean rating (all sessions): {sum(ratings) / len(ratings):.2f}")
        if rewards:
            console.print(f"Mean reward (all sessions): {sum(rewards) / len(rewards):.3f}")
        console.print(f"Total labeled: {len(labels)} → {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
