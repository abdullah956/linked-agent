"""Validate the reward function against human ratings.

Loads data/labeled_posts.yaml (produced by collect_human_labels.py) and
reports:

  - Spearman + Pearson correlation between my_rating and reward_total.
    Both, because they answer different questions:
        Spearman → "do we agree on the RANKING?"  (rank-only, robust to scale)
        Pearson  → "do we agree on the LINEAR map?" (sensitive to scale + outliers)
  - An ASCII scatter plot (rendered with rich, no extra deps).
  - The 5 biggest disagreements, with full breakdown + a hint at the dominant
    sub-signal explaining the gap.
  - Per-feature Spearman of hook/structure/style vs my_rating, so we can see
    WHICH feature is most aligned with my taste vs misaligned.

Exit code is 0 if Spearman_total > 0.6, else 1, so this script can be wired
into CI or used as a script-level gate before training.

Scatter plot dependency choice: rich does not have a native scatter renderer,
and asciichartpy is a line-chart library (treats the second series as a curve
between sequential x's, not a point cloud). For 30-100 posts a small hand-
drawn ASCII grid is plenty and keeps the dep surface to what is already in
pyproject.toml. If we later want a richer plot, plotext is the upgrade path.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from scipy import stats


GATE_THRESHOLD = 0.6
N_DISAGREEMENTS = 5


def _load_labels(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"Labels file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise SystemExit(f"Labels file is not a mapping: {path}")
    rows: list[dict[str, Any]] = []
    for entry in raw.values():
        if not isinstance(entry, dict):
            continue
        if "my_rating" not in entry or "reward_total" not in entry:
            continue
        rows.append(entry)
    return rows


def _ascii_scatter(
    xs: list[float],
    ys: list[float],
    width: int = 50,
    height: int = 18,
    x_label: str = "my_rating (1-10)",
    y_label: str = "reward_total (0-1)",
) -> Text:
    """Tiny ASCII scatter: x in [1, 10], y in [0, 1]. Multiple points in the
    same cell render as digits 2..9 and '+' for >9 — keeps density visible."""
    grid = [[" " for _ in range(width)] for _ in range(height)]
    for x, y in zip(xs, ys):
        col = int(round((x - 1) / 9 * (width - 1)))
        row = int(round((1 - max(0.0, min(1.0, y))) * (height - 1)))
        col = max(0, min(width - 1, col))
        row = max(0, min(height - 1, row))
        cur = grid[row][col]
        if cur == " ":
            grid[row][col] = "."
        elif cur == ".":
            grid[row][col] = "2"
        elif cur.isdigit() and cur != "9":
            grid[row][col] = str(int(cur) + 1)
        elif cur == "9":
            grid[row][col] = "+"
        # else already '+'

    out = Text()
    out.append(f"{y_label}\n", style="dim")
    for r, row in enumerate(grid):
        # y-axis tick labels at top, middle, bottom
        if r == 0:
            tick = "1.0 |"
        elif r == height // 2:
            tick = "0.5 |"
        elif r == height - 1:
            tick = "0.0 |"
        else:
            tick = "    |"
        out.append(tick, style="dim")
        out.append("".join(row) + "\n")
    out.append("    +" + "-" * width + "\n", style="dim")
    out.append("     1" + " " * (width - 4) + "10\n", style="dim")
    out.append(f"     {x_label}\n", style="dim")
    return out


def _safe_spearman(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Returns (rho, p). When all of either series is constant, scipy returns
    nan with a warning — we surface 0.0 with p=1.0 so the table is readable."""
    if len(xs) < 3 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return 0.0, 1.0
    res = stats.spearmanr(xs, ys)
    rho = float(res.correlation) if hasattr(res, "correlation") else float(res.statistic)
    p = float(res.pvalue)
    if rho != rho:  # NaN
        return 0.0, 1.0
    return rho, p


def _diagnose_gap(entry: dict[str, Any]) -> str:
    """Heuristic: identify the feature most responsible for the gap between
    my_rating and reward_total, then point at the sub-signal in that feature
    that best explains the direction of the gap.

    Safety is checked first because a non-1.0 multiplier dominates everything
    else — a clean post with safety.multiplier=0.3 has reward = 0.3 * additive
    regardless of what hook/structure/style say, so attributing the gap to
    one of those features would be misleading.
    """
    breakdown = entry.get("reward_breakdown", {})
    my_norm = (entry["my_rating"] - 1) / 9
    reward_total = float(entry.get("reward_total", 0.0))

    # Safety check first — if the multiplier is suppressing the score, that
    # is almost certainly the load-bearing cause of the gap.
    safety = breakdown.get("safety", {})
    if isinstance(safety, dict):
        mult = float(safety.get("multiplier", 1.0))
        if mult < 1.0:
            hits = safety.get("total_hits", "?")
            return (
                f"safety multiplier ×{mult:.2f} (hits={hits}) — overriding all "
                f"other features; |Δ|={abs(my_norm - reward_total):.2f}"
            )

    deltas: list[tuple[float, str]] = []
    for feat in ("hook", "structure", "style"):
        sub = breakdown.get(feat, {})
        if not isinstance(sub, dict) or "total" not in sub:
            continue
        deltas.append((abs(my_norm - float(sub["total"])), feat))
    if not deltas:
        return "no breakdown available"
    deltas.sort(reverse=True)
    feat = deltas[0][1]
    sub = breakdown.get(feat, {})
    feat_total = float(sub.get("total", 0.0))
    # If human rated HIGH but feature is LOW, the lowest sub-signal is the
    # likely suppressor. If human rated LOW but feature is HIGH, the highest
    # sub-signal is the likely false-positive.
    direction = "suppressing" if my_norm > feat_total else "inflating"
    candidates = [
        (k, float(v)) for k, v in sub.items()
        if k != "total" and isinstance(v, (int, float))
    ]
    if not candidates:
        return f"{feat} (Δ={deltas[0][0]:.2f})"
    candidates.sort(key=lambda kv: kv[1], reverse=(direction == "inflating"))
    sig, val = candidates[0]
    return f"{feat} (Δ={deltas[0][0]:.2f}); {direction} sub-signal: {sig}={val:.2f}"


def _format_breakdown(breakdown: dict[str, Any]) -> str:
    parts: list[str] = []
    for feat in ("hook", "structure", "style"):
        sub = breakdown.get(feat, {})
        if isinstance(sub, dict) and "total" in sub:
            parts.append(f"{feat}={float(sub['total']):.2f}")
    safety = breakdown.get("safety", {})
    if isinstance(safety, dict) and "multiplier" in safety:
        parts.append(f"safety×{float(safety['multiplier']):.2f}")
    return "  ".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path,
                        default=Path("data/labeled_posts.yaml"))
    parser.add_argument("--gate", type=float, default=GATE_THRESHOLD,
                        help="Spearman threshold for exit code 0.")
    args = parser.parse_args()

    console = Console()
    rows = _load_labels(args.labels)
    if len(rows) < 3:
        console.print(f"[red]Need at least 3 labeled posts; have {len(rows)}.[/red]")
        return 2

    my_ratings = [float(r["my_rating"]) for r in rows]
    reward_totals = [float(r["reward_total"]) for r in rows]

    rho, p_s = _safe_spearman(my_ratings, reward_totals)
    pearson = stats.pearsonr(my_ratings, reward_totals)
    pear_r, pear_p = float(pearson[0]), float(pearson[1])
    if pear_r != pear_r:  # NaN
        pear_r, pear_p = 0.0, 1.0

    console.rule(f"[bold]Reward calibration  ·  N={len(rows)}[/bold]")
    summary = Table.grid(padding=(0, 2))
    summary.add_row("Spearman ρ (rank agreement)", f"{rho:+.3f}", f"p={p_s:.3f}")
    summary.add_row("Pearson r  (linear agreement)", f"{pear_r:+.3f}", f"p={pear_p:.3f}")
    summary.add_row("Gate threshold (Spearman)", f"{args.gate:+.3f}", "")
    console.print(summary)

    pass_gate = rho > args.gate
    verdict = "[green]PASS[/green]" if pass_gate else "[red]FAIL[/red]"
    console.print(f"Gate: {verdict}\n")

    # Scatter
    console.print(_ascii_scatter(my_ratings, reward_totals))

    # Per-feature Spearman
    console.rule("[bold]Per-feature Spearman vs my_rating[/bold]")
    feat_table = Table(show_header=True, header_style="bold")
    feat_table.add_column("Feature")
    feat_table.add_column("Spearman ρ", justify="right")
    feat_table.add_column("p", justify="right")
    feat_table.add_column("Reading", style="dim")
    for feat in ("hook", "structure", "style"):
        feat_vals = [
            float(r["reward_breakdown"].get(feat, {}).get("total", 0.0))
            for r in rows
        ]
        f_rho, f_p = _safe_spearman(my_ratings, feat_vals)
        if f_rho > 0.5:
            reading = "aligned with my taste"
        elif f_rho > 0.2:
            reading = "weakly aligned"
        elif f_rho > -0.2:
            reading = "no signal"
        else:
            reading = "ANTI-aligned — investigate"
        feat_table.add_row(feat, f"{f_rho:+.3f}", f"{f_p:.3f}", reading)
    console.print(feat_table)

    # Biggest disagreements
    console.rule(f"[bold]Top {N_DISAGREEMENTS} disagreements[/bold]")
    enriched = []
    for r in rows:
        my_norm = (r["my_rating"] - 1) / 9
        gap = abs(my_norm - r["reward_total"])
        enriched.append((gap, r))
    enriched.sort(key=lambda kv: kv[0], reverse=True)

    for gap, r in enriched[:N_DISAGREEMENTS]:
        my_norm = (r["my_rating"] - 1) / 9
        header = (
            f"id={r.get('id', '?')}  |Δ|={gap:.2f}  "
            f"my={r['my_rating']}/10 ({my_norm:.2f})  "
            f"reward={r['reward_total']:.2f}"
        )
        body = Text()
        body.append(r["text"] + "\n\n")
        body.append(f"My reason: {r.get('my_reason', '—')}\n", style="cyan")
        body.append(f"Breakdown: {_format_breakdown(r['reward_breakdown'])}\n", style="dim")
        body.append(f"Likely culprit: {_diagnose_gap(r)}", style="yellow")
        console.print(Panel(body, title=header, border_style="red" if gap > 0.4 else "yellow"))

    return 0 if pass_gate else 1


if __name__ == "__main__":
    sys.exit(main())
