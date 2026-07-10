#!/usr/bin/env python3
"""Reviewer-requested per-model significance tests for Experiment 1.

The analysis intentionally contains only:
1. Markov vs. non-Markov exact-identity accuracy in type-2 games (independent
   two-proportion z-test, Fisher's exact test, and Newcombe CI).
2. Degradation of Markov exact-identity accuracy with context length
   (Cochran-Armitage and logistic-regression trend tests, plus a 100-vs-1000
   Fisher exact test and Newcombe CI).

Outputs are written as CSV files plus a short Markdown report suitable for
the paper/rebuttal.  No Experiment 2 or Experiment 3 data are read.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import fisher_exact, norm


MODELS = ("deepseek-chat", "deepseek-reasoner", "gpt-5", "gpt-5-mini")
CONTEXTS = (100, 200, 500, 1000)
MARKOV_IDS = {"X", "Y", "Z"}


def load_type2(root: Path, model: str, context: int) -> list[dict[str, int]]:
    detail = root / model / str(context) / "evaluation_detail.json"
    with detail.open(encoding="utf-8") as handle:
        rows = json.load(handle)["rows"]

    observations: list[dict[str, int]] = []
    for row in rows:
        gt = (row["gt_player1"], row["gt_player2"])
        pred = (row["pred_player1"], row["pred_player2"])
        markov_positions = [i for i, identity in enumerate(gt) if identity in MARKOV_IDS]
        if len(markov_positions) != 1:
            continue
        markov_i = markov_positions[0]
        nonmarkov_i = 1 - markov_i
        observations.append(
            {
                "markov_correct": int(pred[markov_i] == gt[markov_i]),
                "nonmarkov_correct": int(pred[nonmarkov_i] == gt[nonmarkov_i]),
            }
        )
    if len(observations) != 50:
        raise ValueError(f"Expected 50 type-2 games for {model}/{context}; got {len(observations)}")
    return observations


def newcombe_diff_ci(x1: int, n1: int, x0: int, n0: int) -> tuple[float, float]:
    """Newcombe 95% CI for the difference of two independent proportions."""
    z = norm.ppf(0.975)

    def wilson(x: int, n: int) -> tuple[float, float]:
        p = x / n
        denom = 1 + z * z / n
        center = (p + z * z / (2 * n)) / denom
        half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
        return center - half, center + half

    p1, p0 = x1 / n1, x0 / n0
    lo1, hi1 = wilson(x1, n1)
    lo0, hi0 = wilson(x0, n0)
    difference = p1 - p0
    # Newcombe's method 10: combine the two Wilson score intervals without
    # imposing equal variances or using a potentially invalid Wald interval.
    lower = difference - math.sqrt((p1 - lo1) ** 2 + (hi0 - p0) ** 2)
    upper = difference + math.sqrt((hi1 - p1) ** 2 + (p0 - lo0) ** 2)
    return lower, upper


def cochran_armitage(successes: list[int], totals: list[int], scores: list[float]) -> tuple[float, float]:
    """Two-sided Cochran-Armitage test for a monotone binomial trend."""
    total_success = sum(successes)
    total_n = sum(totals)
    pooled = total_success / total_n
    weighted_score = sum(n * score for n, score in zip(totals, scores)) / total_n
    numerator = sum(score * (x - n * pooled) for x, n, score in zip(successes, totals, scores))
    denominator = math.sqrt(
        pooled * (1 - pooled)
        * sum(n * (score - weighted_score) ** 2 for n, score in zip(totals, scores))
    )
    z = numerator / denominator
    return z, 2 * norm.sf(abs(z))


def two_proportion_z(x1: int, n1: int, x0: int, n0: int) -> tuple[float, float]:
    """Two-sided pooled two-proportion z-test for independent samples."""
    pooled = (x1 + x0) / (n1 + n0)
    denominator = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n0))
    z = (x1 / n1 - x0 / n0) / denominator
    return z, 2 * norm.sf(abs(z))


def logistic_trend(successes: list[int], totals: list[int], scores: list[float]) -> tuple[float, float, float]:
    """Grouped-binomial logistic regression; return slope, Wald z, and p."""
    def objective(beta: list[float]) -> float:
        eta = beta[0] + beta[1] * scores_array
        # Stable negative grouped-binomial log likelihood (constant omitted).
        return float(sum(totals_array * np.logaddexp(0, eta) - successes_array * eta))

    successes_array = np.asarray(successes, dtype=float)
    totals_array = np.asarray(totals, dtype=float)
    scores_array = np.asarray(scores, dtype=float)
    initial_p = successes_array.sum() / totals_array.sum()
    initial = np.array([math.log(initial_p / (1 - initial_p)), 0.0])
    fit = minimize(objective, initial, method="BFGS")
    if not fit.success:
        raise RuntimeError(f"Logistic regression failed: {fit.message}")
    probabilities = expit(fit.x[0] + fit.x[1] * scores_array)
    design = np.column_stack((np.ones(len(scores_array)), scores_array))
    information = design.T @ ((totals_array * probabilities * (1 - probabilities))[:, None] * design)
    slope_se = math.sqrt(float(np.linalg.inv(information)[1, 1]))
    slope = float(fit.x[1])
    z = slope / slope_se
    return slope, z, 2 * norm.sf(abs(z))


def holm_adjust(pvalues: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, returned in original order."""
    order = sorted(range(len(pvalues)), key=pvalues.__getitem__)
    adjusted = [0.0] * len(pvalues)
    running_max = 0.0
    for rank, index in enumerate(order):
        running_max = max(running_max, (len(pvalues) - rank) * pvalues[index])
        adjusted[index] = min(1.0, running_max)
    return adjusted


def fmt_p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def run(parsed_root: Path) -> tuple[list[dict], list[dict]]:
    gap_rows: list[dict] = []
    degradation_rows: list[dict] = []

    for model in MODELS:
        by_context = {context: load_type2(parsed_root, model, context) for context in CONTEXTS}
        pooled = [obs for context in CONTEXTS for obs in by_context[context]]

        markov_success = sum(o["markov_correct"] for o in pooled)
        nonmarkov_success = sum(o["nonmarkov_correct"] for o in pooled)
        z, z_p = two_proportion_z(markov_success, len(pooled), nonmarkov_success, len(pooled))
        _, fisher_p = fisher_exact(
            [[markov_success, len(pooled) - markov_success],
             [nonmarkov_success, len(pooled) - nonmarkov_success]],
            alternative="two-sided",
        )
        difference = markov_success / len(pooled) - nonmarkov_success / len(pooled)
        ci_low, ci_high = newcombe_diff_ci(
            markov_success, len(pooled), nonmarkov_success, len(pooled)
        )
        gap_rows.append(
            {
                "model": model,
                "games": len(pooled),
                "markov_correct": markov_success,
                "markov_accuracy": markov_success / len(pooled),
                "nonmarkov_correct": nonmarkov_success,
                "nonmarkov_accuracy": nonmarkov_success / len(pooled),
                "markov_minus_nonmarkov": difference,
                "difference_95ci_low": ci_low,
                "difference_95ci_high": ci_high,
                "two_proportion_z": z,
                "two_proportion_p": z_p,
                "fisher_p": fisher_p,
            }
        )

        successes = [sum(o["markov_correct"] for o in by_context[c]) for c in CONTEXTS]
        totals = [len(by_context[c]) for c in CONTEXTS]
        # log2 scores reflect multiplicative context growth and make the test
        # correspond to a degradation slope per context doubling.
        scores = [math.log2(c / CONTEXTS[0]) for c in CONTEXTS]
        trend_z, trend_p = cochran_armitage(successes, totals, scores)
        logistic_slope, logistic_z, logistic_p = logistic_trend(successes, totals, scores)
        start_x, end_x = successes[0], successes[-1]
        _, fisher_p = fisher_exact(
            [[start_x, totals[0] - start_x], [end_x, totals[-1] - end_x]],
            alternative="two-sided",
        )
        diff = end_x / totals[-1] - start_x / totals[0]
        ci_low, ci_high = newcombe_diff_ci(end_x, totals[-1], start_x, totals[0])
        degradation_rows.append(
            {
                "model": model,
                **{f"correct_{c}": x for c, x in zip(CONTEXTS, successes)},
                **{f"accuracy_{c}": x / n for c, x, n in zip(CONTEXTS, successes, totals)},
                "trend_z": trend_z,
                "trend_p": trend_p,
                "logistic_slope": logistic_slope,
                "logistic_wald_z": logistic_z,
                "logistic_wald_p": logistic_p,
                "difference_1000_minus_100": diff,
                "difference_95ci_low": ci_low,
                "difference_95ci_high": ci_high,
                "fisher_100_vs_1000_p": fisher_p,
            }
        )
    # Each p-value family comprises the same test across the four models.
    for rows, raw_key, adjusted_key in (
        (gap_rows, "two_proportion_p", "two_proportion_p_holm"),
        (gap_rows, "fisher_p", "fisher_p_holm"),
        (degradation_rows, "trend_p", "trend_p_holm"),
        (degradation_rows, "logistic_wald_p", "logistic_wald_p_holm"),
        (degradation_rows, "fisher_100_vs_1000_p", "fisher_100_vs_1000_p_holm"),
    ):
        for row, adjusted_p in zip(rows, holm_adjust([r[raw_key] for r in rows])):
            row[adjusted_key] = adjusted_p
    return gap_rows, degradation_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, gap_rows: list[dict], degradation_rows: list[dict]) -> None:
    lines = [
        "# Experiment 1: reviewer-requested significance tests",
        "",
        "Exact identity accuracy is the outcome throughout. Treating the Markov and non-Markov observations as independent samples, their accuracies are compared using a two-sided pooled two-proportion z-test, with a two-sided Fisher exact test as a robustness check and a Newcombe 95% CI for Markov minus non-Markov accuracy. Context degradation is tested on the Markov player using a two-sided Cochran–Armitage trend test with log2(context/100) scores, with a grouped-binomial logistic-regression slope Wald test as a robustness check. The endpoint contrast (100 versus 1000 rounds) uses a two-sided Fisher exact test and a Newcombe 95% CI for accuracy(1000) minus accuracy(100). For every test, its four model-specific p-values are Holm-adjusted as one family.",
        "",
        "## Markov versus non-Markov",
        "",
        "| Model | Markov | Non-Markov | Difference (95% CI) | z p (Holm) | Fisher p (Holm) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in gap_rows:
        lines.append(
            f"| {r['model']} | {r['markov_accuracy']:.1%} | {r['nonmarkov_accuracy']:.1%} | "
            f"{r['markov_minus_nonmarkov']:+.1%} ({r['difference_95ci_low']:+.1%}, "
            f"{r['difference_95ci_high']:+.1%}) | {fmt_p(r['two_proportion_p'])} "
            f"({fmt_p(r['two_proportion_p_holm'])}) | {fmt_p(r['fisher_p'])} "
            f"({fmt_p(r['fisher_p_holm'])}) |"
        )
    lines.extend(
        [
            "",
            "## Context degradation of Markov identity accuracy",
            "",
            "| Model | 100 | 200 | 500 | 1000 | CA p (Holm) | Logistic slope; p (Holm) | 1000−100 (95% CI) | Fisher p (Holm) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for r in degradation_rows:
        lines.append(
            f"| {r['model']} | {r['accuracy_100']:.1%} | {r['accuracy_200']:.1%} | "
            f"{r['accuracy_500']:.1%} | {r['accuracy_1000']:.1%} | "
            f"{fmt_p(r['trend_p'])} ({fmt_p(r['trend_p_holm'])}) | "
            f"{r['logistic_slope']:+.3f}; {fmt_p(r['logistic_wald_p'])} "
            f"({fmt_p(r['logistic_wald_p_holm'])}) | {r['difference_1000_minus_100']:+.1%} "
            f"({r['difference_95ci_low']:+.1%}, {r['difference_95ci_high']:+.1%}) | "
            f"{fmt_p(r['fisher_100_vs_1000_p'])} ({fmt_p(r['fisher_100_vs_1000_p_holm'])}) |"
        )
    lines.extend(
        [
            "",
            "Notes: each p-value cell shows raw p (Holm-adjusted p). Logistic slopes are log-odds changes per context doubling. Exact counts, test statistics, and full-precision p-values are in the accompanying CSV files.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parsed-root",
        type=Path,
        default=Path("exp1(strategy)/parsed_output"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("exp1(strategy)/analysis_results/reviewer_tests"),
    )
    args = parser.parse_args()
    gap_rows, degradation_rows = run(args.parsed_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "markov_vs_nonmarkov.csv", gap_rows)
    write_csv(args.output_dir / "context_degradation.csv", degradation_rows)
    write_report(args.output_dir / "report.md", gap_rows, degradation_rows)
    print(f"Wrote reviewer tests to {args.output_dir}")


if __name__ == "__main__":
    main()
