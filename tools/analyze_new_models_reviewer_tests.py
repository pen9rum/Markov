#!/usr/bin/env python3
"""Reviewer-style significance tests for the three newly added Experiment 1 models.

Input is the high/low selected summary produced by
tools/summarize_new_models_success.py:

    exp1(strategy)/reviewer/new_models_success_highlow_selected.csv

The tests mirror tools/analyze_exp1_reviewer_tests.py:
1. Markov vs. non-Markov exact-identity accuracy per model using a pooled
   two-proportion z-test, Fisher exact test, and Newcombe CI.
2. Context degradation of Markov exact-identity accuracy using a
   Cochran-Armitage trend test, grouped-binomial logistic-regression trend
   test, and 100-vs-1000 Fisher exact test.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import fisher_exact, norm


MODELS = ("gpt-4.1", "gemini", "qwen")
CONTEXTS = (100, 200, 500, 1000)


def newcombe_diff_ci(x1: int, n1: int, x0: int, n0: int) -> tuple[float, float]:
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
    lower = difference - math.sqrt((p1 - lo1) ** 2 + (hi0 - p0) ** 2)
    upper = difference + math.sqrt((hi1 - p1) ** 2 + (p0 - lo0) ** 2)
    return lower, upper


def cochran_armitage(successes: list[int], totals: list[int], scores: list[float]) -> tuple[float, float]:
    total_success = sum(successes)
    total_n = sum(totals)
    pooled = total_success / total_n
    weighted_score = sum(n * score for n, score in zip(totals, scores)) / total_n
    numerator = sum(score * (x - n * pooled) for x, n, score in zip(successes, totals, scores))
    denominator = math.sqrt(
        pooled
        * (1 - pooled)
        * sum(n * (score - weighted_score) ** 2 for n, score in zip(totals, scores))
    )
    z = numerator / denominator
    return z, 2 * norm.sf(abs(z))


def two_proportion_z(x1: int, n1: int, x0: int, n0: int) -> tuple[float, float]:
    pooled = (x1 + x0) / (n1 + n0)
    denominator = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n0))
    z = (x1 / n1 - x0 / n0) / denominator
    return z, 2 * norm.sf(abs(z))


def logistic_trend(successes: list[int], totals: list[int], scores: list[float]) -> tuple[float, float, float]:
    successes_array = np.asarray(successes, dtype=float)
    totals_array = np.asarray(totals, dtype=float)
    scores_array = np.asarray(scores, dtype=float)

    def objective(beta: list[float]) -> float:
        eta = beta[0] + beta[1] * scores_array
        return float(sum(totals_array * np.logaddexp(0, eta) - successes_array * eta))

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
    order = sorted(range(len(pvalues)), key=pvalues.__getitem__)
    adjusted = [0.0] * len(pvalues)
    running_max = 0.0
    for rank, index in enumerate(order):
        running_max = max(running_max, (len(pvalues) - rank) * pvalues[index])
        adjusted[index] = min(1.0, running_max)
    return adjusted


def fmt_p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def parse_fraction(text: str) -> tuple[int, int]:
    match = text.split("=", 1)[0].strip()
    x, n = match.split("/", 1)
    return int(x), int(n)


def load_summary(path: Path) -> dict[str, dict[int, dict[str, int]]]:
    by_model: dict[str, dict[int, dict[str, int]]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            model = row["model"]
            context = int(row["rounds"])
            markov_correct, markov_total = parse_fraction(row["markov_player_acc"])
            nonmarkov_correct, nonmarkov_total = parse_fraction(row["nonmarkov_player_acc"])
            by_model.setdefault(model, {})[context] = {
                "markov_correct": markov_correct,
                "markov_total": markov_total,
                "nonmarkov_correct": nonmarkov_correct,
                "nonmarkov_total": nonmarkov_total,
            }
    return by_model


def run(summary_csv: Path) -> tuple[list[dict], list[dict]]:
    data = load_summary(summary_csv)
    gap_rows: list[dict] = []
    degradation_rows: list[dict] = []

    for model in MODELS:
        by_context = data[model]
        markov_success = sum(by_context[c]["markov_correct"] for c in CONTEXTS)
        markov_total = sum(by_context[c]["markov_total"] for c in CONTEXTS)
        nonmarkov_success = sum(by_context[c]["nonmarkov_correct"] for c in CONTEXTS)
        nonmarkov_total = sum(by_context[c]["nonmarkov_total"] for c in CONTEXTS)

        z, z_p = two_proportion_z(markov_success, markov_total, nonmarkov_success, nonmarkov_total)
        _, fisher_p = fisher_exact(
            [
                [markov_success, markov_total - markov_success],
                [nonmarkov_success, nonmarkov_total - nonmarkov_success],
            ],
            alternative="two-sided",
        )
        difference = markov_success / markov_total - nonmarkov_success / nonmarkov_total
        ci_low, ci_high = newcombe_diff_ci(
            markov_success, markov_total, nonmarkov_success, nonmarkov_total
        )
        gap_rows.append(
            {
                "model": model,
                "markov_total": markov_total,
                "markov_correct": markov_success,
                "markov_accuracy": markov_success / markov_total,
                "nonmarkov_total": nonmarkov_total,
                "nonmarkov_correct": nonmarkov_success,
                "nonmarkov_accuracy": nonmarkov_success / nonmarkov_total,
                "markov_minus_nonmarkov": difference,
                "difference_95ci_low": ci_low,
                "difference_95ci_high": ci_high,
                "two_proportion_z": z,
                "two_proportion_p": z_p,
                "fisher_p": fisher_p,
            }
        )

        successes = [by_context[c]["markov_correct"] for c in CONTEXTS]
        totals = [by_context[c]["markov_total"] for c in CONTEXTS]
        scores = [math.log2(c / CONTEXTS[0]) for c in CONTEXTS]
        trend_z, trend_p = cochran_armitage(successes, totals, scores)
        logistic_slope, logistic_z, logistic_p = logistic_trend(successes, totals, scores)
        start_x, end_x = successes[0], successes[-1]
        _, endpoint_fisher_p = fisher_exact(
            [[start_x, totals[0] - start_x], [end_x, totals[-1] - end_x]],
            alternative="two-sided",
        )
        diff = end_x / totals[-1] - start_x / totals[0]
        end_ci_low, end_ci_high = newcombe_diff_ci(end_x, totals[-1], start_x, totals[0])
        degradation_rows.append(
            {
                "model": model,
                **{f"correct_{c}": x for c, x in zip(CONTEXTS, successes)},
                **{f"total_{c}": n for c, n in zip(CONTEXTS, totals)},
                **{f"accuracy_{c}": x / n for c, x, n in zip(CONTEXTS, successes, totals)},
                "trend_z": trend_z,
                "trend_p": trend_p,
                "logistic_slope": logistic_slope,
                "logistic_wald_z": logistic_z,
                "logistic_wald_p": logistic_p,
                "difference_1000_minus_100": diff,
                "difference_95ci_low": end_ci_low,
                "difference_95ci_high": end_ci_high,
                "fisher_100_vs_1000_p": endpoint_fisher_p,
            }
        )

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
        "# Experiment 1 new models: reviewer-style significance tests",
        "",
        "Input: `new_models_success_highlow_selected.csv`. qwen/200 uses the selected 50 Markov-player and 150 non-Markov-player observations.",
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
            "| Model | 100 | 200 | 500 | 1000 | CA p (Holm) | Logistic slope; p (Holm) | 1000-100 (95% CI) | Fisher p (Holm) |",
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
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("exp1(strategy)/reviewer/new_models_success_highlow_selected.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("exp1(strategy)/reviewer/new_models_significance_tests"),
    )
    args = parser.parse_args()
    gap_rows, degradation_rows = run(args.summary_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "markov_vs_nonmarkov.csv", gap_rows)
    write_csv(args.output_dir / "context_degradation.csv", degradation_rows)
    write_report(args.output_dir / "report.md", gap_rows, degradation_rows)
    print(args.output_dir)


if __name__ == "__main__":
    main()
