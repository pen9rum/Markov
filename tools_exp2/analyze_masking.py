#!/usr/bin/env python3
"""Measure whether incorrect sequential mechanisms mask behind close marginals.

This analysis reads existing Experiment 2 generation JSON files only. For each
Markov player, it compares the marginal action distribution in the generated
trajectory with that player's target distribution in the supplied context.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


MARKOV_IDS = {"X", "Y", "Z"}
ACTIONS = ("Rock", "Paper", "Scissors")
DEFAULT_TOLERANCES = (0.02, 0.05, 0.10)
MODEL_ORDER = ("deepseek-chat", "deepseek-reasoner", "gpt-5", "gpt-5-mini")


def stats_vector(stats: dict) -> list[float]:
    """Return a probability vector from count or percentage statistics."""
    counts = [float(stats.get(a.lower(), 0) or 0) for a in ACTIONS]
    total = sum(counts)
    if total > 0:
        return [x / total for x in counts]
    percentages = [float(stats.get(f"{a.lower()}_pct", 0) or 0) for a in ACTIONS]
    return [x / 100 for x in percentages]


def trajectory_vector(trajectory: str) -> tuple[list[float], int]:
    moves = [move for move in (trajectory or "").split() if move in ACTIONS]
    counts = [moves.count(action) for action in ACTIONS]
    return ([count / len(moves) for count in counts] if moves else [math.nan] * 3), len(moves)


def tv_distance(first: list[float], second: list[float]) -> float:
    return 0.5 * sum(abs(a - b) for a, b in zip(first, second))


def expected_markov_action(identity: str, opponent_previous: str) -> str | None:
    if opponent_previous not in ACTIONS:
        return None
    if identity == "X":
        return {"Rock": "Paper", "Paper": "Scissors", "Scissors": "Rock"}[opponent_previous]
    if identity == "Y":
        return {"Rock": "Scissors", "Paper": "Rock", "Scissors": "Paper"}[opponent_previous]
    if identity == "Z":
        return opponent_previous
    return None


def rule_metrics(identity: str, markov_trajectory: str, opponent_trajectory: str, required: int) -> tuple[int, int, int, float]:
    markov_moves = (markov_trajectory or "").split()
    opponent_moves = (opponent_trajectory or "").split()
    available = min(len(markov_moves), len(opponent_moves), required)
    evaluated = max(0, available - 1)
    matched = 0
    for index in range(1, available):
        expected = expected_markov_action(identity, opponent_moves[index - 1])
        matched += int(expected is not None and markov_moves[index] == expected)
    overlap = matched / evaluated if evaluated else math.nan
    complete = available >= required
    strict_match = int(complete and evaluated > 0 and matched == required - 1)
    return matched, evaluated, strict_match, overlap


def iter_records(root: Path):
    for path in root.rglob("*.json"):
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        if data.get("success"):
            yield path, data


def collect(root: Path, overlap_threshold: float) -> tuple[list[dict], dict[str, list[float]]]:
    detail_rows: list[dict] = []
    nonmarkov_tv: dict[str, list[float]] = defaultdict(list)

    for path, data in iter_records(root):
        model = data.get("model") or path.parts[-4]
        context = data.get("context") or {}
        generated = data.get("llm_simulation") or {}
        required = int(data.get("simulate_rounds") or data.get("capture_rounds") or 0)
        identities = {"p1": data.get("player1_id"), "p2": data.get("player2_id")}

        for side in ("p1", "p2"):
            identity = identities[side]
            target = stats_vector(context.get(f"{side}_stats") or {})
            surface, generated_rounds = trajectory_vector(generated.get(f"{side}_trajectory") or "")
            if generated_rounds == 0 or sum(target) == 0:
                continue
            tv = tv_distance(target, surface)
            if identity not in MARKOV_IDS:
                nonmarkov_tv[model].append(tv)
                continue

            other = "p2" if side == "p1" else "p1"
            matched, evaluated, strict, overlap = rule_metrics(
                identity,
                generated.get(f"{side}_trajectory") or "",
                generated.get(f"{other}_trajectory") or "",
                required,
            )
            predicted_identity = (generated.get(f"{side}_identity") or "").strip()
            detail_rows.append(
                {
                    "file": str(path),
                    "model": model,
                    "combo_type": data.get("combo_type"),
                    "markov_side": side,
                    "true_identity": identity,
                    "predicted_identity": predicted_identity,
                    "identity_wrong": int(predicted_identity != identity),
                    "generated_rounds": generated_rounds,
                    "required_rounds": required,
                    "rule_matches": matched,
                    "rule_rounds_evaluated": evaluated,
                    "strict_rule_match": strict,
                    "strict_rule_mismatch": 1 - strict,
                    "rule_overlap": overlap,
                    "low_rule_overlap": int(overlap < overlap_threshold),
                    "target_rock": target[0],
                    "target_paper": target[1],
                    "target_scissors": target[2],
                    "generated_rock": surface[0],
                    "generated_paper": surface[1],
                    "generated_scissors": surface[2],
                    "tv_distance": tv,
                }
            )
    return detail_rows, nonmarkov_tv


def masking_summary(
    rows: list[dict],
    nonmarkov_tv: dict[str, list[float]],
    tolerances: tuple[float, ...],
) -> list[dict]:
    pooled_reference = [value for values in nonmarkov_tv.values() for value in values]
    summaries: list[dict] = []
    definitions = (
        ("identity_wrong", "Identity wrong"),
        ("strict_rule_mismatch", "Strict rule mismatch"),
        ("low_rule_overlap", "Low rule overlap"),
    )
    models = [model for model in MODEL_ORDER if any(r["model"] == model for r in rows)]

    for key, label in definitions:
        for model in (*models, "Overall"):
            subset = rows if model == "Overall" else [r for r in rows if r["model"] == model]
            wrong = [r for r in subset if r[key] == 1]
            reference_values = pooled_reference if model == "Overall" else nonmarkov_tv.get(model, [])
            reference = statistics.median(reference_values) if reference_values else math.nan
            result = {
                "mechanism_definition": key,
                "mechanism_label": label,
                "model": model,
                "all_markov_cases": len(subset),
                "wrong_mechanism_cases": len(wrong),
                "wrong_mechanism_rate": len(wrong) / len(subset) if subset else math.nan,
            }
            for tolerance in tolerances:
                suffix = f"{tolerance:.2f}".replace(".", "_")
                count = sum(r["tv_distance"] <= tolerance for r in wrong)
                result[f"masked_tv_le_{suffix}_n"] = count
                result[f"masking_rate_tv_le_{suffix}"] = count / len(wrong) if wrong else math.nan
            reference_count = sum(r["tv_distance"] <= reference for r in wrong)
            result.update(
                {
                    "nonmarkov_median_tv_reference": reference,
                    "masked_by_nonmarkov_median_n": reference_count,
                    "masking_rate_by_nonmarkov_median": reference_count / len(wrong) if wrong else math.nan,
                }
            )
            summaries.append(result)
    return summaries


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pct(value: float) -> str:
    return "NA" if math.isnan(value) else f"{value:.1%}"


def write_report(path: Path, summaries: list[dict], tolerances: tuple[float, ...], overlap_threshold: float) -> None:
    lines = [
        "# Experiment 2: masking analysis",
        "",
        "A masking case is a Markov generation with an incorrect mechanism but a generated Markov-player marginal distribution close to its target context marginal. Surface distance is total variation (TV): 0.5 × Σ|generated − target|.",
        "",
        f"Mechanism definitions are reported separately: incorrect predicted identity; failure of strict rule execution; and rule overlap below {overlap_threshold:.0%}. Strict matching requires every evaluable generated round to follow the true Markov rule and the requested trajectory to be complete.",
    ]
    for mechanism in ("identity_wrong", "strict_rule_mismatch", "low_rule_overlap"):
        selected = [row for row in summaries if row["mechanism_definition"] == mechanism]
        lines.extend(
            [
                "",
                f"## {selected[0]['mechanism_label']}",
                "",
                "| Model | Markov cases | Wrong mechanism | "
                + " | ".join(f"TV ≤ {tolerance:.2f}" for tolerance in tolerances)
                + " | ≤ Non-Markov median TV |",
                "|:--|--:|--:|" + "--:|" * (len(tolerances) + 1),
            ]
        )
        for row in selected:
            threshold_cells = []
            for tolerance in tolerances:
                suffix = f"{tolerance:.2f}".replace(".", "_")
                threshold_cells.append(
                    f"{row[f'masked_tv_le_{suffix}_n']} ({pct(row[f'masking_rate_tv_le_{suffix}'])})"
                )
            lines.append(
                f"| {row['model']} | {row['all_markov_cases']} | {row['wrong_mechanism_cases']} "
                f"({pct(row['wrong_mechanism_rate'])}) | " + " | ".join(threshold_cells)
                + f" | {row['masked_by_nonmarkov_median_n']} ({pct(row['masking_rate_by_nonmarkov_median'])}); "
                f"threshold={row['nonmarkov_median_tv_reference']:.3f} |"
            )
    lines.extend(
        [
            "",
            "Notes: masking-rate denominators are wrong-mechanism cases, not all Markov cases. The Non-Markov reference is model-specific; the Overall row uses the pooled Non-Markov median TV. Case-level values are in masking_detail.csv.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-root", type=Path, default=Path("exp2(generation_blind)/generation"))
    parser.add_argument("--output-dir", type=Path, default=Path("exp2(generation_blind)/analysis_results/masking"))
    parser.add_argument("--rule-overlap-threshold", type=float, default=0.95)
    parser.add_argument("--tv-tolerances", type=float, nargs="+", default=DEFAULT_TOLERANCES)
    args = parser.parse_args()
    tolerances = tuple(sorted(set(args.tv_tolerances)))
    rows, references = collect(args.generation_root, args.rule_overlap_threshold)
    if not rows:
        raise RuntimeError(f"No successful Markov generation cases found under {args.generation_root}")
    summaries = masking_summary(rows, references, tolerances)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "masking_detail.csv", rows)
    write_csv(args.output_dir / "masking_summary.csv", summaries)
    write_report(args.output_dir / "report.md", summaries, tolerances, args.rule_overlap_threshold)
    print(f"Analyzed {len(rows)} Markov cases; wrote results to {args.output_dir}")


if __name__ == "__main__":
    main()
