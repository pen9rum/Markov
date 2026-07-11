"""Approximate teacher-forced transition accuracy from generation round 1.

For each Exp2 case containing one Markov player, score that player's first
generated action against the action implied by the opponent's final action in
the controlled context.  This transition is not affected by free-running
generation drift.
"""

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GENERATION_ROOT = ROOT / "exp2(generation_blind)" / "generation"
OUTPUT_ROOT = ROOT / "exp2(generation_blind)" / "analysis_results"
MODELS = ("deepseek-chat", "deepseek-reasoner", "gpt-5", "gpt-5-mini")
MARKOV_IDS = {"X", "Y", "Z"}


def expected_action(strategy_id, opponent_previous):
    win = {"Rock": "Paper", "Paper": "Scissors", "Scissors": "Rock"}
    lose = {"Rock": "Scissors", "Paper": "Rock", "Scissors": "Paper"}
    if strategy_id == "X":
        return win.get(opponent_previous)
    if strategy_id == "Y":
        return lose.get(opponent_previous)
    if strategy_id == "Z":
        return opponent_previous if opponent_previous in win else None
    return None


def trajectory(value):
    if isinstance(value, list):
        return value
    return (value or "").split()


def wilson_interval(correct, total, z=1.959963984540054):
    if not total:
        return None, None
    p = correct / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return centre - margin, centre + margin


def score_file(path, model):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not data.get("success"):
        return None

    p1_id, p2_id = data.get("player1_id"), data.get("player2_id")
    if (p1_id in MARKOV_IDS) == (p2_id in MARKOV_IDS):
        return None

    markov_side = "p1" if p1_id in MARKOV_IDS else "p2"
    markov_id = p1_id if markov_side == "p1" else p2_id
    opponent_side = "p2" if markov_side == "p1" else "p1"
    context = data.get("context") or {}
    generated = data.get("llm_simulation") or {}
    context_moves = trajectory(context.get(f"{opponent_side}_trajectory"))
    generated_moves = trajectory(generated.get(f"{markov_side}_trajectory"))
    if not context_moves or not generated_moves:
        return None

    previous = context_moves[-1]
    actual = generated_moves[0]
    expected = expected_action(markov_id, previous)
    if expected is None:
        return None
    predicted_id = generated.get(f"{markov_side}_identity") or ""
    return {
        "model": model,
        "file": str(path.relative_to(ROOT)),
        "player1_id": p1_id,
        "player2_id": p2_id,
        "markov_side": markov_side,
        "markov_id": markov_id,
        "predicted_markov_id": predicted_id,
        "identity_correct": int(predicted_id == markov_id),
        "context_opponent_last_action": previous,
        "expected_first_action": expected,
        "generated_first_action": actual,
        "correct": int(actual == expected),
    }


def main():
    rows = []
    for model in MODELS:
        for path in sorted((GENERATION_ROOT / model).rglob("*.json")):
            row = score_file(path, model)
            if row is not None:
                rows.append(row)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    detail_path = OUTPUT_ROOT / "first_transition_cases.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summaries = []
    for model in MODELS:
        model_rows = [row for row in rows if row["model"] == model]
        correct = sum(row["correct"] for row in model_rows)
        low, high = wilson_interval(correct, len(model_rows))
        identified = [row for row in model_rows if row["identity_correct"]]
        summaries.append({
            "model": model,
            "correct": correct,
            "n": len(model_rows),
            "accuracy": correct / len(model_rows) if model_rows else "",
            "ci95_low": low if low is not None else "",
            "ci95_high": high if high is not None else "",
            "identity_correct_n": len(identified),
            "identity_correct_transition_correct": sum(row["correct"] for row in identified),
            "identity_correct_transition_accuracy": (
                sum(row["correct"] for row in identified) / len(identified) if identified else ""
            ),
        })

    summary_path = OUTPUT_ROOT / "first_transition_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    for row in summaries:
        print(
            f'{row["model"]}: {row["correct"]}/{row["n"]} = '
            f'{100 * row["accuracy"]:.2f}% '
            f'(95% CI {100 * row["ci95_low"]:.2f}-{100 * row["ci95_high"]:.2f}%)'
        )
    print(f"Wrote {detail_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
